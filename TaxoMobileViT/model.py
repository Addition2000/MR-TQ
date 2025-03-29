from typing import Optional, Tuple, Union, Dict, List
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import os

from model_config import get_config


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvLayer(nn.Module):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        skip_connection: Optional[bool] = True,
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileViTBlock(nn.Module):


    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int = 2,
        head_dim: int = 32,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        patch_h: int = 8,
        patch_w: int = 8,
        conv_ksize: Optional[int] = 3,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False
        )

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        conv_3x3_out = ConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


class MultiHeadAttention(nn.Module):


    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        bias: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = nn.Linear(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x_q: Tensor) -> Tensor:
        # [N, P, C]
        b_sz, n_patches, in_channels = x_q.shape

        # self-attention
        # [N, P, C] -> [N, P, 3C] -> [N, P, 3, h, c] where C = hc
        qkv = self.qkv_proj(x_q).reshape(b_sz, n_patches, 3, self.num_heads, -1)

        # [N, P, 3, h, c] -> [N, h, 3, P, C]
        qkv = qkv.transpose(1, 3).contiguous()

        # [N, h, 3, P, C] -> [N, h, P, C] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling

        # [N h, P, c] -> [N, h, c, P]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, P, c] x [N, h, c, P] -> [N, h, P, P]
        attn = torch.matmul(query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, P, P] x [N, h, P, c] -> [N, h, P, c]
        out = torch.matmul(attn, value)

        # [N, h, P, c] -> [N, P, h, c] -> [N, P, C]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out


class TransformerEncoder(nn.Module):


    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        *args,
        **kwargs
    ) -> None:

        super().__init__()

        attn_unit = MultiHeadAttention(
            embed_dim,
            num_heads,
            attn_dropout=attn_dropout,
            bias=True
        )

        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim),
            attn_unit,
            nn.Dropout(p=dropout)
        )

        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout)
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        # multi-head attention
        res = x
        x = self.pre_norm_mha(x)
        x = x + res

        # feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class TaxoBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_classes_family: int,
        num_classes_genus: int,
        num_classes_species: int,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        
        # Family-level classifier
        self.family_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes_family)
        )
        
        # Genus-level classifier (conditioned on family features)
        self.genus_classifier = nn.Sequential(
            nn.Linear(embed_dim + num_classes_family, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes_genus)
        )
        
        # Species-level classifier (conditioned on family and genus features)
        self.species_classifier = nn.Sequential(
            nn.Linear(embed_dim + num_classes_family + num_classes_genus, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes_species)
        )
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: Tensor, dataset=None) -> tuple:
        # Predict family-level probabilities
        family_logits = self.family_classifier(x)
        family_probs = self.softmax(family_logits)
        
        # Concatenate family probabilities with original features to predict genus probabilities
        genus_input = torch.cat([x, family_probs], dim=1)
        genus_logits = self.genus_classifier(genus_input)
        genus_probs = self.softmax(genus_logits)
        
        # Concatenate family and genus probabilities with original features to predict species probabilities
        species_input = torch.cat([x, family_probs, genus_probs], dim=1)
        species_logits = self.species_classifier(species_input)
        species_probs = self.softmax(species_logits)
        
        return family_probs, genus_probs, species_probs


class MobileViT(nn.Module):

    def __init__(self, model_cfg: Dict, num_classes: int = 1000):
        super().__init__()

        image_channels = 3
        out_channels = 16

        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])

        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            in_channels=out_channels,
            out_channels=exp_channels,
            kernel_size=1
        )

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name="flatten", module=nn.Flatten())
        if 0.0 < model_cfg["cls_dropout"] < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=model_cfg["cls_dropout"]))
        self.classifier.add_module(name="fc", module=nn.Linear(in_features=exp_channels, out_features=num_classes))

        # weight init
        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)

        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        x = self.classifier(x)
        return x


def mobile_vit_xx_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt
    config = get_config("xx_small")
    m = MobileViT(config, num_classes=num_classes)
    return m


def mobile_vit_x_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt
    config = get_config("x_small")
    m = MobileViT(config, num_classes=num_classes)
    return m


def mobile_vit_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt
    config = get_config("small")
    m = MobileViT(config, num_classes=num_classes)
    return m


class HierarchicalMobileViT(nn.Module):

    def __init__(self, model_cfg: Dict, num_family_classes: int, num_genus_classes: int, num_species_classes: int):
        super().__init__()
        
        # Shared frontend feature extraction layers
        image_channels = 3
        out_channels = 16
        
        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )
        
        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        
        # Family classification branch 
        self.family_layer_3, f_out = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.family_layer_4, f_out = self._make_layer(input_channel=f_out, cfg=model_cfg["layer4"])
        exp_channels = min(model_cfg["last_layer_exp_factor"] * f_out, 960)
        self.family_conv_1x1_exp = ConvLayer(
            in_channels=f_out,
            out_channels=exp_channels,
            kernel_size=1
        )
        self.family_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=model_cfg["cls_dropout"]) if 0.0 < model_cfg["cls_dropout"] < 1.0 else nn.Identity(),
            nn.Linear(in_features=exp_channels, out_features=num_family_classes)
        )
        
        # Main branch layer_3
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        
        # Genus classification branch 
        self.genus_layer_4, g_out = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.genus_conv_1x1_exp = ConvLayer(
            in_channels=g_out,
            out_channels=exp_channels,
            kernel_size=1
        )
        self.genus_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=model_cfg["cls_dropout"]) if 0.0 < model_cfg["cls_dropout"] < 1.0 else nn.Identity(),
            nn.Linear(in_features=exp_channels, out_features=num_genus_classes)
        )
        
        # Species classification branch 
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])
        self.conv_1x1_exp = ConvLayer(
            in_channels=out_channels,
            out_channels=exp_channels,
            kernel_size=1
        )
        self.species_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=model_cfg["cls_dropout"]) if 0.0 < model_cfg["cls_dropout"] < 1.0 else nn.Identity(),
            nn.Linear(in_features=exp_channels, out_features=num_species_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)
    
    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Shared frontend feature extraction
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        
        # Family classification branch
        family_feat = x.clone()  # Use clone to avoid modifying original features
        family_feat = self.family_layer_3(family_feat)
        family_feat = self.family_layer_4(family_feat)  # Add layer_4
        family_feat = self.family_conv_1x1_exp(family_feat)
        family_out = self.family_classifier(family_feat)
        
        # Main branch continues
        x = self.layer_3(x)
        
        # Genus classification branch
        genus_feat = x.clone()  # Use clone to avoid modifying original features
        genus_feat = self.genus_layer_4(genus_feat)
        genus_feat = self.genus_conv_1x1_exp(genus_feat)
        genus_out = self.genus_classifier(genus_feat)
        
        # Species classification branch (main branch continues)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        species_out = self.species_classifier(x)
        
        return family_out, genus_out, species_out

def create_hierarchical_mobilevit(num_family_classes: int, num_genus_classes: int, num_species_classes: int, pretrained_path: str = None):

    config = get_config("small")  # Use small configuration
    m = HierarchicalMobileViT(
        config,
        num_family_classes=num_family_classes,
        num_genus_classes=num_genus_classes,
        num_species_classes=num_species_classes
    )
    
    if pretrained_path is not None and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights: {pretrained_path}")
        # Load pretrained weights
        pretrained_dict = torch.load(pretrained_path)
        
        # If official pretrained weights, process keys
        if 'model_state_dict' not in pretrained_dict:
            # Load shared layers and main branch weights only
            model_dict = m.state_dict()
            
            # Create new weights dictionary
            new_pretrained_dict = {}
            
            # Iterate over pretrained weights
            for k, v in pretrained_dict.items():
                # Skip classifier layers
                if 'classifier' in k:
                    continue
                
                # Process main branch weights
                if k in model_dict:
                    if v.size() == model_dict[k].size():
                        new_pretrained_dict[k] = v
                    else:
                        # Special handling for conv_1x1_exp layer
                        if 'conv_1x1_exp.block.conv.weight' in k:
                            target_shape = model_dict[k].size()
                            if len(target_shape) == 4:  # Ensure 4D tensor
                                if v.size(0) > target_shape[0]:
                                    new_pretrained_dict[k] = v[:target_shape[0], :target_shape[1], :, :]
                                else:
                                    new_tensor = torch.zeros_like(model_dict[k])
                                    new_tensor[:v.size(0), :v.size(1), :, :] = v
                                    new_pretrained_dict[k] = new_tensor
                        elif 'conv_1x1_exp.block' in k and ('norm.weight' in k or 'norm.bias' in k or 'norm.running' in k):
                            target_shape = model_dict[k].size()
                            if v.size(0) > target_shape[0]:
                                new_pretrained_dict[k] = v[:target_shape[0]]
                            else:
                                new_tensor = torch.zeros_like(model_dict[k])
                                new_tensor[:v.size(0)] = v
                                new_pretrained_dict[k] = new_tensor
                
                # Process family branch weights
                family_key = k.replace('layer_3', 'family_layer_3').replace('layer_4', 'family_layer_4')
                if family_key in model_dict:
                    if v.size() == model_dict[family_key].size():
                        new_pretrained_dict[family_key] = v.clone()
                    else:
                        if 'conv_1x1_exp.block.conv.weight' in family_key:
                            target_shape = model_dict[family_key].size()
                            if len(target_shape) == 4:
                                if v.size(0) > target_shape[0]:
                                    new_pretrained_dict[family_key] = v[:target_shape[0], :target_shape[1], :, :]
                                else:
                                    new_tensor = torch.zeros_like(model_dict[family_key])
                                    new_tensor[:v.size(0), :v.size(1), :, :] = v
                                    new_pretrained_dict[family_key] = new_tensor
                        elif 'conv_1x1_exp.block' in family_key and ('norm.weight' in k or 'norm.bias' in k or 'norm.running' in k):
                            target_shape = model_dict[family_key].size()
                            if v.size(0) > target_shape[0]:
                                new_pretrained_dict[family_key] = v[:target_shape[0]]
                            else:
                                new_tensor = torch.zeros_like(model_dict[family_key])
                                new_tensor[:v.size(0)] = v
                                new_pretrained_dict[family_key] = new_tensor
                
                # Process genus branch weights
                genus_key = k.replace('layer_4', 'genus_layer_4')
                if genus_key in model_dict:
                    if v.size() == model_dict[genus_key].size():
                        new_pretrained_dict[genus_key] = v.clone()
                    else:
                        if 'conv_1x1_exp.block.conv.weight' in genus_key:
                            target_shape = model_dict[genus_key].size()
                            if len(target_shape) == 4:
                                if v.size(0) > target_shape[0]:
                                    new_pretrained_dict[genus_key] = v[:target_shape[0], :target_shape[1], :, :]
                                else:
                                    new_tensor = torch.zeros_like(model_dict[genus_key])
                                    new_tensor[:v.size(0), :v.size(1), :, :] = v
                                    new_pretrained_dict[genus_key] = new_tensor
                        elif 'conv_1x1_exp.block' in genus_key and ('norm.weight' in k or 'norm.bias' in k or 'norm.running' in k):
                            target_shape = model_dict[genus_key].size()
                            if v.size(0) > target_shape[0]:
                                new_pretrained_dict[genus_key] = v[:target_shape[0]]
                            else:
                                new_tensor = torch.zeros_like(model_dict[genus_key])
                                new_tensor[:v.size(0)] = v
                                new_pretrained_dict[genus_key] = new_tensor
            
            # Update model weights
            model_dict.update(new_pretrained_dict)
            m.load_state_dict(model_dict, strict=False)
            
            print("Pretrained weights loaded successfully")
        else:
            # If our own saved checkpoint
            m.load_state_dict(pretrained_dict['model_state_dict'])
            print("Checkpoint loaded successfully")
    else:
        print("Using random initialization for weights")
    
    return m

