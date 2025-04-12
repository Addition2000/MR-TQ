import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class Dataset(Dataset):
    def __init__(self, root_dir, annotation_file, txt_path=None, transform=None, split='train', image_size=224):
        self.root_dir = root_dir
        self.split = split
        
        # Set transform
        if transform is not None:
            self.transform = transform
        else:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                ])
            else:  # Validation set
                self.transform = transforms.Compose([
                    transforms.Resize(int(image_size * 1.143)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                ])

        try:
            self.annotations = pd.read_csv(annotation_file, sep='\t')
            self.family_to_idx = {family: idx for idx, family in enumerate(sorted(self.annotations['family'].unique()))}
            self.genus_to_idx = {genus: idx for idx, genus in enumerate(sorted(self.annotations['genus'].unique()))}
            self.species_to_idx = {species: idx for idx, species in enumerate(sorted(self.annotations['species'].unique()))}
            
            self.idx_to_family = {idx: family for family, idx in self.family_to_idx.items()}
            self.idx_to_genus = {idx: genus for genus, idx in self.genus_to_idx.items()}
            self.idx_to_species = {idx: species for species, idx in self.species_to_idx.items()}
        except Exception as e:
            raise Exception(f"Error reading annotation file {annotation_file}: {e}")

        if txt_path is not None:
            self.images = []
            self.labels = []
            
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                img_path, species_idx = line.strip().split('\t')
                img_path = os.path.join(root_dir, img_path)
                species_idx = int(species_idx)
                
                species_name = self.idx_to_species.get(species_idx)
                if species_name is None:
                    print(f"Warning: Unknown species index {species_idx} for image {img_path}")
                    continue
                
                species_info = self.annotations[self.annotations['species'] == species_name].iloc[0]
                family_idx = self.family_to_idx[species_info['family']]
                genus_idx = self.genus_to_idx[species_info['genus']]
                
                self.images.append(img_path)
                self.labels.append((family_idx, genus_idx, species_idx))
            
            print(f"\n=== {split.upper()} Set Information ===")
            print(f"Number of images: {len(self.images)}")
            print(f"Number of families: {self.num_families}")
            print(f"Number of genera: {self.num_genera}")
            print(f"Number of species: {self.num_species}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        family_idx, genus_idx, species_idx = self.labels[idx]
        
        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise
        
        if self.transform:
            image = self.transform(image)
        
        return image, (
            torch.tensor(family_idx, dtype=torch.long),
            torch.tensor(genus_idx, dtype=torch.long),
            torch.tensor(species_idx, dtype=torch.long)
        )
    
    @property
    def num_families(self):
        return len(self.family_to_idx)
    
    @property
    def num_genera(self):
        return len(self.genus_to_idx)
    
    @property
    def num_species(self):
        return len(self.species_to_idx)

def get_data_loaders(root_dir, annotation_file, train_txt, val_txt, batch_size=32, num_workers=4, image_size=224):
    train_dataset = FrogDataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        txt_path=train_txt,
        split='train',
        image_size=image_size
    )
    
    val_dataset = FrogDataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        txt_path=val_txt,
        split='val',
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.num_families, train_dataset.num_genera, train_dataset.num_species
