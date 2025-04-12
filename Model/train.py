import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
from model import create_hierarchical_mobilevit, TaxoBlock
from dataset import get_data_loaders
import torch.backends.cudnn as cudnn
from datetime import datetime

# Training parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 300
DATA_ROOT = r"your DATA_ROOT path"

# GPU settings
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
if CUDA:
    cudnn.benchmark = True  # Enable automatic optimization
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

def get_metrics(y_true, y_pred):
    """Calculate various evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    
    # Use weighted average instead of macro average, considering class imbalance
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate precision for each class for detailed analysis
    class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    valid_precisions = class_precision[class_precision != 0]
    avg_valid_precision = valid_precisions.mean() if len(valid_precisions) > 0 else 0
    
    return accuracy, precision, recall, f1

def train(model, train_loader, val_loader, num_epochs, device):
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    run_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    
    writer = SummaryWriter('runs/' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    # Create optimizer
    optimizer = AdamW([
        {'params': model.parameters()}
    ], lr=1e-4, weight_decay=5e-5)
    
    # Loss weights - can be adjusted based on task importance
    loss_weights = {
        'family': 0.2,  # Family classification weight
        'genus': 0.3,   # Genus classification weight
        'species': 0.5  # Species classification weight
    }
    
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    
    # Create TaxoBlock instance
    taxo_block = TaxoBlock(
        embed_dim=1000,  # Using feature dimension from model output
        num_classes_family=len(train_loader.dataset.family_to_idx),
        num_classes_genus=len(train_loader.dataset.genus_to_idx),
        num_classes_species=len(train_loader.dataset.species_to_idx),
        dropout=0.2
    ).to(device)
    
    for epoch in range(num_epochs):
        model.train()
        taxo_block.train()  # Ensure TaxoBlock is in training mode
        train_preds = {'family': [], 'genus': [], 'species': [], 'final': []}
        train_labels = {'family': [], 'genus': [], 'species': [], 'final': []}
        train_losses = {'family': 0.0, 'genus': 0.0, 'species': 0.0, 'total': 0.0, 'final': 0.0}
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            images, (family_labels, genus_labels, species_labels) = batch
            images = images.to(device)
            family_labels = family_labels.to(device)
            genus_labels = genus_labels.to(device)
            species_labels = species_labels.to(device)
            
            # Forward pass
            family_pred, genus_pred, species_pred = model(images)
            
            # Calculate weighted losses
            family_loss = criterion(family_pred, family_labels) * loss_weights['family']
            genus_loss = criterion(genus_pred, genus_labels) * loss_weights['genus']
            species_loss = criterion(species_pred, species_labels) * loss_weights['species']
            
            # Calculate total loss
            total_loss = family_loss + genus_loss + species_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Calculate probabilities
            family_prob = torch.softmax(family_pred, dim=1)
            genus_prob = torch.softmax(genus_pred, dim=1)
            species_prob = torch.softmax(species_pred, dim=1)
            
            # Use TaxoBlock to calculate final hierarchical probabilities
            final_family_prob, final_genus_prob, final_species_prob = taxo_block(
                torch.cat([family_prob, genus_prob, species_prob], dim=1), 
                train_loader.dataset
            )
            
            # Calculate final loss (using species_labels as target)
            final_loss = criterion(torch.log(final_species_prob + 1e-8), species_labels)
            
            # Record prediction results
            train_preds['family'].extend(torch.argmax(family_pred, dim=1).cpu().numpy())
            train_preds['genus'].extend(torch.argmax(genus_pred, dim=1).cpu().numpy())
            train_preds['species'].extend(torch.argmax(species_pred, dim=1).cpu().numpy())
            train_preds['final'].extend(torch.argmax(final_species_prob, dim=1).cpu().numpy())
            
            train_labels['family'].extend(family_labels.cpu().numpy())
            train_labels['genus'].extend(genus_labels.cpu().numpy())
            train_labels['species'].extend(species_labels.cpu().numpy())
            train_labels['final'].extend(species_labels.cpu().numpy())
            
            train_losses['family'] += family_loss.item()
            train_losses['genus'] += genus_loss.item()
            train_losses['species'] += species_loss.item()
            train_losses['total'] += total_loss.item()
            train_losses['final'] += final_loss.item()
            
            pbar.set_postfix({
                'total_loss': total_loss.item(),
                'family_loss': family_loss.item(),
                'genus_loss': genus_loss.item(),
                'species_loss': species_loss.item(),
                'final_loss': final_loss.item()
            })
        
        # Calculate training metrics
        for level in ['family', 'genus', 'species', 'final']:
            acc, prec, rec, f1 = get_metrics(
                train_labels[level], 
                train_preds[level]
            )
            
            writer.add_scalar(f'Loss/train_{level}', train_losses[level]/len(train_loader), epoch)
            writer.add_scalar(f'Accuracy/train_{level}', acc, epoch)
            
            print(f'\nTraining {level} metrics:')
            print(f'Accuracy: {acc:.4f}, Precision: {prec:.4f}')
            print(f'Recall: {rec:.4f}, F1-score: {f1:.4f}')
        
        # Validation
        model.eval()
        taxo_block.eval()  # Ensure TaxoBlock is in evaluation mode
        val_preds = {'family': [], 'genus': [], 'species': [], 'final': []}
        val_labels = {'family': [], 'genus': [], 'species': [], 'final': []}
        val_losses = {'family': 0.0, 'genus': 0.0, 'species': 0.0, 'total': 0.0, 'final': 0.0}
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}')
            for batch in val_pbar:
                images, (family_labels, genus_labels, species_labels) = batch
                images = images.to(device)
                family_labels = family_labels.to(device)
                genus_labels = genus_labels.to(device)
                species_labels = species_labels.to(device)
                
                # Forward pass
                family_pred, genus_pred, species_pred = model(images)
                
                # Calculate losses
                family_loss = criterion(family_pred, family_labels)
                genus_loss = criterion(genus_pred, genus_labels)
                species_loss = criterion(species_pred, species_labels)
                total_loss = family_loss + genus_loss + species_loss
                
                # Calculate probabilities
                family_prob = torch.softmax(family_pred, dim=1)
                genus_prob = torch.softmax(genus_pred, dim=1)
                species_prob = torch.softmax(species_pred, dim=1)
                
                # Use TaxoBlock to calculate hierarchical probabilities
                final_family_prob, final_genus_prob, final_species_prob = taxo_block(
                    torch.cat([family_prob, genus_prob, species_prob], dim=1), 
                    val_loader.dataset
                )
                
                # Calculate final loss
                final_loss = criterion(torch.log(final_species_prob + 1e-8), species_labels)
                
                # Record prediction results
                val_preds['family'].extend(torch.argmax(family_pred, dim=1).cpu().numpy())
                val_preds['genus'].extend(torch.argmax(genus_pred, dim=1).cpu().numpy())
                val_preds['species'].extend(torch.argmax(species_pred, dim=1).cpu().numpy())
                val_preds['final'].extend(torch.argmax(final_species_prob, dim=1).cpu().numpy())
                
                val_labels['family'].extend(family_labels.cpu().numpy())
                val_labels['genus'].extend(genus_labels.cpu().numpy())
                val_labels['species'].extend(species_labels.cpu().numpy())
                val_labels['final'].extend(species_labels.cpu().numpy())
                
                val_losses['family'] += family_loss.item()
                val_losses['genus'] += genus_loss.item()
                val_losses['species'] += species_loss.item()
                val_losses['total'] += total_loss.item()
                val_losses['final'] += final_loss.item()
                
                val_pbar.set_postfix({
                    'total_loss': total_loss.item(),
                    'family_loss': family_loss.item(),
                    'genus_loss': genus_loss.item(),
                    'species_loss': species_loss.item(),
                    'final_loss': final_loss.item()
                })
        
        # Calculate validation metrics
        val_final_acc = 0
        metrics = {}
        for level in ['family', 'genus', 'species', 'final']:
            acc, prec, rec, f1 = get_metrics(
                val_labels[level], 
                val_preds[level]
            )
            
            metrics[f'{level}_acc'] = acc
            metrics[f'{level}_prec'] = prec
            metrics[f'{level}_rec'] = rec
            metrics[f'{level}_f1'] = f1
            
            if level == 'final':
                val_final_acc = acc
            
            writer.add_scalar(f'Loss/val_{level}', val_losses[level]/len(val_loader), epoch)
            writer.add_scalar(f'Accuracy/val_{level}', acc, epoch)
            writer.add_scalar(f'Precision/val_{level}', prec, epoch)
            writer.add_scalar(f'Recall/val_{level}', rec, epoch)
            writer.add_scalar(f'F1/val_{level}', f1, epoch)
            
            print(f'\nValidation {level} metrics:')
            print(f'Accuracy: {acc:.4f}, Precision: {prec:.4f}')
            print(f'Recall: {rec:.4f}, F1-score: {f1:.4f}')
        
        # Save model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'taxo_block_state_dict': taxo_block.state_dict(),  # Save TaxoBlock state
            'metrics': metrics
        }
        
        # Save current epoch model
        epoch_save_path = os.path.join(run_dir, f'epoch_{epoch+1}.pth')
        torch.save(checkpoint, epoch_save_path)
        
        # If best model, save an additional copy
        if val_final_acc > best_val_acc:
            best_val_acc = val_final_acc
            best_model_path = os.path.join(run_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"\nSaved best model with validation accuracy: {val_final_acc:.4f}")
        
        print(f"\nSaved Epoch {epoch+1} model to: {epoch_save_path}")

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists("./weights"):
        os.makedirs("./weights")
    
    # Get data loaders and class counts
    train_loader, val_loader, num_families, num_genera, num_species = get_data_loaders(
        root_dir=args.data_path,
        annotation_file=os.path.join(args.data_path, "annotations.txt"),
        train_txt=os.path.join(args.data_path, "train.txt"),
        val_txt=os.path.join(args.data_path, "val.txt"),
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])
    )

    # Create hierarchical classification model
    model = create_hierarchical_mobilevit(
        num_family_classes=num_families,
        num_genus_classes=num_genera,
        num_species_classes=num_species,
        pretrained_path=args.pretrained
    ).to(device)
    
    print("Created hierarchical classification model")
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    total_params_mb = total_params / 1e6
    print(f"Total model parameters: {total_params_mb:.2f} MB")

    # Train model
    train(model, train_loader, val_loader, args.epochs, device)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--data-path', type=str, default=DATA_ROOT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pretrained', type=str, default='mobilevit_s.pt')
    
    opt = parser.parse_args()
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU index: {torch.cuda.current_device()}")
    
    main(opt)