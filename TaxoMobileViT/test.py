import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont

from model import create_hierarchical_mobilevit, TaxoBlock
from dataset import get_data_loaders
from train import get_metrics

# Test parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = r"your MODEL_PATH path"  # Replace with your model path
DATA_ROOT = r"your DATA_ROO path"  # Replace with your dataset path

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    sns.heatmap(cm_normalized, annot=False, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_misclassified_samples(model, test_loader, device, num_samples=10, 
                                   save_dir='misclassified_samples'):
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    misclassified = []
    
    # Create TaxoBlock instance
    taxo_block = TaxoBlock(
        embed_dim=1000,  # Using feature dimension from model output
        num_classes_family=len(test_loader.dataset.family_to_idx),
        num_classes_genus=len(test_loader.dataset.genus_to_idx),
        num_classes_species=len(test_loader.dataset.species_to_idx),
        dropout=0.2
    ).to(device)
    taxo_block.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Finding misclassified samples"):
            images, (family_labels, genus_labels, species_labels) = batch
            images = images.to(device)
            species_labels = species_labels.to(device)
            
            # Forward pass
            family_pred, genus_pred, species_pred = model(images)
            
            # Calculate probabilities
            family_prob = torch.softmax(family_pred, dim=1)
            genus_prob = torch.softmax(genus_pred, dim=1)
            species_prob = torch.softmax(species_pred, dim=1)
            
            # Use TaxoBlock to calculate hierarchical probabilities
            _, _, final_probs = taxo_block(
                torch.cat([family_prob, genus_prob, species_prob], dim=1),
                test_loader.dataset
            )
            
            final_preds = torch.argmax(final_probs, dim=1)
            
            # Find misclassified samples
            incorrect_mask = (final_preds != species_labels)
            incorrect_indices = torch.where(incorrect_mask)[0]
            
            for idx in incorrect_indices:
                if len(misclassified) >= num_samples:
                    break
                
                img = images[idx].cpu()
                true_label = species_labels[idx].item()
                pred_label = final_preds[idx].item()
                confidence = final_probs[idx][pred_label].item()
                
                true_species = test_loader.dataset.idx_to_species[true_label]
                pred_species = test_loader.dataset.idx_to_species[pred_label]
                
                misclassified.append({
                    'image': img,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': confidence,
                    'true_species': true_species,
                    'pred_species': pred_species
                })
            
            if len(misclassified) >= num_samples:
                break
    
    # Visualize misclassified samples
    for i, sample in enumerate(misclassified):
        img = sample['image']
        img = img.permute(1, 2, 0).numpy()  # Convert to HWC format
        img = (img * 255).astype(np.uint8)  # Restore to 0-255 range
        
        # Create image with labels
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        draw = ImageDraw.Draw(pil_img)
        
        # Set font, use default if not specified
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        text = f"True: {sample['true_species']}\nPred: {sample['pred_species']}\nConf: {sample['confidence']:.4f}"
        draw.text((10, 10), text, fill=(255, 0, 0), font=font)
        
        save_path = os.path.join(save_dir, f"misclassified_{i}.png")
        pil_img.save(save_path)
        print(f"Saved misclassified sample to {save_path}")

def test_model(model_path, data_root, device):

    # Create results directory
    results_dir = os.path.join(os.path.dirname(model_path), 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    _, test_loader, num_families, num_genera, num_species = get_data_loaders(
        root_dir=data_root,
        annotation_file=os.path.join(data_root, "annotations.txt"),
        train_txt=os.path.join(data_root, "train.txt"),
        val_txt=os.path.join(data_root, "test.txt"),  # Use test.txt or val.txt as test set
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])
    )
    
    # Create model
    print(f"Loading model: {model_path}")
    model = create_hierarchical_mobilevit(
        num_family_classes=num_families,
        num_genus_classes=num_genera,
        num_species_classes=num_species
    ).to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model checkpoint (Epoch {checkpoint['epoch']})")
    
    # Create TaxoBlock instance
    taxo_block = TaxoBlock(
        embed_dim=1000,  # Using feature dimension from model output
        num_classes_family=num_families,
        num_classes_genus=num_genera,
        num_classes_species=num_species,
        dropout=0.2
    ).to(device)
    
    # If checkpoint contains TaxoBlock state dict, load it
    if 'taxo_block_state_dict' in checkpoint:
        taxo_block.load_state_dict(checkpoint['taxo_block_state_dict'])
        print("Loaded TaxoBlock weights")
    
    # Test model
    model.eval()
    taxo_block.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_preds = {'family': [], 'genus': [], 'species': [], 'final': []}
    test_labels = {'family': [], 'genus': [], 'species': [], 'final': []}
    test_probs = {'family': [], 'genus': [], 'species': [], 'final': []}
    test_losses = {'family': 0.0, 'genus': 0.0, 'species': 0.0, 'total': 0.0}
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing progress")
        for batch in pbar:
            images, (family_labels, genus_labels, species_labels) = batch
            images = images.to(device)
            family_labels = family_labels.to(device)
            genus_labels = genus_labels.to(device)
            species_labels = species_labels.to(device)
            
            # Forward pass
            family_pred, genus_pred, species_pred = model(images)
            
            # Calculate probabilities
            family_prob = torch.softmax(family_pred, dim=1)
            genus_prob = torch.softmax(genus_pred, dim=1)
            species_prob = torch.softmax(species_pred, dim=1)
            
            # Use TaxoBlock to calculate hierarchical probabilities
            final_family_prob, final_genus_prob, final_species_prob = taxo_block(
                torch.cat([family_prob, genus_prob, species_prob], dim=1),
                test_loader.dataset
            )
            
            # Record prediction results
            test_preds['family'].extend(torch.argmax(family_prob, dim=1).cpu().numpy())
            test_preds['genus'].extend(torch.argmax(genus_prob, dim=1).cpu().numpy())
            test_preds['species'].extend(torch.argmax(species_prob, dim=1).cpu().numpy())
            test_preds['final'].extend(torch.argmax(final_species_prob, dim=1).cpu().numpy())
            
            test_labels['family'].extend(family_labels.cpu().numpy())
            test_labels['genus'].extend(genus_labels.cpu().numpy())
            test_labels['species'].extend(species_labels.cpu().numpy())
            test_labels['final'].extend(species_labels.cpu().numpy())
            
            # Save probability distributions for later analysis
            test_probs['family'].extend(family_prob.cpu().numpy())
            test_probs['genus'].extend(genus_prob.cpu().numpy())
            test_probs['species'].extend(species_prob.cpu().numpy())
            test_probs['final'].extend(final_species_prob.cpu().numpy())
            
            family_loss = criterion(family_prob, family_labels)
            genus_loss = criterion(genus_prob, genus_labels)
            species_loss = criterion(species_prob, species_labels)
            total_loss = family_loss + genus_loss + species_loss
            
            test_losses['family'] += family_loss.item()
            test_losses['genus'] += genus_loss.item()
            test_losses['species'] += species_loss.item()
            test_losses['total'] += total_loss.item()
            
            pbar.set_postfix({
                'total_loss': total_loss.item(),
                'family_loss': family_loss.item(),
                'genus_loss': genus_loss.item(),
                'species_loss': species_loss.item()
            })
    
    # Convert lists to numpy arrays for processing
    for level in ['family', 'genus', 'species', 'final']:
        test_probs[level] = np.array(test_probs[level])
    
    # Calculate evaluation metrics and generate report
    results = {}
    for level in ['family', 'genus', 'species', 'final']:
        # Basic metrics
        acc, prec, rec, f1 = get_metrics(test_labels[level], test_preds[level])
        
        print(f"\n{level.title()} classification performance:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        results[level] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'loss': test_losses[level] / len(test_loader)
        }
        
        # Get class names
        if level == 'family':
            class_names = [test_loader.dataset.idx_to_family[i] for i in range(len(test_loader.dataset.family_to_idx))]
        elif level == 'genus':
            class_names = [test_loader.dataset.idx_to_genus[i] for i in range(len(test_loader.dataset.genus_to_idx))]
        elif level in ['species', 'final']:
            class_names = [test_loader.dataset.idx_to_species[i] for i in range(len(test_loader.dataset.species_to_idx))]
        
        # Detailed classification report
        if len(class_names) <= 30:  # Too many classes might be difficult to display
            report = classification_report(
                test_labels[level], 
                test_preds[level],
                target_names=class_names,
                digits=4
            )
            print(f"\n{level.title()} classification report:")
            print(report)
            
            # Save report to file
            with open(os.path.join(results_dir, f"{level}_classification_report.txt"), "w") as f:
                f.write(f"{level.title()} Classification Report:\n")
                f.write(report)
        
        # Plot confusion matrix
        if len(class_names) <= 30:  # Too many classes might make confusion matrix unclear
            plot_confusion_matrix(
                test_labels[level],
                test_preds[level],
                class_names,
                level.title(),
                os.path.join(results_dir, f"{level}_confusion_matrix.png")
            )
    
    # Save test results summary
    results_df = pd.DataFrame({
        'Level': ['Family', 'Genus', 'Species', 'Final (Hierarchical)'],
        'Accuracy': [results[k]['accuracy'] for k in ['family', 'genus', 'species', 'final']],
        'Precision': [results[k]['precision'] for k in ['family', 'genus', 'species', 'final']],
        'Recall': [results[k]['recall'] for k in ['family', 'genus', 'species', 'final']],
        'F1-Score': [results[k]['f1'] for k in ['family', 'genus', 'species', 'final']],
        'Loss': [results[k]['loss'] for k in ['family', 'genus', 'species', 'final']]
    })
    
    results_df.to_csv(os.path.join(results_dir, "test_results_summary.csv"), index=False)
    print(f"\nTest results summary saved to {os.path.join(results_dir, 'test_results_summary.csv')}")
    
    # Visualize some misclassified samples
    print("\nVisualizing misclassified samples...")
    visualize_misclassified_samples(model, test_loader, device, 
                                   num_samples=20, 
                                   save_dir=os.path.join(results_dir, 'misclassified'))
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test mushroom classification model")
    parser.add_argument('--model-path', type=str, default=MODEL_PATH, help='Model checkpoint path')
    parser.add_argument('--data-path', type=str, default=DATA_ROOT, help='Dataset path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Test model
    results = test_model(args.model_path, args.data_path, device)
    
    print("\nTesting completed!")

if __name__ == "__main__":
    main()
