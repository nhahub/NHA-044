import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main():
    # Set device for GPU acceleration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Define your paths
    train_folder_path = r'C:\Users\Access\PythonImp\project_DEPI\augmented_train'
    val_folder_path = r'C:\Users\Access\PythonImp\project_DEPI\augmented_val'
    test_folder_path = r'C:\Users\Access\PythonImp\project_DEPI\augmented_test'
    
    # SIMPLIFIED DATASET CLASS
    class CustomDataset(Dataset):
        def __init__(self, folder_path, transform=None, is_training=True):
            self.image_paths = []
            self.labels = []
            self.transform = transform
            self.is_training = is_training
            
            # Get all image files and their labels
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(root, file)
                        self.image_paths.append(full_path)
                        
                        # Extract label from folder structure
                        parent_folder = os.path.basename(os.path.dirname(full_path))
                        self.labels.append(parent_folder)
            
            # Get unique classes and create mapping
            self.unique_labels = sorted(set(self.labels))
            self.class_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
            self.num_classes = len(self.unique_labels)
            
            # Convert labels to integers
            self.labels_int = [self.class_to_idx[label] for label in self.labels]
            
            print(f"Found {len(self.image_paths)} images with {self.num_classes} classes: {self.unique_labels}")
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            try:
                # Load image
                image = Image.open(self.image_paths[idx]).convert('RGB')
                
                # Apply transformations
                if self.transform:
                    image = self.transform(image)
                
                label = self.labels_int[idx]
                return image, label
                
            except Exception as e:
                print(f"Error loading {self.image_paths[idx]}: {e}")
                # Return a random valid item if there's an error
                return self.__getitem__(idx % len(self.image_paths))
    
    # SIMPLIFIED DATA TRANSFORMS WITHOUT AUGMENTATION
    def get_transforms():
        # Same transforms for all phases since you already augmented the data
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    print("\n" + "="*50)
    print("CREATING DATALOADERS")
    print("="*50)
    
    # Create datasets and dataloaders - same transforms for all
    train_dataset = CustomDataset(train_folder_path, transform=get_transforms(), is_training=True)
    val_dataset = CustomDataset(val_folder_path, transform=get_transforms(), is_training=False)
    test_dataset = CustomDataset(test_folder_path, transform=get_transforms(), is_training=False)
    
    batch_size = 32
    
    # Use 0 workers for Windows to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    num_classes = train_dataset.num_classes
    class_names = train_dataset.unique_labels
    
    print(f"\nDataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # IMPROVED MODEL ARCHITECTURE
    class ImprovedMobileNetV2(nn.Module):
        def __init__(self, num_classes):
            super(ImprovedMobileNetV2, self).__init__()
            
            # Load pre-trained MobileNetV2
            self.base_model = models.mobilenet_v2(pretrained=True)
            
            # Freeze base model layers initially
            for param in self.base_model.parameters():
                param.requires_grad = False
            
            # Get the number of features from the base model
            in_features = self.base_model.classifier[1].in_features
            
            # IMPROVED CLASSIFIER - SIMPLER AND MORE EFFECTIVE
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            return self.base_model(x)
        
        def unfreeze_layers(self, num_layers=80):
            """Unfreeze last few layers for fine-tuning"""
            # Unfreeze classifier first
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True
                
            # Unfreeze last num_layers of base model
            total_layers = len(list(self.base_model.features))
            layers_to_unfreeze = min(num_layers, total_layers)
            
            print(f"Unfreezing last {layers_to_unfreeze} layers of {total_layers} total layers")
            
            for i, layer in enumerate(self.base_model.features):
                if i >= total_layers - layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True
    
    print("\n" + "="*50)
    print("CREATING IMPROVED MODEL WITH REGULARIZATION")
    print("="*50)
    model = ImprovedMobileNetV2(num_classes).to(device)
    
    # Count trainable parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Test the model with a sample batch to ensure it works
    print("Testing model with sample batch...")
    model.eval()
    with torch.no_grad():
        sample_input = torch.randn(2, 3, 224, 224).to(device)
        sample_output = model(sample_input)
        print(f"Sample input shape: {sample_input.shape}")
        print(f"Sample output shape: {sample_output.shape}")
        print(f"Model test passed! Output shape: {sample_output.shape}")
    
    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    
    # IMPROVED TRAINING FUNCTION WITH GRADIENT CLIPPING
    def train_epoch(model, dataloader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_batches = len(dataloader)
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
            # Simple progress indicator
            if batch_idx % 20 == 0:
                current_acc = torch.sum(preds == labels.data).float() / inputs.size(0)
                print(f'  Batch {batch_idx}/{total_batches}, Loss: {loss.item():.4f}, Acc: {current_acc:.4f}')
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        
        return epoch_loss, epoch_acc.cpu().item()
    
    # VALIDATION FUNCTION
    def validate_epoch(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        
        return epoch_loss, epoch_acc.cpu().item()
    
    # TEST FUNCTION WITH CONFUSION MATRIX
    def test_model_with_confusion(model, dataloader, criterion, device, class_names):
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_loss = running_loss / len(dataloader.dataset)
        test_acc = running_corrects.double() / len(dataloader.dataset)
        
        return test_loss, test_acc.cpu().item(), all_preds, all_labels
    
    # CONFUSION MATRIX AND PER-CLASS ANALYSIS
    def plot_confusion_matrix_and_analysis(y_true, y_pred, class_names, filename='confusion_matrix.png'):
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Confusion Matrix
        im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xticks(np.arange(len(class_names)))
        ax1.set_yticks(np.arange(len(class_names)))
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=8)
        
        # Plot 2: Per-Class Accuracy
        colors = ['green' if acc >= 0.8 else 'orange' if acc >= 0.6 else 'red' for acc in per_class_accuracy]
        bars = ax2.barh(range(len(class_names)), per_class_accuracy, color=colors, alpha=0.7)
        ax2.set_xlabel('Accuracy', fontsize=12)
        ax2.set_ylabel('Class', fontsize=12)
        ax2.set_title('Per-Class Accuracy', fontsize=16, fontweight='bold')
        ax2.set_yticks(range(len(class_names)))
        ax2.set_yticklabels(class_names)
        ax2.set_xlim([0, 1.0])
        ax2.grid(axis='x', alpha=0.3)
        
        # Add accuracy values on bars
        for i, (bar, acc) in enumerate(zip(bars, per_class_accuracy)):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{acc:.3f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm, per_class_accuracy
    
    # DETAILED CLASSIFICATION REPORT
    def print_detailed_classification_report(y_true, y_pred, class_names, per_class_accuracy):
        print("\n" + "="*60)
        print("DETAILED PER-CLASS PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Print per-class accuracy
        print(f"\n{'Class':<20} {'Accuracy':<10} {'Support':<10}")
        print("-" * 45)
        
        for i, class_name in enumerate(class_names):
            acc = per_class_accuracy[i]
            support = np.sum(np.array(y_true) == i)
            print(f"{class_name:<20} {acc:<10.3f} {support:<10}")
        
        # Identify problematic classes
        problematic_classes = []
        for i, class_name in enumerate(class_names):
            if per_class_accuracy[i] < 0.6:
                problematic_classes.append((class_name, per_class_accuracy[i]))
        
        if problematic_classes:
            print(f"\n PROBLEMATIC CLASSES (Accuracy < 0.6):")
            for class_name, acc in problematic_classes:
                print(f"   • {class_name}: {acc:.3f}")
        else:
            print(f"\n All classes have good accuracy (≥ 0.6)")
    
    # TRAINING HISTORY TRACKING
    class TrainingHistory:
        def __init__(self):
            self.train_loss = []
            self.train_acc = []
            self.val_loss = []
            self.val_acc = []
            self.lr_history = []
        
        def update(self, train_loss, train_acc, val_loss, val_acc, lr=None):
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)
            if lr is not None:
                self.lr_history.append(lr)
        
        def get_combined_history(self):
            return {
                'loss': self.train_loss,
                'accuracy': self.train_acc,
                'val_loss': self.val_loss,
                'val_accuracy': self.val_acc
            }
    
    # IMPROVED EARLY STOPPING
    class EarlyStopping:
        def __init__(self, patience=2, min_delta=0.001, restore_best_weights=True):
            self.patience = patience
            self.min_delta = min_delta
            self.restore_best_weights = restore_best_weights
            self.counter = 0
            self.best_loss = None
            self.best_acc = None
            self.best_weights = None
            self.early_stop = False
            
        def __call__(self, val_loss, val_acc, model):
            if self.best_loss is None:
                self.best_loss = val_loss
                self.best_acc = val_acc
                self.best_weights = model.state_dict().copy()
            elif val_loss > self.best_loss - self.min_delta and val_acc <= self.best_acc:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.restore_best_weights:
                        model.load_state_dict(self.best_weights)
                    print(f"Early stopping triggered! Restoring weights from epoch with loss: {self.best_loss:.4f}, acc: {self.best_acc:.4f}")
            else:
                self.best_loss = val_loss
                self.best_acc = val_acc
                self.best_weights = model.state_dict().copy()
                self.counter = 0
    
    # MODEL CHECKPOINT
    def save_checkpoint(model, filename, epoch, accuracy):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
            'class_names': class_names,
            'num_classes': num_classes
        }, filename)
        print(f"✓ Model saved as {filename}")

    # ============================================================================
    # SINGLE PHASE: FINE-TUNING ONLY (NO SEPARATE FEATURE EXTRACTION PHASE)
    # ============================================================================
    
    print("\n" + "="*50)
    print("SINGLE PHASE: FINE-TUNING")
    print("="*50)
    
    # Unfreeze layers immediately for fine-tuning
    print("Unfreezing layers for fine-tuning...")
    model.unfreeze_layers(80)
    
    # Count trainable parameters after unfreezing
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # OPTIMIZER FOR FINE-TUNING
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Initialize training components
    history = TrainingHistory()
    early_stopping = EarlyStopping(patience=3, min_delta=0.001, restore_best_weights=True)
    best_val_acc = 0.0

    # FINE-TUNING TRAINING LOOP
    print("Starting fine-tuning...")
    for epoch in range(50):  # More epochs for single phase
        print(f"\nEpoch {epoch+1}/50")
        print("-" * 50)
        
        # Training
        print("Training...")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        print("Validating...")
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history.update(train_loss, train_acc, val_loss, val_acc, current_lr)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, 'best_model_final.pth', epoch, val_acc)
        
        # Early stopping check
        early_stopping(val_loss, val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # FINAL EVALUATION WITH CONFUSION MATRIX
    print("\n" + "="*50)
    print("FINAL EVALUATION WITH CONFUSION MATRIX")
    print("="*50)
    
    # Load best model for final evaluation
    try:
        checkpoint = torch.load('best_model_final.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best model")
    except:
        print("Using current model for evaluation")
    
    print("Testing with confusion matrix analysis...")
    test_loss, test_accuracy, test_preds, test_labels = test_model_with_confusion(
        model, test_loader, criterion, device, class_names
    )
    
    print(f"\nFinal Test Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Generate confusion matrix and per-class analysis
    print("\n" + "="*50)
    print("GENERATING CONFUSION MATRIX AND PER-CLASS ANALYSIS")
    print("="*50)
    
    cm, per_class_accuracy = plot_confusion_matrix_and_analysis(
        test_labels, test_preds, class_names, 'confusion_matrix_detailed.png'
    )
    
    # Print detailed classification report
    print_detailed_classification_report(test_labels, test_preds, class_names, per_class_accuracy)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes,
        'test_accuracy': test_accuracy,
        'per_class_accuracy': dict(zip(class_names, per_class_accuracy)),
        'history': history.get_combined_history()
    }, 'improved_mobilenetv2_final.pth')
    print("✓ Final model saved as 'improved_mobilenetv2_final.pth'")
    
    # Plot training history
    def plot_enhanced_training_history(history):
        combined_history = history.get_combined_history()
        
        plt.figure(figsize=(15, 5))
    
        # Accuracy plot
        plt.subplot(1, 3, 1)
        plt.plot(combined_history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(combined_history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss plot
        plt.subplot(1, 3, 2)
        plt.plot(combined_history['loss'], label='Training Loss', linewidth=2)
        plt.plot(combined_history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(1, 3, 3)
        plt.plot(history.lr_history, label='Learning Rate', color='green', linewidth=2)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    plot_enhanced_training_history(history)
    
    # Performance analysis
    print("\n" + "="*50)
    print("OVERALL PERFORMANCE SUMMARY")
    print("="*50)
    
    combined_history = history.get_combined_history()
    if combined_history['accuracy']:
        final_train_acc = combined_history['accuracy'][-1]
        final_val_acc = combined_history['val_accuracy'][-1]
        final_train_loss = combined_history['loss'][-1]
        final_val_loss = combined_history['val_loss'][-1]
        
        accuracy_gap = final_train_acc - final_val_acc
        
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Accuracy Gap: {accuracy_gap:.4f}")
        
        # Class performance summary
        print(f"\nClass Performance Summary:")
        print(f"  • Best class: {class_names[np.argmax(per_class_accuracy)]} ({np.max(per_class_accuracy):.3f})")
        print(f"  • Worst class: {class_names[np.argmin(per_class_accuracy)]} ({np.min(per_class_accuracy):.3f})")
        print(f"  • Average class accuracy: {np.mean(per_class_accuracy):.3f}")
        
        if accuracy_gap > 0.1:
            print(" Significant overfitting detected!")
        elif accuracy_gap > 0.05:
            print("  Moderate overfitting detected")
        else:
            print(" Good generalization - minimal overfitting")
            
        if test_accuracy < 0.5:
            print(" POOR PERFORMANCE: Model accuracy is very low!")
        elif test_accuracy < 0.7:
            print(" MODERATE PERFORMANCE: Model could be improved")
        else:
            print(" EXCELLENT PERFORMANCE: Model is working well!")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()