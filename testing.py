# --- PyTorch and other imports ---
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics.pairwise import cosine_similarity
import random
import serial
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from collections import Counter

dataset_path = "dataset"
augmented_dataset_path = "augmented_dataset"
feedback_dataset_path = "feedback_dataset"
combined_dataset_path = "combined_dataset"
both_dataset_path = "both_dataset"
categories = ["plastic", "paper", "metal", "glass", "organic", "unsupported"]
SERVO_COMMAND_MAP = {
    "plastic": "5",
    "paper": "2",
    "metal": "1",
    "glass": "4",
    "organic": "3",
}
num_classes = len(categories)
img_size = 128

# --- Database and records ---
database = {category: 0 for category in categories}
thesis_folder = r"D:\thesis\throwaway_with_hardware\src"
database_folder = os.path.join(thesis_folder, "database")
os.makedirs(database_folder, exist_ok=True)
records = []

# --- Augmentation (albumentations-based) ---
def get_albumentations_pipeline(img_size=128, mode='training'):

    if mode == 'validation':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    # Lighter augmentations for training
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Rotate(limit=10, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def custom_augmentation_albumentations(img, pipeline=None, img_size=128):

    if pipeline is None:
        pipeline = get_albumentations_pipeline(img_size=img_size, mode='training')
    
    # Ensure image is in correct format (uint8, RGB)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    
    # Apply albumentations pipeline
    augmented = pipeline(image=img)
    
    # Return the augmented image (normalized)
    return augmented['image']

# Legacy function for backward compatibility
def custom_augmentation_torch(img):

    return custom_augmentation_albumentations(img)

def augment_and_save(input_folder, output_folder, category, augment_count=10, target_size_for_cv2=128, progress_callback=None, augmentation_mode='training'):
    """
    Generate and save augmented images using albumentations.
    
    Args:
        input_folder: Source folder containing images
        output_folder: Destination folder for augmented images
        category: Category name (subdirectory)
        augment_count: Number of augmentations per image
        target_size_for_cv2: Target image size
        progress_callback: Callback for progress updates
        augmentation_mode: 'training', 'heavy', or 'light'
    """
    try:
        category_output_folder = os.path.join(output_folder, category)
        os.makedirs(category_output_folder, exist_ok=True)
        category_input_folder = os.path.join(input_folder, category)
        
        if not os.path.exists(category_input_folder):
            raise FileNotFoundError(f"Input folder not found: {category_input_folder}")
        
        files = [f for f in os.listdir(category_input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        processed_files = 0
        total_files = len(files)
        
        # Create albumentations pipeline for augmentation (without normalization for saving)
        aug_pipeline = A.Compose([
            A.Resize(target_size_for_cv2, target_size_for_cv2),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.7
            ),
            A.OneOf([
                A.Blur(blur_limit=3, p=1),
                A.MedianBlur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ], p=0.3),
            A.RandomResizedCrop(
                size=(target_size_for_cv2, target_size_for_cv2), 
                scale=(0.8, 1.0), 
                ratio=(0.9, 1.1), 
                p=0.8
            ),
            A.OneOf([
                A.OpticalDistortion(p=1),
                A.GridDistortion(p=1),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1),
            ], p=0.3),
            A.CLAHE(clip_limit=2, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CoarseDropout(
                num_holes_range=(3, 8),
                hole_height_range=(target_size_for_cv2//8, target_size_for_cv2//8),
                hole_width_range=(target_size_for_cv2//8, target_size_for_cv2//8),
                p=0.3
            ),
        ])
        
        if augmentation_mode == 'heavy':
            heavy_augs = [
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
                ], p=0.4),
                A.OneOf([
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=1),
                    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=1),
                ], p=0.2),
                A.RandomShadow(p=0.3),
                A.OneOf([
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1),
                ], p=0.3),
            ]
            aug_pipeline = A.Compose(aug_pipeline.transforms + heavy_augs)
        
        for idx, file in enumerate(files):
            img_path = os.path.join(category_input_folder, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Convert to RGB for albumentations
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            for i in range(augment_count):
                # Apply albumentations pipeline
                augmented = aug_pipeline(image=img)
                aug_img = augmented['image']
                
                # Save augmented image
                save_prefix = f"aug_{os.path.splitext(file)[0]}_{i}"
                save_path = os.path.join(category_output_folder, f"{save_prefix}.jpg")
                
                # Convert back to BGR for OpenCV saving
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, aug_img_bgr)
            
            processed_files += 1
            if progress_callback:
                progress_callback('preparing', idx+1, total_files)
        
        print(f"Generated {augment_count} augmentations each for {processed_files} {category} images using albumentations")
    except Exception as e:
        print(f"Error augmenting {category}: {str(e)}")
        raise

# --- PyTorch Datasets and Loaders with Albumentations ---
class FolderDatasetAlbumentations(Dataset):
    def __init__(self, folder, categories, pipeline=None, img_size=128):
        self.samples = []
        self.labels = []
        self.pipeline = pipeline
        self.categories = categories
        self.img_size = img_size
        for idx, cat in enumerate(categories):
            cat_folder = os.path.join(folder, cat)
            if not os.path.exists(cat_folder):
                continue
            for fname in os.listdir(cat_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cat_folder, fname), idx))
                    self.labels.append(idx)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.pipeline:
            augmented = self.pipeline(image=img)
            img = augmented['image']
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img).permute(2, 0, 1).float()
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img = normalize(img)
        return img, label

def get_loaders_albumentations(dataset_path, categories, batch_size=32, val_pct=0.2, img_size=128, augmentation_mode='training'):
    train_pipeline = get_albumentations_pipeline(img_size=img_size, mode=augmentation_mode)
    val_pipeline = get_albumentations_pipeline(img_size=img_size, mode='validation')
    manual_train_dir = os.path.join(dataset_path, 'train')
    manual_val_dir = os.path.join(dataset_path, 'validation')
    has_manual_split = os.path.isdir(manual_train_dir) and os.path.isdir(manual_val_dir)
    if has_manual_split:
        train_dataset = FolderDatasetAlbumentations(manual_train_dir, categories, pipeline=train_pipeline, img_size=img_size)
        val_dataset = FolderDatasetAlbumentations(manual_val_dir, categories, pipeline=val_pipeline, img_size=img_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        # Print class distribution
        print('Train class distribution:', np.bincount(train_dataset.labels))
        print('Val class distribution:', np.bincount(val_dataset.labels))
        return train_loader, val_loader
    # Fallback: stratified split on entire dataset root
    full_dataset = FolderDatasetAlbumentations(dataset_path, categories, pipeline=None, img_size=img_size)
    indices = list(range(len(full_dataset)))
    labels = full_dataset.labels
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_pct,
        stratify=labels,
        random_state=42
    )
    train_dataset = FolderDatasetAlbumentations(dataset_path, categories, pipeline=train_pipeline, img_size=img_size)
    val_dataset = FolderDatasetAlbumentations(dataset_path, categories, pipeline=val_pipeline, img_size=img_size)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    # Print class distribution for stratified split
    train_labels = [train_dataset.labels[i] for i in train_indices]
    val_labels = [val_dataset.labels[i] for i in val_indices]
    print('Train class distribution:', Counter(train_labels))
    print('Val class distribution:', Counter(val_labels))
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    return train_loader, val_loader

# Legacy function for backward compatibility
def get_loaders(dataset_path, categories, batch_size=32, val_pct=0.2, img_size=128):
    """Legacy function - now uses albumentations backend"""
    return get_loaders_albumentations(dataset_path, categories, batch_size, val_pct, img_size, 'training')

# --- Model Definitions ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes=6, input_size=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear((input_size//16)*(input_size//16)*256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model(model_type="cnn", num_classes=6, input_size=128):
    if model_type == "transfer":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
    else:
        return CustomCNN(num_classes=num_classes, input_size=input_size)

# --- Training ---
def train_model(callbacks=None, progress_callback=None, dataset_mode="augmented", model_type="cnn", custom_dataset_path=None, epochs=30, augmentation_mode="training", log_queue=None, initial_weights_path=None, freeze_backbone=False):
    categories = ["plastic", "paper", "metal", "glass", "organic", "unsupported"]
    target_size = (224, 224) if model_type == "transfer" else (128, 128)
    input_shape = (*target_size, 3)
    if progress_callback:
        progress_callback('status', 'preparing')
    if custom_dataset_path:
        dataset_path = custom_dataset_path
    else:
        original_path = "dataset"
        if dataset_mode == "original":
            dataset_path = original_path
        elif dataset_mode == "augmented":
            # Build or reuse an augmented dataset that mirrors manual split
            dataset_path = "augmented_dataset"
            orig_train = os.path.join(original_path, 'train')
            orig_val = os.path.join(original_path, 'validation')
            aug_train = os.path.join(dataset_path, 'train')
            aug_val = os.path.join(dataset_path, 'validation')

            def _is_image(fname: str) -> bool:
                return fname.lower().endswith((".jpg", ".jpeg", ".png"))

            # Ensure augmented split folders exist
            for cat in categories:
                os.makedirs(os.path.join(aug_train, cat), exist_ok=True)
                os.makedirs(os.path.join(aug_val, cat), exist_ok=True)

            # Count existing augmented images
            def _count_dir_images(root_dir: str) -> int:
                total = 0
                for cat in categories:
                    cat_dir = os.path.join(root_dir, cat)
                    if os.path.isdir(cat_dir):
                        total += sum(1 for f in os.listdir(cat_dir) if _is_image(f))
                return total

            total_aug_train = _count_dir_images(aug_train)
            total_aug_val = _count_dir_images(aug_val)

            # If train or val in augmented dataset is empty, (re)build from the original manual split
            # - Train: augment originals and save into augmented/train/<cat>
            # - Val: copy originals into augmented/validation/<cat> without augmentation
            if total_aug_train == 0 or total_aug_val == 0:
                has_manual_split = os.path.isdir(orig_train) and os.path.isdir(orig_val)
                if has_manual_split:
                    steps = len(categories)
                    for idx, category in enumerate(categories):
                        # Copy validation images if needed
                        src_val_cat = os.path.join(orig_val, category)
                        dst_val_cat = os.path.join(aug_val, category)
                        os.makedirs(dst_val_cat, exist_ok=True)
                        if os.path.isdir(src_val_cat):
                            for f in os.listdir(src_val_cat):
                                if not _is_image(f):
                                    continue
                                src_f = os.path.join(src_val_cat, f)
                                dst_f = os.path.join(dst_val_cat, f)
                                if not os.path.exists(dst_f):
                                    shutil.copy2(src_f, dst_f)

                        # Augment training images into augmented/train/<category>
                        augment_and_save(
                            input_folder=orig_train,
                            output_folder=aug_train,
                            category=category,
                            augment_count=10,
                            target_size_for_cv2=target_size[0],
                            progress_callback=progress_callback,
                            augmentation_mode=augmentation_mode,
                        )
                        if progress_callback:
                            progress_callback('preparing', idx+1, steps)
                else:
                    # Fallback: treat original as flat per-category; augment flat into aug root
                    os.makedirs(dataset_path, exist_ok=True)
                    for idx, category in enumerate(categories):
                        augment_and_save(
                            input_folder=original_path,
                            output_folder=dataset_path,
                            category=category,
                            augment_count=10,
                            target_size_for_cv2=target_size[0],
                            progress_callback=progress_callback,
                            augmentation_mode=augmentation_mode,
                        )
                        if progress_callback:
                            progress_callback('preparing', idx+1, len(categories))
        elif dataset_mode == "both":
            dataset_path = "both_dataset"
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
                for idx, category in enumerate(categories):
                    src = os.path.join(original_path, category)
                    dst = os.path.join(dataset_path, category)
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    augment_and_save(original_path, dataset_path, category, progress_callback=progress_callback, augmentation_mode=augmentation_mode)
                    if progress_callback:
                        progress_callback('preparing', idx+1, len(categories))
    print_class_distribution(dataset_path, categories)
    train_loader, val_loader = get_loaders_albumentations(dataset_path, categories, batch_size=32, img_size=target_size[0], augmentation_mode=augmentation_mode)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_type=model_type, num_classes=len(categories), input_size=target_size[0]).to(device)

    # Optionally freeze backbone for transfer learning
    if model_type == "transfer" and freeze_backbone:
        try:
            for param in model.features.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
            if log_queue is not None:
                log_queue.put("[Frozen backbone parameters; training classifier head only]")
        except Exception:
            pass

    # Optionally load initial weights for fine-tuning
    def _log(msg: str):
        print(msg)
        if log_queue is not None:
            log_queue.put(msg)

    loaded_from = None
    try:
        if initial_weights_path and os.path.exists(initial_weights_path):
            state = torch.load(initial_weights_path, map_location=device)
            model.load_state_dict(state, strict=False)
            loaded_from = initial_weights_path
        else:
            # Auto fallback
            auto_map = {
                'transfer': 'trash_classifier_transfer_improved.pth',
                'cnn': 'trash_classifier_cnn_improved.pth',
            }
            auto_path = auto_map.get(model_type)
            if auto_path and os.path.exists(auto_path):
                state = torch.load(auto_path, map_location=device)
                model.load_state_dict(state, strict=False)
                loaded_from = auto_path
        if loaded_from:
            _log(f"[Loaded initial weights from {loaded_from}]")
    except Exception as e:
        _log(f"[Warning] Could not load initial weights: {e}")
    criterion = nn.CrossEntropyLoss()
    # If backbone is frozen, optimize only trainable params
    params_to_optimize = (p for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam(params_to_optimize, lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    best_val_acc = 0.0
    best_weights = None
    patience = 8
    min_delta = 1e-4
    wait = 0
    best_val_loss = float('inf')
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'epoch': []}
    total_epochs = epochs
    if progress_callback:
        progress_callback('status', 'training')
    for epoch in range(epochs):
        if progress_callback:
            progress_callback('epoch', epoch+1, total_epochs)
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        total_batches = len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if progress_callback:
                progress_callback('batch', batch_idx+1, total_batches)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        history['epoch'].append(epoch)
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        print(msg)
        if log_queue is not None:
            log_queue.put(msg)
        scheduler.step(val_loss)
        # Only check for early stopping after 5 epochs
        if epoch >= 5:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_weights = model.state_dict()
                wait = 0
                torch.save(model.state_dict(), f"trash_classifier_{model_type}_improved.pth")
                msg2 = f"[Saved best model to trash_classifier_{model_type}_improved.pth]"
                print(msg2)
                if log_queue is not None:
                    log_queue.put(msg2)
            else:
                wait += 1
            if wait >= patience:
                msg3 = f"Early stopping at epoch {epoch+1} (no val_loss improvement for {patience} epochs)"
                print(msg3)
                if log_queue is not None:
                    log_queue.put(msg3)
                break
    print("Training complete.")
    if log_queue is not None:
        log_queue.put("Training complete.")
    if progress_callback:
        progress_callback('status', 'done')
    # Save training history and plot
    with open('training_report.json', 'w') as f:
        json.dump({
            'stopped_epoch': epoch+1,
            'history': history,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss
        }, f, indent=2)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['epoch'], history['accuracy'], label='Train Acc')
    plt.plot(history['epoch'], history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history['epoch'], history['loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.suptitle(f'Stopped at epoch {epoch+1}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('training_curves.png')
    plt.show(block=False)
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return None, model

# --- Retrain model with log_queue support ---
def retrain_model(epochs=30, log_queue=None, progress_callback=None, model_type: str = "cnn", initial_weights_path: str | None = None):

    total_steps = len(categories)
    # Create train/validation folders for combined
    for category in categories:
        os.makedirs(os.path.join(combined_dataset_path, 'train', category), exist_ok=True)
        os.makedirs(os.path.join(combined_dataset_path, 'validation', category), exist_ok=True)

    # Detect if original dataset has manual split
    original_has_manual = (
        os.path.isdir(os.path.join(dataset_path, 'train')) and
        os.path.isdir(os.path.join(dataset_path, 'validation'))
    )

    for idx, category in enumerate(categories):
        # Copy original data
        if original_has_manual:
            # Copy train
            src_train_cat = os.path.join(dataset_path, 'train', category)
            dst_train_cat = os.path.join(combined_dataset_path, 'train', category)
            if os.path.isdir(src_train_cat):
                for file in os.listdir(src_train_cat):
                    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    src = os.path.join(src_train_cat, file)
                    dst = os.path.join(dst_train_cat, file)
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)
            # Copy validation
            src_val_cat = os.path.join(dataset_path, 'validation', category)
            dst_val_cat = os.path.join(combined_dataset_path, 'validation', category)
            if os.path.isdir(src_val_cat):
                for file in os.listdir(src_val_cat):
                    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    src = os.path.join(src_val_cat, file)
                    dst = os.path.join(dst_val_cat, file)
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)
        else:
            # Flat original: copy all into combined/train
            src_cat = os.path.join(dataset_path, category)
            dst_cat = os.path.join(combined_dataset_path, 'train', category)
            if os.path.isdir(src_cat):
                for file in os.listdir(src_cat):
                    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    src = os.path.join(src_cat, file)
                    dst = os.path.join(dst_cat, file)
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)

        # Split feedback 80/20 per category
        feedback_cat = os.path.join(feedback_dataset_path, category)
        if os.path.isdir(feedback_cat):
            files = [f for f in os.listdir(feedback_cat) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(files) > 0:
                # Shuffle deterministically per run for variety
                import random
                random.shuffle(files)
                split_idx = int(len(files) * 0.8)
                train_files = files[:split_idx]
                val_files = files[split_idx:]

                # Copy to combined/train
                dst_train_cat = os.path.join(combined_dataset_path, 'train', category)
                for file in train_files:
                    src = os.path.join(feedback_cat, file)
                    dst = os.path.join(dst_train_cat, file)
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)

                # Copy to combined/validation
                dst_val_cat = os.path.join(combined_dataset_path, 'validation', category)
                for file in val_files:
                    src = os.path.join(feedback_cat, file)
                    dst = os.path.join(dst_val_cat, file)
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)

        if progress_callback:
            progress_callback('preparing', idx + 1, total_steps)

    if log_queue is not None:
        log_queue.put("Combined dataset (train/validation) ready. Starting retraining...")

    # Train using combined dataset which now has manual split
    train_model(
        dataset_mode="augmented",
        model_type=model_type,
        custom_dataset_path=combined_dataset_path,
        epochs=epochs,
        log_queue=log_queue,
        progress_callback=progress_callback,
        initial_weights_path=initial_weights_path,
    )

def print_class_distribution(dataset_path, categories):

    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'validation')
    has_manual = os.path.isdir(train_dir) and os.path.isdir(val_dir)

    if has_manual:
        print("Class distribution (manual split):")
        total_train, total_val = 0, 0
        for cat in categories:
            train_cat = os.path.join(train_dir, cat)
            val_cat = os.path.join(val_dir, cat)
            train_count = len([f for f in os.listdir(train_cat) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(train_cat) else 0
            val_count = len([f for f in os.listdir(val_cat) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(val_cat) else 0
            total_train += train_count
            total_val += val_count
            print(f"  {cat}: train={train_count}, val={val_count}")
        print(f"Totals: train={total_train}, val={total_val}")
    else:
        print("Class distribution:")
        for cat in categories:
            cat_folder = os.path.join(dataset_path, cat)
            if os.path.exists(cat_folder):
                count = len([f for f in os.listdir(cat_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            else:
                count = 0
            print(f"  {cat}: {count}")

def preprocess_frame_pytorch(frame, img_size=224):

    # Convert BGR to RGB if needed
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        img = frame
    
    # Create preprocessing pipeline
    preprocess_pipeline = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Apply preprocessing
    processed = preprocess_pipeline(image=img)
    img_tensor = processed['image'].unsqueeze(0)  # Add batch dimension
    
    return img_tensor


def predict_real_time_with_box(num_boxes=1, callback=None, yolo_weights_path: str | None = None, classifier_weights_path: str | None = None, stop_event=None, serial_callback=None):

    import torch
    import cv2
    from ultralytics import YOLO

    # Load YOLOv8 model
    default_yolo = 'D:\thesis\throwaway_with_hardware\src\yolofinalmodel\weights\best.pt'
    yolo_path = yolo_weights_path if (yolo_weights_path and os.path.isfile(yolo_weights_path)) else default_yolo
    yolo_model = YOLO(yolo_path)

    # Load classifier model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Infer classifier type based on provided checkpoint name
    if classifier_weights_path and os.path.isfile(classifier_weights_path):
        fname = os.path.basename(classifier_weights_path).lower()
        if 'cnn' in fname:
            classifier_model_type = 'cnn'
            input_size = 128
        else:
            classifier_model_type = 'transfer'
            input_size = 224
        checkpoint_path = classifier_weights_path
    else:
        # Fallback to defaults
        # Prefer transfer if available
        if os.path.isfile('trash_classifier_transfer_improved.pth'):
            classifier_model_type = 'transfer'
            input_size = 224
            checkpoint_path = 'trash_classifier_transfer_improved.pth'
        else:
            classifier_model_type = 'cnn'
            input_size = 128
            checkpoint_path = 'trash_classifier_cnn_improved.pth'

    classifier = get_model(model_type=classifier_model_type, num_classes=len(categories), input_size=input_size)
    if os.path.isfile(checkpoint_path):
        classifier.load_state_dict(torch.load(checkpoint_path, map_location=device))
    classifier.eval()
    classifier.to(device)

    # Use DirectShow backend on Windows to reduce latency
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Prefer MJPG to get higher FPS on many webcams
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Initialize Arduino serial (Windows default COM4). If unavailable, try to auto-detect.
    ser = None
    selected_port = None
    try:
        ser = serial.Serial('COM4', 115200, timeout=1)
        selected_port = 'COM4'
    except Exception:
        # Try to auto-detect Arduino-like ports
        try:
            from serial.tools import list_ports
            candidates = []
            for p in list_ports.comports():
                desc = (p.description or '').lower()
                hwid = (p.hwid or '').lower()
                if any(k in desc for k in ['arduino', 'wchusb', 'usb serial', 'usb-serial']) or any(k in hwid for k in ['2341', '1a86', '10c4']):
                    candidates.append(p.device)
            if not candidates:
                # Fallback: try common COM numbers
                candidates = [f'COM{i}' for i in range(3, 11)]
            for dev in candidates:
                try:
                    ser = serial.Serial(dev, 115200, timeout=1)
                    selected_port = dev
                    break
                except Exception:
                    continue
        except Exception:
            pass
    if ser is not None:
        # Allow Arduino to reset after opening serial
        try:
            time.sleep(2.0)
            ser.reset_input_buffer()
        except Exception:
            pass
        print(f"[Serial] Connected @ 115200 on {selected_port}")
    else:
        print("[Serial] No serial connection available")

    # Cooldown tracking for servo triggers
    last_servo_time = 0.0
    last_cooldown_seconds = 0.0
    # Map Arduino hold delays per command (seconds), add 0.5s buffer after each
    COMMAND_HOLD_SECONDS = {
        '1': 5.0,
        '2': 5.5,
        '3': 6.5,
        '4': 5.0,
        '5': 5.0,
    }

    # yoloconfidence
    yolo_confidence_threshold = 0.3  # yoloconfidence
    # cnnconfidence
    cnn_confidence_threshold = 0.75  # cnnconfidence
    process_every_n = 3
    frame_count = 0
    exit_flag = False

    def _should_stop() -> bool:
        try:
            return bool(stop_event.is_set()) if stop_event is not None else False
        except Exception:
            return False

    # Simple box IoU for track matching
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        if interArea <= 0:
            return 0.0
        boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
        boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0 else 0.0

    # Persist detections to reduce flicker between frames
    persisted_tracks = []  # each: {bbox:(x1,y1,x2,y2), label:str, conf:float, ttl:int}
    persistence_frames = 6
    smoothing_alpha = 0.6  # weight for new box when smoothing
    print("Starting real-time prediction... Press 'q' to quit")
    while not exit_flag and not _should_stop():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        frame_disp = frame.copy()
        # Draw trigger lines: plastic at 40%, others at 20%
        h, w = frame_disp.shape[:2]
        line_y_plastic = int(0.40 * h)
        line_y_other = int(0.20 * h)
        # Plastic & Metal line (red)
        cv2.line(frame_disp, (0, line_y_plastic), (w - 1, line_y_plastic), (0, 0, 255), 2)
        cv2.putText(frame_disp, "plastic 40%", (10, line_y_plastic - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Others line (blue)
        cv2.line(frame_disp, (0, line_y_other), (w - 1, line_y_other), (255, 0, 0), 2)
        cv2.putText(frame_disp, "others 20%", (10, line_y_other - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        overlay_messages = []
        labels_this_frame = []
        confidences_this_frame = []
        rois = []
        category_counts = {cat: 0 for cat in categories}

        # Only run YOLO every Nth frame to keep UI responsive
        if frame_count % process_every_n == 0:
            results = yolo_model(frame, conf=yolo_confidence_threshold)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else []
            confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else []
            classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []
        else:
            boxes, confs, classes = [], [], []

        # Classify detected boxes only on processed frames and update persisted tracks
        detections = []
        if frame_count % process_every_n == 0 and len(boxes) > 0:
            # Sort by confidence, take top num_boxes
            sorted_indices = confs.argsort()[::-1][:num_boxes]
            for idx in sorted_indices:
                x1, y1, x2, y2 = map(int, boxes[idx])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # Preprocess crop for classifier
                input_crop = preprocess_frame_pytorch(crop, img_size=input_size).to(device)
                with torch.no_grad():
                    output = classifier(input_crop)
                    prob = torch.softmax(output, dim=1)
                    confidence, predicted_class_idx = torch.max(prob, 1)
                    confidence = float(confidence.item())
                    predicted_class_idx = int(predicted_class_idx.item())
                    predicted_class = categories[predicted_class_idx]
                # Treat any low-confidence result as 'unsupported'
                if confidence < cnn_confidence_threshold:
                    label = 'unsupported'
                else:
                    label = predicted_class
                labels_this_frame.append(label)
                confidences_this_frame.append(confidence)
                rois.append(crop)
                category_counts[label] = category_counts.get(label, 0) + 1
                detections.append({'bbox': (x1, y1, x2, y2), 'label': label, 'conf': confidence})

            # Update persisted tracks with smoothing and TTL
            matched = set()
            for det in detections:
                best_iou = 0.0
                best_idx = -1
                for j, track in enumerate(persisted_tracks):
                    if j in matched:
                        continue
                    iou_val = _iou(det['bbox'], track['bbox'])
                    if iou_val > 0.3 and iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = j
                if best_idx >= 0:
                    # Smooth update
                    tb = persisted_tracks[best_idx]['bbox']
                    nb = det['bbox']
                    smoothed = (
                        int((1 - smoothing_alpha) * tb[0] + smoothing_alpha * nb[0]),
                        int((1 - smoothing_alpha) * tb[1] + smoothing_alpha * nb[1]),
                        int((1 - smoothing_alpha) * tb[2] + smoothing_alpha * nb[2]),
                        int((1 - smoothing_alpha) * tb[3] + smoothing_alpha * nb[3]),
                    )
                    persisted_tracks[best_idx]['bbox'] = smoothed
                    persisted_tracks[best_idx]['label'] = det['label']
                    persisted_tracks[best_idx]['conf'] = det['conf']
                    persisted_tracks[best_idx]['ttl'] = persistence_frames
                    matched.add(best_idx)
                else:
                    persisted_tracks.append({
                        'bbox': det['bbox'],
                        'label': det['label'],
                        'conf': det['conf'],
                        'ttl': persistence_frames
                    })
            # Decrement TTL for unmatched tracks
            for j, track in enumerate(persisted_tracks):
                if j not in matched:
                    track['ttl'] -= 1
            persisted_tracks = [t for t in persisted_tracks if t['ttl'] > 0]
        else:
            # Not running YOLO this frame: decay TTL slightly to allow natural fade
            for t in persisted_tracks:
                t['ttl'] -= 1
            persisted_tracks = [t for t in persisted_tracks if t['ttl'] > 0]

        # Draw all persisted tracks every frame to avoid flicker
        for t in persisted_tracks:
            x1, y1, x2, y2 = map(int, t['bbox'])
            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame_disp, f"{t['label']} ({t['conf']:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        for t in persisted_tracks:
            x1, y1, x2, y2 = map(int, t['bbox'])
            crosses_plastic_line = y1 <= line_y_plastic <= y2
            crosses_other_line = y1 <= line_y_other <= y2
            if not (crosses_plastic_line or crosses_other_line):
                continue
            now_time = time.time()
            if now_time - last_servo_time >= last_cooldown_seconds:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # Prefer persisted on-screen label when confident; otherwise treat as unsupported (no reclassification)
                if t['label'] in categories and float(t.get('conf', 0.0)) >= cnn_confidence_threshold:
                    predicted_class = t['label']
                    predicted_class_idx = categories.index(predicted_class)
                    overlay_messages.append(f"TRIGGER: using {predicted_class} ({t['conf']:.2f})")
                    conf_for_log = float(t.get('conf', 0.0))
                else:
                    predicted_class = 'unsupported'
                    predicted_class_idx = categories.index(predicted_class)
                    overlay_messages.append("TRIGGER: using unsupported (low confidence)")
                    conf_for_log = float(t.get('conf', 0.0))

                # Decide if this crossing should trigger based on which line was crossed and class
                allowed = False
                which_line = None
                if crosses_plastic_line and predicted_class in ('plastic'):
                    allowed = True
                    which_line = 'plastic(40%)'
                elif crosses_other_line and predicted_class not in ('plastic'):
                    allowed = True
                    which_line = 'others(20%)'

                if not allowed:
                    overlay_messages.append(f"IGNORED {predicted_class} at line rule")
                    continue

                # Log in format: <class_index> <class_name> with 1-based index
                class_index_1_based = predicted_class_idx + 1
                log_line = f"{class_index_1_based} {predicted_class}"
                print(log_line)
                overlay_messages.append(f"LOG: {log_line} @ {which_line}")
                # Send corresponding servo command based on mapping
                if ser is not None:
                    try:
                        cmd_char = SERVO_COMMAND_MAP.get(predicted_class, str(class_index_1_based))
                        cmd = cmd_char + '\n'
                        ser.write(cmd.encode())
                        try:
                            ser.flush()
                        except Exception:
                            pass
                        sent_msg = f"SERIAL SENT: {cmd_char} ({predicted_class})"
                        print(f"[Serial] {sent_msg}")
                        overlay_messages.append(sent_msg)
                        # Notify GUI for logging/counters
                        try:
                            if serial_callback is not None:
                                serial_callback(predicted_class, conf_for_log)
                        except Exception:
                            pass
                        # Update cooldown based on Arduino routine for this command
                        hold = float(COMMAND_HOLD_SECONDS.get(cmd_char, 3.0))
                        # Values in COMMAND_HOLD_SECONDS already include buffer
                        last_cooldown_seconds = hold
                    except Exception:
                        err_msg = "SERIAL ERROR: write failed"
                        print(f"[Serial] {err_msg}")
                        overlay_messages.append(err_msg)
                else:
                    not_connected = "SERIAL NOT CONNECTED"
                    print(f"[Serial] {not_connected}")
                    overlay_messages.append(not_connected)
                last_servo_time = now_time
            else:
                remaining = max(0.0, last_cooldown_seconds - (now_time - last_servo_time))
                overlay_messages.append(f"COOLDOWN {remaining:.1f}s")
            # If within cooldown, skip sending duplicate commands

        # Ensure global cooldown overlay persists even when no active tracks
        try:
            rem_global = max(0.0, last_cooldown_seconds - (time.time() - last_servo_time))
            if rem_global > 0:
                if not any(isinstance(m, str) and m.startswith("COOLDOWN ") for m in overlay_messages):
                    overlay_messages.append(f"COOLDOWN {rem_global:.1f}s")
        except Exception:
            pass

        # Draw mini log overlay (up to last 3 messages)
        for i, msg in enumerate(overlay_messages[-3:][::-1]):
            y_text = h - 10 - i * 20
            cv2.putText(frame_disp, msg, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Prepare data for callback using persisted tracks if needed
        if callback is not None:
            if labels_this_frame:
                # Use current frame's classification results
                roi_to_send = rois[0] if rois else None
                callback(frame_disp, labels_this_frame, confidences_this_frame, category_counts, roi_to_send)
            else:
                # Build from persisted tracks
                labels_persisted = [t['label'] for t in persisted_tracks]
                confs_persisted = [t['conf'] for t in persisted_tracks]
                counts = {cat: 0 for cat in categories}
                for t in persisted_tracks:
                    if t['label'] in counts:
                        counts[t['label']] += 1
                # Use ROI from the highest-confidence track if available
                roi_to_send = None
                if persisted_tracks:
                    top = max(persisted_tracks, key=lambda t: t['conf'])
                    x1, y1, x2, y2 = map(int, top['bbox'])
                    roi = frame[y1:y2, x1:x2]
                    if roi.size != 0:
                        roi_to_send = roi
                callback(frame_disp, labels_persisted, confs_persisted, counts, roi_to_send)

        frame_count += 1
        # Exit on 'q' key only when not driven by GUI
        if callback is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag = True

    cap.release()
    cv2.destroyAllWindows()
    try:
        if ser is not None:
            ser.close()
    except Exception:
        pass