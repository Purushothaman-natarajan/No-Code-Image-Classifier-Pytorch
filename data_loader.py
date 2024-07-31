import torch
import os
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
from tqdm import tqdm  # For progress display
import sys
import uuid  # Import uuid for unique filename generation

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Data Loader with Augmentation and Splits')
    parser.add_argument('--path', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--dim', type=int, default=224, help='Required image dimension')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--target_folder', type=str, required=True, help='Folder to store the train, test, and val splits')
    parser.add_argument('--augment_data', action='store_true', help='Apply data augmentation')
    return parser.parse_args()

def create_transforms(image_size, augment_data):
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    if augment_data:
        augment_transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], p=0.5),
            transforms.ToTensor(),
        ])
        return base_transform, augment_transform
    
    return base_transform, None

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, label_to_name, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.label_to_name = label_to_name
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label_idx = self.labels[idx]
        label_name = self.label_to_name[label_idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_name

def save_image(image, file_path):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    image.save(file_path)

def load_data(path, image_size, batch_size, augment_data, num_workers):
    all_images = []
    labels = []
    label_to_idx = {}
    idx_to_label = {}
    
    for subdir, _, files in os.walk(path):
        label = os.path.basename(subdir)
        if label not in label_to_idx:
            idx = len(label_to_idx)
            label_to_idx[label] = idx
            idx_to_label[idx] = label
        
        for fname in files:
            if fname.endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(subdir, fname))
                labels.append(label_to_idx[label])
    
    unique_labels = set(labels)
    print(f"Found {len(all_images)} images in {path}\n")
    print(f"Labels found ({len(unique_labels)}): {', '.join(idx_to_label[i] for i in unique_labels)}\n")
    
    if len(all_images) == 0:
        raise ValueError(f"No images found in the specified path: {path}")
    
    dataset = CustomDataset(all_images, labels, idx_to_label, transform=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]))

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    val_size = int(0.5 * len(test_dataset))
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])
    
    print(f"Data split into {len(train_dataset)} train, {len(val_dataset)} validation, and {len(test_dataset)} test images.\n")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def save_datasets_to_folders(dataloader, folder_path, transform=None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    count = 0
    for images, labels in tqdm(dataloader, desc=f"Saving to {folder_path}"):
        for i in range(images.shape[0]):
            image = images[i]
            label_name = labels[i]
            label_folder = os.path.join(folder_path, label_name)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)

            # Save the original image
            file_path = os.path.join(label_folder, f"{uuid.uuid4().hex}.jpg")
            save_image(image, file_path)
            count += 1

            # Apply augmentations if transform is provided
            if transform:
                # Convert tensor back to PIL image for augmentation
                pil_image = transforms.ToPILImage()(image)
                aug_image = transform(pil_image)
                file_path = os.path.join(label_folder, f"{uuid.uuid4().hex}_aug.jpg")
                save_image(aug_image, file_path)
                count += 1
    
    print(f"Saved {count} images to {folder_path}\n")
    return count

def main():
    args = parse_arguments()
    
    if not os.path.exists(args.target_folder):
        os.makedirs(args.target_folder)
        
    train_folder = os.path.join(args.target_folder, 'train')
    val_folder = os.path.join(args.target_folder, 'val')
    test_folder = os.path.join(args.target_folder, 'test')
    
    base_transform, augment_transform = create_transforms(args.dim, args.augment_data)

    train_loader, val_loader, test_loader = load_data(
        args.path,
        args.dim,
        args.batch_size,
        args.augment_data,
        args.num_workers
    )
    
    # Save datasets to respective folders and count images
    train_count = save_datasets_to_folders(train_loader, train_folder, augment_transform)
    val_count = save_datasets_to_folders(val_loader, val_folder)
    test_count = save_datasets_to_folders(test_loader, test_folder)
    
    print(f"Train dataset saved to: {train_folder}\n")
    print(f"Validation dataset saved to: {val_folder}\n")
    print(f"Test dataset saved to: {test_folder}\n")
    
    print('-'*20)

    print(f"Number of images in training set: {train_count}\n")
    print(f"Number of images in validation set: {val_count}\n")
    print(f"Number of images in test set: {test_count}\n")

if __name__ == "__main__":
    # Redirect stdout and stderr to avoid encoding issues
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
    main()
