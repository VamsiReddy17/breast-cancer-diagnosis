import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class BreakHisDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Path to the extracted images folder (e.g., data/raw/images)
        transform: PyTorch image transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Recursively search for png/jpg files
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
            
        # Deduplicate and resolve labels
        for path in sorted(set(image_paths)):
            if ".DS_Store" in path:
                continue
                
            path_lower = path.lower()
            if "benign" in path_lower:
                self.samples.append((path, 0)) # 0 = Benign
            elif "malignant" in path_lower:
                self.samples.append((path, 1)) # 1 = Malignant

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    """
    Returns train and val/test transforms.
    Standard input size for pretrained models (like ResNet, EfficientNet) is 224x224.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform
