import os
import sys
import time
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
from PIL import Image
# ─── Path Bootstrap ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import BreakHisDataset, get_transforms
from src.cnn_model import HistopathologyCNN
from config.config import DL_CONFIG

# Setup logs
os.makedirs("logs", exist_ok=True)
os.makedirs("reports/figures/dl", exist_ok=True)
os.makedirs("reports/results", exist_ok=True)
os.makedirs("models/deep_learning", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/deep_learning.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trainer")

def train_model(epochs=None, batch_size=None, lr=None, model_name=None, patience=None):
    # 1. Configuration loading from DL_CONFIG
    if epochs is None: epochs = DL_CONFIG["epochs"]
    if batch_size is None: batch_size = DL_CONFIG["batch_size"]
    if lr is None: lr = DL_CONFIG["learning_rate"]
    if model_name is None: model_name = DL_CONFIG["backbone"]
    if patience is None: patience = DL_CONFIG["patience"]

    # Device configuration
    config_device = DL_CONFIG["device"]
    if config_device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon GPU Acceleration (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using NVIDIA CUDA GPU Acceleration")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
    else:
        device = torch.device(config_device)
        logger.info(f"Using configured device: {config_device}")

    # 2. Load dataset recursively from BreaKHis 400X parent folder
    dataset_dir = os.path.join("data", "raw", "images", "BreaKHis 400X")
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory missing: {dataset_dir}")
        return

    train_transform, val_transform = get_transforms()
    
    # Load all images recursively to perform custom exact 70/15/15 splits
    full_dataset = BreakHisDataset(dataset_dir, transform=None)
    if len(full_dataset) == 0:
        logger.error("No images found in the dataset directory.")
        return
        
    logger.info(f"Loaded total of {len(full_dataset)} images.")

    # 3. Splits (70% train, 15% val, 15% test)
    total_size = len(full_dataset)
    test_size = int(DL_CONFIG["test_size"] * total_size)
    val_size = int(DL_CONFIG["val_size"] * total_size)
    train_size = total_size - test_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_split, val_split, test_split = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Wrap splits to apply different transforms
    class DatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        def __len__(self):
            return len(self.subset)

    train_dataset_split = DatasetWrapper(train_split, train_transform)
    val_dataset_split = DatasetWrapper(val_split, val_transform)
    test_dataset_split = DatasetWrapper(test_split, val_transform)

    train_loader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset_split, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Final splits: Train={train_size}, Val={val_size}, Test={test_size}")

    # 4. Initialize model, loss, optimizer
    model = HistopathologyCNN(model_name=model_name, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5. Training loop with early stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    logger.info("Starting CNN training pipeline...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Checkpoint save & early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "models/deep_learning/best_model.pth")
            logger.info("--> Saved new best validation model checkpoint.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logger.info(f"--> Validation loss did not improve. (Patience: {epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs!")
                break

    logger.info("Training complete. Evaluating model on test set...")
    
    # 6. Evaluation
    model.load_state_dict(torch.load("models/deep_learning/best_model.pth"))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])

    # Convert to arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Classification report
    rep = classification_report(all_labels, all_preds, output_dict=True)
    logger.info(f"Test Accuracy: {rep['accuracy']:.4f}")
    
    # Save metrics JSON
    with open("reports/results/dl_metrics.json", "w") as f:
        json.dump(rep, f, indent=4)

    # 7. Generate Loss/Accuracy Curves Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("reports/figures/dl/loss_accuracy_curves.png")
    plt.close()

    # 8. Generate ROC and Confusion Matrix Plot
    plt.figure(figsize=(12, 5))
    
    # Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(all_labels, all_preds)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Test Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Benign", "Malignant"])
    plt.yticks(tick_marks, ["Benign", "Malignant"])
    
    # Write numbers in boxes
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')

    # ROC Curve
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("reports/figures/dl/test_evaluation.png")
    plt.close()

    logger.info("Evaluation complete! Figures saved to reports/figures/dl/.")

if __name__ == "__main__":
    train_model()
