# Deep Learning Image Classification Pipeline

OncoSense includes a deep learning pipeline to classify breast histopathology biopsy images as Benign or Malignant. This document details the neural network architecture, data preparation, and training strategy.

---

## 🧠 1. Model Architecture: EfficientNet-B0

The classifier uses **EfficientNet-B0** as the feature extraction backbone, combined with a custom classification head.

```
Input Biopsy (224x224x3)
       │
       ▼
EfficientNet-B0 Backbone (Feature Extractor)
       │
       ▼
Global Average Pooling
       │
       ▼
Linear Layer (1280 neurons ➔ 256 neurons) + ReLU
       │
       ▼
Dropout (30% probability)
       │
       ▼
Output Linear Layer (256 neurons ➔ 2 logits)
```

### Why EfficientNet-B0?
1.  **Compound Scaling**: EfficientNet balances network depth, width, and resolution using a compound coefficient, outperforming architectures like ResNet-50 while using $5\times$ fewer parameters.
2.  **Edge/CPU Friendliness**: Its lightweight structure (only ~5.3 million parameters) allows it to run real-time inference on low-resource hosting servers (like Render's free tier) in under 1 second.
3.  **Transfer Learning**: We initialize the model with pre-trained ImageNet weights to leverage rich feature hierarchies (edges, textures, shapes), fine-tuning only the classification layers on histopathology biopsies.

---

## 🖼️ 2. Image Preprocessing & Augmentations

During training, images from the BreaKHis dataset are preprocessed and augmented to prevent overfitting:
*   **Resizing**: Images are resized to $224 \times 224$ pixels to match the input resolution of the backbone.
*   **Random Rotations**: Randomly rotates images up to $15^\circ$ to handle various biopsy slide orientations.
*   **Color Jitter**: Randomly adjusts brightness, contrast, saturation, and hue to simulate varying lighting and staining conditions.
*   **Normalization**: Standardizes pixels using the ImageNet channel means and standard deviations:
    $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

---

## 🔄 3. Split Strategy & Data Wrapping

To guarantee honest evaluation, the entire pool of **1,693 biopsy images** is split randomly with a fixed seed (`42`):
*   **Training Set (70% - 1,187 images)**: Backpropagation adjusts head weights.
*   **Validation Set (15% - 253 images)**: Monitored during training to guide early stopping.
*   **Test Set (15% - 253 images)**: Evaluated once training ends to compute final metrics.

---

## 🛑 4. Early Stopping & Loss Optimization
*   **Optimizer**: Adam optimizer with a learning rate of $\eta = 0.001$.
*   **Loss Function**: Cross-Entropy Loss.
*   **Early Stopping (Patience = 3)**: The training loop checks the validation loss at the end of each epoch. If the validation loss fails to decrease for **3 consecutive epochs**, training terminates early. This prevents the network from memorizing the training subset (overfitting) and exports the weights representing the minimum validation error.
