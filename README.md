# DermaScope

A convolutional neural network (CNN) for skin lesion classification using the HAM10000 dataset. The model uses ResNet18 transfer learning to classify dermoscopic images as benign or malignant, with a real-time inference pipeline that returns confidence scores alongside each prediction.

## How It Works

1. Skin lesion images from the HAM10000 dataset are preprocessed and normalized
2. A pretrained ResNet18 backbone extracts image features via transfer learning
3. A custom classification head maps features to binary output (benign / malignant)
4. Softmax produces a confidence score alongside each prediction

## Dataset

The model is trained on the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) — 10,015 dermoscopic images across 7 diagnostic categories:

| Label | Classes |
|-------|---------|
| Benign | `nv`, `bkl`, `df`, `vasc` |
| Malignant | `mel`, `bcc`, `akiec` |

## Tech Stack

**Model:** PyTorch, ResNet18 (pretrained on ImageNet)  
**Data:** torchvision transforms, custom `HAMDataset` class  
**Preprocessing:** Resize to 224×224, normalization with ImageNet mean/std

## Setup

### 1. Clone the repo

### 2. Download the HAM10000 dataset

### 3. Install dependencies

```bash
pip install torch torchvision pandas pillow
```

### 4. Train the model

Open and run `src/train.ipynb` in Jupyter. The final model weights are saved as `skin_lesion_model.pth`.

## Inference

```python
label, confidence = predict_image(model, image_path, transform, device)
print(label, f"{confidence*100:.2f}%")
```
