# Pneumonia Detection from Chest X-Ray Images

This project implements a **medical image classification system** to detect **pneumonia** using chest X-ray images.  
It uses **Convolutional Neural Networks (CNNs)** and **Transfer Learning (ResNet50)** for high accuracy.

---

## Features

- Preprocessing of X-ray images (normalization, augmentation)
- Transfer learning with ResNet50
- Early stopping and model checkpointing
- Evaluation with **confusion matrix** and **ROC curve**
- Predict pneumonia on new images

---

## Folder Structure

```text
Medical_Pneumonia_Detection/
│
├── data/                # Dataset directory
├── src/                 # Source code
├── saved_models/        # Trained model weights
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

```
## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python src/train.py
```

3. Evaluate:

```bash
python src/evaluate.py
```
4. Predict new images:

```bash
python src/predict.py
```

Dataset
Chest X-Ray Images (Pneumonia) dataset from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
