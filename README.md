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

PneumoniaDetection/
│
├── data/ # Dataset (not included)
├── src/ # Code: data loading, modeling, training, evaluation, prediction
├── saved_models/ # Trained model weights
├── requirements.txt
└── README.md

yaml
Copy code

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
Copy code
python src/train.py
```

3. Evaluate:

```bash
Copy code
python src/evaluate.py
```
4. Predict new images:

```bash
Copy code
python src/predict.py
```

Dataset
Chest X-Ray Images (Pneumonia) dataset from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia