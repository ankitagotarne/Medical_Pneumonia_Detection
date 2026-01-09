import numpy as np
from tensorflow.keras.models import load_model
from data_loader import get_data_generators
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import os

# Paths
BASE_DIR = "../data"
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_PATH = "../saved_models/best_model.h5"

# Load model
model = load_model(MODEL_PATH)

# Load test data
_, _, test_gen = get_data_generators(BASE_DIR+"/train", BASE_DIR+"/val", TEST_DIR)
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = (y_pred_probs > 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
