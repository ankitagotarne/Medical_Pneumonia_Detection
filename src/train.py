import os
from data_loader import get_data_generators
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths
BASE_DIR = os.path.join(os.getcwd(),"data")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_PATH = (os.path.join(os.getcwd(),"saved_models/best_model.h5"))

# Load data
train_gen, val_gen, test_gen = get_data_generators(TRAIN_DIR, VAL_DIR, TEST_DIR)

# Build model
model = build_model()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True)

# Train
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[early_stop, checkpoint]
)

# Save training history plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.savefig(os.path.join(os.getcwd(),"saved_models/training_plot.png"))
plt.show()
