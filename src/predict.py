import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "../saved_models/best_model.h5"
IMG_SIZE = (224,224)

model = load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred_prob = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if pred_prob > 0.5 else "NORMAL"
    print(f"{img_path} --> {label} (Prob={pred_prob:.2f})")

# Example
# predict_image("../data/test/NORMAL/IM-0001-0001.jpeg")
