import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Get class names
data_dir = "dataset"
class_names = sorted(os.listdir(data_dir))

IMG_SIZE = 224

# Change this to any image path you want to test
img_path = "test.jpg"

img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print("Predicted Disease:", predicted_class)