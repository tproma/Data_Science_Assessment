import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model

# Load the pre-trained model
model = load_model('D:\A_Category\Assesment\model\ensemble_model.h5')

# Define the class labels
class_labels = ['berry', 'bird', 'dog', 'flower'] 

# Define the image dimensions expected by the model
image_width, image_height = 256, 256

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(image_height, image_width ))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def predict_image_class(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    confidence = prediction[0][predicted_class]
    return predicted_label, confidence

# Path to the image you want to classify
image_path = 'D:\A_Category\Assesment\dataset_256X256\path\a.jpg'

# Perform image classification
predicted_label, confidence = predict_image_class(image_path)

print("Predicted class:", predicted_label)
print("Confidence:", confidence)
