from flask import Flask
import os

from src.training_pipeline import train_image_Classification
from src.prediction_pipeline import predict_image_class


app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home():
    return "Image Classifiers"


@app.route('/pipelines', methods = ['GET'])
def pipelines():
    ensemble_model = train_image_Classification()
    predicted_label, confidence = predict_image_class()
    print( predicted_label )
    return "Training of Image Classifier completed"


if __name__=="__main__":
    app.run(debug = True, host="0.0.0.0", port = 5000)