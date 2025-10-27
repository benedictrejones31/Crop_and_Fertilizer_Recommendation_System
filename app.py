import os
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins for development

# Environment-driven model paths (works in Docker)
MODEL_DIR = os.getenv('MODEL_DIR', './model')

# Load models with relative paths
crop_model = joblib.load(os.path.join(MODEL_DIR, 'crop_recommendation_model.pkl'))
crop_scaler = joblib.load(os.path.join(MODEL_DIR, 'crop_scaler.pkl'))

fertilizer_model = joblib.load(os.path.join(MODEL_DIR, 'fertilizer_recommendation_model.pkl'))
fertilizer_scaler = joblib.load(os.path.join(MODEL_DIR, 'fertilizer_scaler.pkl'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])
        
        # Create a single array for both crop and fertilizer prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # ----------------------------------------------
        # Crop Prediction
        # ----------------------------------------------
        # Use relevant columns for crop prediction (N, P, K, Temperature, Humidity, pH, Rainfall)
        input_data_crop = input_data[:, [0, 1, 2, 3, 4, 5, 6]]  # Keep all columns

        # Scale the input data for crop recommendation
        input_data_crop_scaled = crop_scaler.transform(input_data_crop)

        # Predict the crop
        predicted_crop = crop_model.predict(input_data_crop_scaled)

        # ----------------------------------------------
        # Fertilizer Prediction
        # ----------------------------------------------
        # Use relevant columns for fertilizer prediction (N, P, K, pH, Rainfall, Temperature)
        input_data_fertilizer = input_data[:, [0, 1, 2, 5, 6, 3]]  # Rearrange columns for fertilizer

        # Scale the input data for fertilizer recommendation
        input_data_fertilizer_scaled = fertilizer_scaler.transform(input_data_fertilizer)

        # Predict the fertilizer
        predicted_fertilizer = fertilizer_model.predict(input_data_fertilizer_scaled)

        # Return the predictions to the HTML page
        return render_template('index.html', crop=predicted_crop[0], fertilizer=predicted_fertilizer[0])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
