import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Environment-driven model directory (default: ./model)
MODEL_DIR = os.getenv('MODEL_DIR', './model')

# Load models and scalers
crop_model = joblib.load(os.path.join(MODEL_DIR, 'crop_recommendation_model.pkl'))
crop_scaler = joblib.load(os.path.join(MODEL_DIR, 'crop_scaler.pkl'))
fertilizer_model = joblib.load(os.path.join(MODEL_DIR, 'fertilizer_recommendation_model.pkl'))
fertilizer_scaler = joblib.load(os.path.join(MODEL_DIR, 'fertilizer_scaler.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'crop-fertilizer-api'}), 200

@app.route('/predict/crop', methods=['POST'])
def predict_crop():
    try:
        data = request.get_json()
        # Extract features (adjust based on your model)
        features = np.array([[
            data['nitrogen'],
            data['phosphorus'],
            data['potassium'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]])
        
        # Scale and predict
        scaled_features = crop_scaler.transform(features)
        prediction = crop_model.predict(scaled_features)
        
        return jsonify({
            'recommended_crop': str(prediction[0]),
            'status': 'success'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

@app.route('/predict/fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        data = request.get_json()
        # Extract features (adjust based on your model)
        features = np.array([[
            data['nitrogen'],
            data['phosphorus'],
            data['potassium'],
            data['temperature'],
            data['humidity'],
            data['moisture']
        ]])
        
        # Scale and predict
        scaled_features = fertilizer_scaler.transform(features)
        prediction = fertilizer_model.predict(scaled_features)
        
        return jsonify({
            'recommended_fertilizer': str(prediction[0]),
            'status': 'success'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

if __name__ == '__main__':
    # Only for local development
    app.run(debug=True, host='0.0.0.0', port=5000)
