from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
with open('heart_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return "Server is working", 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Ensure all fields are present
    try:
        features = np.array([[ 
        float(data['age']), float(data['sex']), float(data['cp']), float(data['trestbps']), float(data['chol']),
        float(data['fbs']), float(data['restecg']), float(data['thalach']), float(data['exang']),
        float(data['oldpeak']), float(data['slope']), float(data['ca']), float(data['thal'])
    ]])

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400

    # Prediction and probability
    probability = model.predict_proba(features)[0][1]  # Prob of class 1 (Heart Disease)
    prediction = int(probability >= 0.5)

    return jsonify({'prediction': prediction, 'probability': float(probability)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
