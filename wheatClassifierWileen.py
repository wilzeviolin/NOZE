import pickle
from flask import Flask, request, jsonify
import numpy as np

# Load the trained model
model = pickle.load(open("seed_type_classification.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the request
    data = request.get_json(force=True)
    
    # Extract the features
    area = data['Area']
    perimeter = data['Perimeter']
    compactness = data['Compactness']
    length = data['Length']
    width = data['Width']
    asymmetry_coeff = data['AsymmetryCoeff']
    groove = data['Groove']
    
    # Ensure the input features are within valid ranges
    if not (0 <= area <= 21.18):
        return jsonify({"error": "Area must be between 0 and 21.18"})
    if not (0 <= perimeter <= 17.25):
        return jsonify({"error": "Perimeter must be between 0 and 17.25"})
    if not (0 <= compactness <= 0.9183):
        return jsonify({"error": "Compactness must be between 0 and 0.9183"})
    if not (0 <= length <= 6.675):
        return jsonify({"error": "Length must be between 0 and 6.675"})
    if not (0 <= width <= 4.033):
        return jsonify({"error": "Width must be between 0 and 4.033"})
    if not (0 <= asymmetry_coeff <= 8.315):
        return jsonify({"error": "AsymmetryCoeff must be between 0 and 8.315"})
    if not (0 <= groove <= 6.55):
        return jsonify({"error": "Groove must be between 0 and 6.55"})
    
    # Prepare the input features as a numpy array for prediction
    features = np.array([[area, perimeter, compactness, length, width, asymmetry_coeff, groove]])
    
    # Make the prediction
    prediction = model.predict(features)
    
    # Return the prediction as a response
    return jsonify({"predicted_wheat_type": int(prediction[0])})


