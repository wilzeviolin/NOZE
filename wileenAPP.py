import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='templates')

# Load the trained model
def load_model(model_path):
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as model_file:
                return pickle.load(model_file)
        else:
            print(f"Model file not found at: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
    return None

# Global model variable
model = None

# Function to load the model from the config file path
def initialize_model(config):
    global model
    if model is None:
        model_path = config.model.path  # Retrieve model path from config
        print(f"Attempting to load model from: {model_path}")  # Debug print for model path
        model = load_model(model_path)  # Pass the model path to load_model function
        if model is None:
            print("Model not loaded")
        else:
            print("Model loaded successfully")

# Initialize the model using Hydra config
@hydra.main(config_path="config", config_name="wheat.yaml")
def setup(config: DictConfig):
    initialize_model(config)  # Initialize model once

# Run the Hydra initialization
setup()  # Call this once during app startup

# Check the model before processing requests
@app.before_request
def before_request():
    global model
    if model is None:
        print("Model is not loaded.")
        return jsonify({"error": "Model not loaded"}), 500  # Return 500 if model isn't loaded

@app.route('/', methods=['GET'])
def home_page():
    return render_template('wheat.html')

@app.route('/process', methods=['POST'])
def process_form():
    global model
    if model is None:
        return jsonify({"error": "Model not loaded"})

    try:
        # Retrieve features from the form
        area = float(request.form['area'])
        perimeter = float(request.form['perimeter'])
        compactness = float(request.form['compactness'])
        length = float(request.form['length'])
        width = float(request.form['width'])
        asymmetry_coeff = float(request.form['asymmetry_coeff'])
        groove = float(request.form['groove'])
        length_width_ratio = length / width if width != 0 else 0

        # Prepare the features for prediction
        features_df = pd.DataFrame({
            'Area': [area],
            'Perimeter': [perimeter],
            'Compactness': [compactness],
            'Length': [length],
            'Width': [width],
            'AsymmetryCoeff': [asymmetry_coeff],
            'Groove': [groove],
            'Length_Width_Ratio': [length_width_ratio]
        })

        # Make prediction
        prediction = int(model.predict(features_df)[0])
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
