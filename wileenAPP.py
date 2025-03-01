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
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    
@hydra.main(config_path="config", config_name="wheat.yaml")
def check_model(config):
    model_path = config.model.path  # Retrieve model path from config
    model = load_model(model_path)  # Pass the model path to load_model function
    if model is None:
        print("Model not loaded")
    else:
        print("Model loaded successfully")
        
# Initialize model to None initially
model = None

# Check the model before processing requests
@app.before_request
def before_request():
    global model
    if model is None:
        check_model()

@app.route('/', methods=['GET'])
def home_page():
    return render_template('wheat.html')

@app.route('/process', methods=['POST'])
def process_form():
    if model is None:
        return jsonify({"error": "Model not loaded"})

    try:
        area = float(request.form['area'])
        perimeter = float(request.form['perimeter'])
        compactness = float(request.form['compactness'])
        length = float(request.form['length'])
        width = float(request.form['width'])
        asymmetry_coeff = float(request.form['asymmetry_coeff'])
        groove = float(request.form['groove'])
        length_width_ratio = length / width if width != 0 else 0

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

        prediction = int(model.predict(features_df)[0])
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
