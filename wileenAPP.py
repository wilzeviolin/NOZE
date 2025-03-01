import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Global model variable
model = None

# Function to load the model from a file path
def load_model(model_path):
    try:
        # Try direct path
        print(f"Attempting to load model from: {model_path}")
        if os.path.exists(model_path):
            print(f"Model file found at: {model_path}")
            with open(model_path, 'rb') as model_file:
                return pickle.load(model_file)
        
        # Try absolute path from project root
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
        absolute_path = os.path.join(base_dir, model_path)
        print(f"Trying absolute path: {absolute_path}")
        if os.path.exists(absolute_path):
            print(f"Model file found at: {absolute_path}")
            with open(absolute_path, 'rb') as model_file:
                return pickle.load(model_file)
                
        # Try with hydra output directory handling
        hydra_output_dir = os.path.join(os.getcwd(), "..", "..")  # Navigate up from Hydra's output dir
        hydra_path = os.path.join(hydra_output_dir, model_path)
        print(f"Trying hydra-adjusted path: {hydra_path}")
        if os.path.exists(hydra_path):
            print(f"Model file found at: {hydra_path}")
            with open(hydra_path, 'rb') as model_file:
                return pickle.load(model_file)
                
        print("Model file not found in any attempted location")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Create Flask app
app = Flask(__name__, template_folder='templates')

# Initialize app with Hydra config
@hydra.main(config_path="config", config_name="wheat.yaml")
def init_app(config: DictConfig):
    global model
    # Print the full config for debugging
    print(OmegaConf.to_yaml(config))
    
    # Get model path from config
    model_path = config.model.path
    print(f"Config model path: {model_path}")
    
    # Load the model
    model = load_model(model_path)
    if model is None:
        print("WARNING: Model not loaded! Application will return errors.")
    else:
        print("Model loaded successfully")
    
    # Configure Flask app from config
    port = config.server.port
    host = config.server.host
    debug = config.server.debug
    
    # Run the Flask app
    print(f"Starting Flask app on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

# Routes
@app.route('/', methods=['GET'])
def home_page():
    if model is None:
        return "Error: Model not loaded. Please check server logs.", 500
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
    # This will call the Hydra-decorated function
    init_app()
