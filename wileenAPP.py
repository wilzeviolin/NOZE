import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
import hydra
from omegaconf import DictConfig

# Initialize Flask app
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

@app.before_request
def check_model():
    global model
    if model is None:
        model = load_model(config.model.path)

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    global model
    # Load the model using the path from the config
    model = load_model(cfg.model.path)

    # Start Flask app
    port = cfg.server.port
    print(f"Starting Flask app on port {port}")
    app.run(host=cfg.server.host, port=port, debug=cfg.server.debug)

if __name__ == '__main__':
    main()
