import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
def load_model():
    try:
        with open("seed_pipeline.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Ensure 'seed_pipeline.pkl' is present.")
        st.stop()

model = load_model()

# App UI
st.title("Wheat Kernel Classifier")
st.write("Input wheat kernel measurements to predict the type (Kama, Rosa, or Canadian).")

# Input fields
area = st.number_input("Area", min_value=0.0, format="%f")
perimeter = st.number_input("Perimeter", min_value=0.0, format="%f")
compactness = st.number_input("Compactness", min_value=0.0, format="%f")
length = st.number_input("Length", min_value=0.0, format="%f")
width = st.number_input("Width", min_value=0.0, format="%f")
asymmetry = st.number_input("Asymmetry Coefficient", min_value=0.0, format="%f")
groove = st.number_input("Groove Length", min_value=0.0, format="%f")

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'Area': [area],
        'Perimeter': [perimeter],
        'Compactness': [compactness],
        'Length': [length],
        'Width': [width],
        'AsymmetryCoeff': [asymmetry],
        'Groove': [groove]
    })

    prediction = model.predict(input_data)
    label_map = {1: 'Kama', 2: 'Rosa', 3: 'Canadian'}
    st.success(f"Predicted Wheat Type: {label_map.get(prediction[0], 'Unknown')}")

# Debug info
st.write("Ensure the model is trained and available as 'seed_pipeline.pkl' in the root directory.")
