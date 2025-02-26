import streamlit as st
import pickle
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    with open('seed_type_classification.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Streamlit UI
st.title("Wheat Kernel Type Classifier")
st.write("Enter kernel attributes to predict the wheat type (Kama, Rosa, Canadian)")

# Input fields
area = st.number_input("Area", min_value=0.0, max_value=21.18, value=10.0)
perimeter = st.number_input("Perimeter", min_value=0.0, max_value=17.25, value=10.0)
compactness = st.number_input("Compactness", min_value=0.0, max_value=0.9183, value=0.5)
length = st.number_input("Length", min_value=0.0, max_value=6.675, value=3.0)
width = st.number_input("Width", min_value=0.0, max_value=4.033, value=2.0)
asymmetry_coeff = st.number_input("Asymmetry Coefficient", min_value=0.0, max_value=8.315, value=4.0)
groove = st.number_input("Groove", min_value=0.0, max_value=6.55, value=3.0)
length_width_ratio = length / width if width != 0 else 0.0

# Predict button
if st.button("Predict Wheat Type"):
    # Prepare input for model
    features = np.array([[area, perimeter, compactness, length, width, asymmetry_coeff, groove, length_width_ratio]])

    # Perform prediction
    prediction = model.predict(features)[0]

    # Map numeric prediction to class name
    wheat_types = {1: 'Kama', 2: 'Rosa', 3: 'Canadian'}
    predicted_type = wheat_types.get(prediction, 'Unknown')

    st.success(f"Predicted Wheat Type: {predicted_type}")

st.write("Max values for each feature:")
st.json({
    "Area": 21.18,
    "Perimeter": 17.25,
    "Compactness": 0.9183,
    "Length": 6.675,
    "Width": 4.033,
    "Asymmetry Coefficient": 8.315,
    "Groove": 6.55
})
