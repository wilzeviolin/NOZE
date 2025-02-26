import streamlit as st
import pickle

# Load your model
@st.cache_resource
def load_model():
    with open('seed_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Use the model for predictions
st.title("Wheat Kernel Classifier")
area = st.number_input("Enter Area:")
perimeter = st.number_input("Enter Perimeter:")

if st.button("Predict"):
    prediction = model.predict([[area, perimeter, 0.87, 5.0, 2.5, 3.0, 4.5]])
    st.write(f"Predicted Class: {prediction[0]}")
