import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io

# Set page configuration
st.set_page_config(
    page_title="Wheat Kernel Classifier",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS to style the app
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #3498db;
        margin-bottom: 1rem;
    }
    .feature-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    .wheat-info {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🌾 Wheat Kernel Classification</h1>', unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('seed_pipeline.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Sidebar for navigation
with st.sidebar:
    st.title("🌾 Wheat Classifier")
    page = st.radio("Navigation", ["Predict", "Dataset Info"])
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This app classifies wheat kernels into three varieties:
    - Kama (Type 1)
    - Rosa (Type 2) 
    - Canadian (Type 3)
    
    Based on kernel measurements.
    """)

# Feature descriptions
feature_descriptions = {
    "area": "The area of the wheat kernel",
    "perimeter": "The perimeter of the wheat kernel",
    "compactness": "A measure of how circular the kernel is (4*pi*area/perimeter^2)",
    "length": "The length of the kernel",
    "width": "The width of the kernel",
    "asymmetry_coefficient": "A measure of how asymmetrical the kernel is",
    "groove_length": "The length of the kernel groove"
}

# Wheat variety descriptions
wheat_descriptions = {
    0: """
    **Kama Wheat (Type 1)**: A wheat variety with medium-sized kernels, moderate compactness, 
    and balanced dimensions. Often used in traditional bread making.
    """,
    1: """
    **Rosa Wheat (Type 2)**: A wheat variety with larger kernels, higher compactness values, 
    and greater width. Often preferred for pastry flour due to its characteristics.
    """,
    2: """
    **Canadian Wheat (Type 3)**: A wheat variety with smaller, more elongated kernels and 
    the lowest compactness values. Known for high protein content and strong gluten strength, 
    often used for bread and pasta.
    """
}

# Define default value ranges for each feature
feature_ranges = {
    "area": (10.0, 20.0, 15.0, 0.1),  # min, max, default, step
    "perimeter": (10.0, 20.0, 15.0, 0.1),
    "compactness": (0.8, 0.95, 0.88, 0.01),
    "length": (5.0, 7.0, 5.7, 0.05),
    "width": (2.6, 4.0, 3.3, 0.05),
    "asymmetry_coefficient": (0.5, 10.0, 3.0, 0.1),
    "groove_length": (4.0, 7.0, 5.5, 0.05)
}

# Predict page
if page == "Predict":
    st.markdown('<h2 class="subheader">Enter Wheat Kernel Measurements</h2>', unsafe_allow_html=True)
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    # Dictionary to store feature values
    feature_values = {}
    
    # Column 1 features
    with col1:
        st.markdown('<div class="feature-container">', unsafe_allow_html=True)
        for feature in list(feature_ranges.keys())[:4]:
            min_val, max_val, default, step = feature_ranges[feature]
            feature_values[feature] = st.slider(
                f"{feature.replace('_', ' ').title()}: {feature_descriptions[feature]}",
                min_value=min_val,
                max_value=max_val,
                value=default,
                step=step
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Column 2 features
    with col2:
        st.markdown('<div class="feature-container">', unsafe_allow_html=True)
        for feature in list(feature_ranges.keys())[4:]:
            min_val, max_val, default, step = feature_ranges[feature]
            feature_values[feature] = st.slider(
                f"{feature.replace('_', ' ').title()}: {feature_descriptions[feature]}",
                min_value=min_val,
                max_value=max_val,
                value=default,
                step=step
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create input DataFrame for prediction
    input_df = pd.DataFrame([feature_values])
    
    # Make prediction
    if st.button("Classify Wheat Kernel", key="predict_button"):
        if model is not None:
            try:
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                
                wheat_types = ["Kama", "Rosa", "Canadian"]
                pred_type = wheat_types[prediction]
                
                # Show prediction with custom styling
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Prediction Result</h2>
                    <h3>This wheat kernel is classified as: 
                        <span style="color: #2980b9; font-weight: bold;">{pred_type} (Type {prediction+1})</span>
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Show probabilities as a horizontal bar chart using Streamlit
                st.markdown("### Prediction Probabilities")
                for i, wheat_type in enumerate(wheat_types):
                    prob = probabilities[i]
                    # Custom color for each wheat type
                    color = "#88d8b0" if wheat_type == "Kama" else "#ff9966" if wheat_type == "Rosa" else "#6699cc"
                    st.markdown(f"**{wheat_type}**: {prob:.4f}")
                    st.progress(float(prob))
                
                # Show wheat type description
                st.markdown(f"""
                <div class="wheat-info">
                    <h3>About {pred_type} Wheat:</h3>
                    {wheat_descriptions[prediction]}
                </div>
                """, unsafe_allow_html=True)
                
                # Simple visualizations using Streamlit components
                st.markdown("### Input Data Visualization")
                
                # Display the input values in a table for reference
                st.markdown("#### Your Input Values:")
                st.write(input_df)
                
                # Create a simple data comparison
                st.markdown("#### How Your Sample Compares:")
                comparison_data = {
                    "Feature": ["Length", "Width", "Compactness"],
                    "Your Sample": [feature_values["length"], feature_values["width"], feature_values["compactness"]],
                    "Kama Typical": [5.6, 3.3, 0.88],
                    "Rosa Typical": [6.3, 3.8, 0.89],
                    "Canadian Typical": [5.2, 2.9, 0.84]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.table(comparison_df)
                
                # Create a simple data representation of how close the sample is to each type
                st.markdown("#### Similarity to Each Wheat Type")
                st.write("The probabilities indicate how closely your sample matches each wheat type.")
                
                # Red to green scale color coding
                wheat_colors = {"Kama": "#88d8b0", "Rosa": "#ff9966", "Canadian": "#6699cc"}
                
                # Show colored indicators for closest match
                cols = st.columns(3)
                for i, (wheat, color) in enumerate(wheat_colors.items()):
                    with cols[i]:
                        prob = probabilities[wheat_types.index(wheat)]
                        st.markdown(f"""
                        <div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;">
                            <h4>{wheat}</h4>
                            <h3>{prob:.4f}</h3>
                            <p>{'✅ MATCH' if wheat == pred_type else ''}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.error("Model not loaded correctly. Please check the model file.")

# Dataset Info page
elif page == "Dataset Info":
    st.markdown('<h2 class="subheader">Wheat Kernel Dataset Information</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Dataset Description
    
    This dataset contains measurements of geometrical properties of kernels belonging to three different varieties of wheat:
    - Kama (Type 1)
    - Rosa (Type 2)
    - Canadian (Type 3)
    
    The dataset was sourced from the UCI Machine Learning Repository and is commonly used for classification tasks.
    
    ### Features
    
    The wheat kernels are described by 7 measurements:
    """)
    
    # Create a table of feature descriptions
    features_df = pd.DataFrame({
        'Feature': feature_descriptions.keys(),
        'Description': feature_descriptions.values()
    })
    st.table(features_df)
    
    st.markdown("""
    ### Wheat Varieties Characteristics
    
    Each wheat variety in this dataset has distinct characteristics:
    """)
    
    # Display wheat variety descriptions
    for type_id, description in wheat_descriptions.items():
        wheat_type = ["Kama", "Rosa", "Canadian"][type_id]
        st.markdown(f"**{wheat_type} (Type {type_id+1}):**")
        st.markdown(description)
    
    # Show typical values for each wheat type
    st.markdown("### Typical Measurements for Each Wheat Type")
    
    typical_data = {
        "Feature": list(feature_descriptions.keys()),
        "Kama (Type 1)": [15.0, 14.5, 0.88, 5.6, 3.3, 2.5, 5.2],
        "Rosa (Type 2)": [16.5, 15.5, 0.89, 6.3, 3.8, 3.0, 5.8],
        "Canadian (Type 3)": [13.5, 13.5, 0.84, 5.2, 2.9, 3.5, 4.8]
    }
    
    typical_df = pd.DataFrame(typical_data)
    st.table(typical_df)
    
    # Key differences
    st.markdown("""
    ### Key Differences Between Wheat Types
    
    - **Kama vs Rosa**: Rosa wheat kernels are generally larger (higher length and width) but have similar compactness to Kama.
    
    - **Kama vs Canadian**: Canadian wheat kernels are smaller and more elongated (lower width-to-length ratio) than Kama, with lower compactness.
    
    - **Rosa vs Canadian**: These represent the extremes in the dataset - Rosa kernels are the largest and most compact, while Canadian are the smallest and least compact.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Wheat Kernel Classification App | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)
