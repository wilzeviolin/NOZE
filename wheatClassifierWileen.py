import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
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
    st.image("https://cdn.pixabay.com/photo/2016/09/12/11/25/wheat-1663713_1280.jpg", width=300)
    st.title("Navigation")
    page = st.radio("Go to", ["Predict", "Dataset Info", "Visualization"])
    
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

# Function to create visualizations
def create_visualization(feature_values):
    # Create sample data similar to what we've seen
    sample_data = {
        'Type': [0, 0, 0, 1, 1, 1, 2, 2, 2],  # 0=Kama, 1=Rosa, 2=Canadian
        'Length': [5.5, 5.6, 5.7, 6.2, 6.3, 6.4, 5.1, 5.2, 5.3],
        'Width': [3.2, 3.3, 3.4, 3.7, 3.8, 3.9, 2.8, 2.9, 3.0],
        'Compactness': [0.88, 0.87, 0.89, 0.89, 0.90, 0.88, 0.84, 0.85, 0.83]
    }
    df = pd.DataFrame(sample_data)
    
    # Add the input data point
    input_df = pd.DataFrame({
        'Type': [3],  # Different type to highlight it
        'Length': [feature_values['length']],
        'Width': [feature_values['width']],
        'Compactness': [feature_values['compactness']]
    })
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Length vs Width scatter plot
    ax1.scatter(df[df['Type'] == 0]['Length'], df[df['Type'] == 0]['Width'], color='#88d8b0', label='Kama')
    ax1.scatter(df[df['Type'] == 1]['Length'], df[df['Type'] == 1]['Width'], color='#ff9966', label='Rosa')
    ax1.scatter(df[df['Type'] == 2]['Length'], df[df['Type'] == 2]['Width'], color='#6699cc', label='Canadian')
    ax1.scatter(input_df['Length'], input_df['Width'], color='red', marker='*', s=200, label='Your Sample')
    ax1.set_xlabel('Length')
    ax1.set_ylabel('Width')
    ax1.set_title('Length vs Width by Wheat Type')
    ax1.legend()
    
    # Compactness box plot
    ax2.boxplot([df[df['Type'] == 0]['Compactness'], 
                 df[df['Type'] == 1]['Compactness'], 
                 df[df['Type'] == 2]['Compactness']], 
                 labels=['Kama', 'Rosa', 'Canadian'])
    ax2.axhline(y=input_df['Compactness'].values[0], color='r', linestyle='-', label='Your Sample')
    ax2.set_ylabel('Compactness')
    ax2.set_title('Compactness Distribution by Wheat Type')
    
    # Adjust layout
    plt.tight_layout()
    return fig

# Define default value ranges for each feature
feature_ranges = {
    "area": (10.0, 20.0, 15.0, 10.0, 25.0),  # min, max, default, step, step count
    "perimeter": (10.0, 20.0, 15.0, 0.1, 100),
    "compactness": (0.8, 0.95, 0.88, 0.01, 15),
    "length": (5.0, 7.0, 5.7, 0.05, 40),
    "width": (2.6, 4.0, 3.3, 0.05, 28),
    "asymmetry_coefficient": (0.5, 10.0, 3.0, 0.1, 95),
    "groove_length": (4.0, 7.0, 5.5, 0.05, 60)
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
            min_val, max_val, default, step, _ = feature_ranges[feature]
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
            min_val, max_val, default, step, _ = feature_ranges[feature]
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
                
                # Show probabilities as a bar chart
                prob_df = pd.DataFrame({
                    'Wheat Type': wheat_types,
                    'Probability': probabilities
                })
                st.markdown("### Prediction Probabilities")
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.bar(prob_df['Wheat Type'], prob_df['Probability'], color=['#88d8b0', '#ff9966', '#6699cc'])
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Probabilities by Wheat Type')
                
                # Add probability values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Show wheat type description
                st.markdown(f"""
                <div class="wheat-info">
                    <h3>About {pred_type} Wheat:</h3>
                    {wheat_descriptions[prediction]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show visualizations of the input compared to typical values
                st.markdown("### Visualizing Your Sample")
                st.pyplot(create_visualization(feature_values))
                
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
    
    # Show sample data distribution visualization
    st.markdown("### Sample Data Visualizations")
    
    # Create sample data
    sample_data = {
        'Type': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],  # 0=Kama, 1=Rosa, 2=Canadian
        'Length': [5.5, 5.6, 5.7, 5.5, 5.6, 6.2, 6.3, 6.4, 6.3, 6.1, 5.1, 5.2, 5.3, 5.0, 5.2],
        'Width': [3.2, 3.3, 3.4, 3.3, 3.2, 3.7, 3.8, 3.9, 3.7, 3.8, 2.8, 2.9, 3.0, 2.7, 2.9],
        'Compactness': [0.88, 0.87, 0.89, 0.88, 0.89, 0.89, 0.90, 0.88, 0.89, 0.88, 0.84, 0.85, 0.83, 0.84, 0.85]
    }
    df = pd.DataFrame(sample_data)
    
    # Convert numeric type to categorical names for better visualization
    df['Wheat Type'] = df['Type'].map({0: 'Kama', 1: 'Rosa', 2: 'Canadian'})
    
    # Length vs Width scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    colors = ['#88d8b0', '#ff9966', '#6699cc']
    for i, wheat in enumerate(['Kama', 'Rosa', 'Canadian']):
        subset = df[df['Wheat Type'] == wheat]
        ax1.scatter(subset['Length'], subset['Width'], color=colors[i], label=wheat)
    
    ax1.set_xlabel('Length')
    ax1.set_ylabel('Width')
    ax1.set_title('Length vs Width by Wheat Type')
    ax1.legend()
    
    # Compactness box plot
    for i, wheat in enumerate(['Kama', 'Rosa', 'Canadian']):
        subset = df[df['Wheat Type'] == wheat]
        ax2.boxplot([subset['Compactness']], positions=[i+1], widths=0.6, 
                    patch_artist=True, boxprops=dict(facecolor=colors[i]))
    
    ax2.set_xticklabels(['Kama', 'Rosa', 'Canadian'])
    ax2.set_ylabel('Compactness')
    ax2.set_title('Compactness Distribution by Wheat Type')
    
    plt.tight_layout()
    st.pyplot(fig)

# Visualization page
elif page == "Visualization":
    st.markdown('<h2 class="subheader">Interactive Visualization</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Use the sliders below to see how changing kernel measurements affects classification.
    This visualization helps understand the decision boundaries between wheat varieties.
    """)
    
    # Create two columns for the visualization page
    viz_col1, viz_col2 = st.columns([1, 2])
    
    with viz_col1:
        st.markdown("### Adjust Parameters")
        # Allow users to adjust key parameters
        length = st.slider("Length", 5.0, 7.0, 5.7, 0.1)
        width = st.slider("Width", 2.6, 4.0, 3.3, 0.1)
        compactness = st.slider("Compactness", 0.8, 0.95, 0.88, 0.01)
        
        # Use default values for other parameters
        feature_values = {
            "area": 15.0,
            "perimeter": 15.0,
            "compactness": compactness,
            "length": length,
            "width": width,
            "asymmetry_coefficient": 3.0,
            "groove_length": 5.5
        }
        
        # Create input DataFrame for prediction
        viz_input = pd.DataFrame([feature_values])
        
        # Make prediction for visualization
        if model is not None:
            try:
                prediction = model.predict(viz_input)[0]
                probabilities = model.predict_proba(viz_input)[0]
                
                wheat_types = ["Kama", "Rosa", "Canadian"]
                pred_type = wheat_types[prediction]
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Predicted Type:</h3>
                    <h2 style="color: #2980b9;">{pred_type}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Show probabilities as horizontal bars
                st.markdown("### Probabilities:")
                for i, wheat in enumerate(wheat_types):
                    st.markdown(f"**{wheat}**: {probabilities[i]:.4f}")
                    st.progress(float(probabilities[i]))
            except Exception as e:
                st.error(f"Visualization prediction error: {e}")
    
    with viz_col2:
        st.markdown("### Visual Representation")
        # Create and display the visualization
        try:
            viz_fig = create_visualization(feature_values)
            st.pyplot(viz_fig)
            
            # Add explanation
            st.markdown("""
            ### Interpretation Guide:
            
            - **Left Plot**: Shows how your sample (red star) compares to typical length and width measurements of each wheat variety.
            - **Right Plot**: Shows how your sample's compactness (red line) compares to the typical distribution of compactness values for each wheat variety.
            
            The closer your sample is to a particular cluster or distribution, the more likely it belongs to that wheat variety.
            """)
        except Exception as e:
            st.error(f"Error creating visualization: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Wheat Kernel Classification App | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)