import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns # Added for a better plot color palette

# --- App Configuration ---
st.set_page_config(
    page_title="Stellar Classification AI",
    layout="wide"
)

# --- Caching Functions for Performance ---
@st.cache_data
def load_data():
    """Loads the original star classification dataset and standardizes column names."""
    df = pd.read_csv('star_classification.csv')
    df.columns = df.columns.str.lower().str.strip() # Standardize column names
    return df

@st.cache_resource
def load_artifacts():
    """Loads the saved model, encoder, and scaler."""
    try:
        model = load_model('stellar_cnn_model.h5')
        label_encoder = joblib.load('label_encoder.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, label_encoder, scaler
    except FileNotFoundError:
        return None, None, None

def get_sdss_image(ra, dec): # The function arguments are still ra and dec, which is fine
    """Fetches an image from the SDSS SkyServer given its coordinates."""
    scale = 0.4
    width = 256
    height = 256
    url = f"http://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale={scale}&width={width}&height={height}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except requests.exceptions.RequestException:
        return None

# --- Load Data and Artifacts ---
df = load_data()
model, label_encoder, scaler = load_artifacts()

# --- Main App UI ---
st.title("Deep Learning for Stellar Classification")
st.write("This app uses a Neural Network to classify celestial objects as Galaxies, Quasars, or Stars based on their observational data from the Sloan Digital Sky Survey (SDSS).")

if model is None or label_encoder is None or scaler is None:
    st.error("Model, encoder, or scaler files not found. Please run the Jupyter notebook to generate these files.")
else:
    # --- Sidebar for User Input ---
    st.sidebar.title("Object Selector")
    test_set_size = int(len(df) * 0.2)
    test_df = df.tail(test_set_size)
    
    selected_index = st.sidebar.number_input(
        "Select an object to classify (by index):",
        min_value=int(test_df.index.min()),
        max_value=int(test_df.index.max()),
        value=int(test_df.index.min()),
        step=1
    )

    with st.sidebar.expander(" Legend"):
        st.markdown("""
        - **redshift**: A measure of how fast a distant object is moving away from us, used to determine its distance.
        - **u, g, r, i, z**: A set of five color filters measuring the object's brightness, from ultraviolet (`u`) to near-infrared (`z`).
        - **plate**: The ID for the photographic plate in the telescope that captured the light from a region of the sky.
        - **MJD (Modified Julian Date)**: A precise, scientific timestamp of when the object was observed.
        - **fiber_id**: The ID of the specific optical fiber that channeled the light from a single object to the spectrograph.
        """)


    # --- Display Selected Object Data ---
    selected_object = df.loc[selected_index]
    st.header(f"Analyzing Object ID: {selected_object['obj_id']}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Object Image")
        # THE FIX: Use 'alpha' and 'delta' columns to get the coordinates
        image = get_sdss_image(selected_object['alpha'], selected_object['delta'])
        if image:
            # THE FIX: Update the caption to use 'alpha' and 'delta'
            st.image(image, caption=f"SDSS Image of Object at Alpha: {selected_object['alpha']:.4f}, Delta: {selected_object['delta']:.4f}")
        else:
            st.warning("Could not retrieve image from SDSS server.")
            
        st.subheader("Observational Data")
        st.dataframe(selected_object[['redshift', 'u', 'g', 'r', 'i', 'z', 'plate', 'mjd', 'fiber_id']])
        

    with col2:
        st.subheader("AI Model Prediction")
        
        # Prepare the data for the model (must match the training process)
        features_to_predict = selected_object.drop(['obj_id', 'rerun_id', 'class'])
        features_scaled = scaler.transform([features_to_predict.values])
        
        # Make a prediction
        prediction_probs = model.predict(features_scaled)[0]
        predicted_class_index = np.argmax(prediction_probs)
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
        prediction_confidence = prediction_probs[predicted_class_index]
        
        # Display the results
        st.metric(
            label="Predicted Class",
            value=predicted_class_label
        )
        
        st.metric(
            label="Prediction Confidence",
            value=f"{prediction_confidence:.2%}"
        )

        st.markdown("---")
        
        st.subheader("Prediction Probabilities")
        
        prob_df = pd.DataFrame({
            'Class': label_encoder.classes_,
            'Probability': prediction_probs
        })
        
        fig, ax = plt.subplots()
        sns.barplot(x='Probability', y='Class', data=prob_df, ax=ax, orient='h', palette='viridis')
        ax.set_title("Probability for Each Class")
        ax.set_xlim(0, 1)
        st.pyplot(fig)
        
        st.markdown("---")
        st.info(f"**Actual Class:** {selected_object['class']}")