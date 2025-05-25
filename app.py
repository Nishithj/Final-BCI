import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the trained model once
@st.cache_resource
def load_eeg_model():
    return load_model(r"\E:\sem8\Final\NEW_TRY\Final-BCI\eeg_model.h5")

model = load_eeg_model()

# Map class index to emoji and label
label_map = {
    0: "🧠👈 Left Hand",
    1: "🧠👉 Right Hand",
    2: "🦶 Foot",
    3: "👅 Tongue"
}

st.title("🧠 EEG Motor Imagery Classifier (Row-wise Prediction)")

uploaded_file = st.file_uploader("Upload EEG CSV file", type="csv")

def preprocess_single_row(row, eeg_channels):
    # Convert the row (Series) to numpy array (channels only)
    data = row[eeg_channels].values.astype(np.float32)
    # Reshape to (channels, time=1) -> since 1 row = 1 time step, but your model might expect more time points
    # If model expects multiple time points, you might need to handle differently.

    # For this example, assume each row is a time sample with all channels (shape: channels,)
    # EEGNet usually expects (batch, channels, time, 1)
    # So we reshape to (1, channels, 1, 1)
    data = data[:, np.newaxis]  # shape (channels, 1)
    data = data[np.newaxis, :, :, np.newaxis]  # shape (1, channels, 1, 1)
    return data

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Identify EEG channels columns, e.g. those starting with 'EEG-'
        eeg_channels = [col for col in df.columns if col.startswith("EEG-")]
        if not eeg_channels:
            st.error("No EEG channels found in CSV (expected columns starting with 'EEG-').")
        else:
            st.write(f"Found EEG channels: {eeg_channels}")
            st.write(f"Processing {len(df)} rows one by one with 5 seconds interval:")

            # Placeholder to update prediction text dynamically
            prediction_placeholder = st.empty()

            for idx, row in df.iterrows():
                input_data = preprocess_single_row(row, eeg_channels)
                preds = model.predict(input_data)
                pred_class = int(np.argmax(preds))
                prediction_text = f"Row {idx+1} prediction: {label_map[pred_class]}"
                prediction_placeholder.text(prediction_text)

                time.sleep(5)  # wait for 5 seconds before next prediction

            st.success("Done with all predictions!")

    except Exception as e:
        st.error(f"Error processing the file: {e}")
