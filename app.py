import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
@st.cache_resource
def load_lstm_model():
    model = load_model("temp.h5")
    return model

model = load_lstm_model()

# Initialize the scaler (same feature range as used in training)
sc = MinMaxScaler(feature_range=(0, 1))

# Page Title
st.title("Temperature Prediction App")
st.write("Enter the last 5 temperature values to predict the next day's temperature.")

# Input fields for the 5 temperature values
input_values = []
for i in range(1, 6):
    value = st.number_input(f"Temperature {i}", value=0.0, step=0.1, format="%.1f")
    input_values.append(value)

# Predict button
if st.button("Predict Next Day's Temperature"):
    # Reshape input for the LSTM model
    input_array = np.array(input_values).reshape(1, 5, 1)  # Shape: (1, timesteps, features)

    # Scale the input values using the same scaler as in training
    input_array_scaled = sc.fit_transform(np.array(input_values).reshape(-1, 1)).reshape(1, 5, 1)

    # Make prediction
    prediction_scaled = model.predict(input_array_scaled)

    # Inverse transform the prediction to get the original scale
    predicted_temperature = sc.inverse_transform(prediction_scaled)[0][0]  # Reshape to match scaler

    # Display the result
    st.success(f"The predicted temperature for the next day is: {predicted_temperature:.2f}°C")
