import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache_resource
def load_lstm_model():
    model = load_model("temp.h5")
    return model

model = load_lstm_model()

# Page Title
st.title("Temperature Prediction App")
st.write("Enter the last 5 temperature values to predict the next day's temperature.")

# Input fields for the 5 temperature values
input_values = []
for i in range(1, 6):
    value = st.number_input(f"Temperature {i}", value=0.0, step=0.1, format="%.1f")
    input_values.append(value)

if st.button("Predict Next Day's Temperature"):
    # Reshape input for the LSTM model
    input_array = np.array(input_values).reshape(1, 5, 1)  # Shape: (1, timesteps, features)

    # Make prediction
    prediction = model.predict(input_array)

    # Extract the scalar value and format it
    predicted_temperature = float(prediction[0][0])  # Ensure it's a float
    st.success(f"The predicted temperature for the next day is: {predicted_temperature:.2f}°C")

