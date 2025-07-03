import streamlit as st
import joblib
import pandas as pd

# Load models
flash_model = joblib.load("rf_flash_model.pkl")
fbp_model = joblib.load("rf_fbp_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔬 KERO Flash & FBP Predictor")

reflux = st.number_input("Reflux Flow", value=40.0)
kero_flow = st.number_input("Kero Draw Flow", value=22.0)
diesel_temp = st.number_input("Diesel Draw Temperature", value=230.0)
bottom_pr = st.number_input("Column Bottom Pressure", value=1.30)
reboil_flow = st.number_input("Kero Reboiling Flow", value=120.0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "agg_REFLUX FLOW": reflux,
        "agg_KERO DRAW FLOW": kero_flow,
        "agg_DIESEL DRAW TEMP": diesel_temp,
        "agg_BOTTOM PR": bottom_pr,
        "agg_KERO REBOILING FLOW": reboil_flow
    }])
    scaled = scaler.transform(input_df)
    flash = flash_model.predict(scaled)[0]
    fbp = fbp_model.predict(scaled)[0]
    st.success(f"Flash Point: {flash:.2f} °C")
    st.success(f"Final Boiling Point (FBP): {fbp:.2f} °C")
