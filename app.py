import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler_minmax.pkl")
model = joblib.load("chaos_regressor.pkl")

st.title("Human Vital Signs Risk Prediction")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", options=["Male", "Female"])
heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, value=70)
resp_rate = st.number_input("Respiratory Rate", min_value=5, max_value=60, value=15)
body_temp = st.number_input("Body Temperature", min_value=34.0, max_value=42.0, value=37.0)
oxygen = st.number_input("Oxygen Saturation", min_value=50, max_value=100, value=98)
sys_bp = st.number_input("Systolic Blood Pressure", min_value=50, max_value=250, value=120)
dia_bp = st.number_input("Diastolic Blood Pressure", min_value=30, max_value=150, value=80)
weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)
height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7)
hrv = st.number_input("Derived HRV", min_value=0.0, max_value=100.0, value=50.0)
pulse_pressure = st.number_input("Derived Pulse Pressure", min_value=0.0, max_value=100.0, value=40.0)
bmi = st.number_input("Derived BMI", min_value=10.0, max_value=50.0, value=22.0)
map_val = st.number_input("Derived MAP", min_value=30.0, max_value=150.0, value=90.0)

# Map gender to numeric
gender_num = 1 if gender == "Male" else 0

input_array = np.array([[heart_rate, resp_rate, body_temp, oxygen, sys_bp, dia_bp,
                        age, weight, height, hrv, pulse_pressure, bmi, map_val, gender_num]])

# Normalize with MinMaxScaler and apply logistic map transform (r=3.7)
input_scaled = scaler.transform(input_array)

def logistic_map_transform(X, r=3.7):
    return r * X * (1 - X)

input_chaos = logistic_map_transform(input_scaled)

# Prediction
risk_score = model.predict(input_chaos)[0]

# Risk band categorization
def risk_band(score):
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Medium"
    else:
        return "High"

risk_level = risk_band(risk_score)

st.write(f"Predicted Deterioration Score: {risk_score:.2f}")
st.write(f"Risk Level: {risk_level}")
