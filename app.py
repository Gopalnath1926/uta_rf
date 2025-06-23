# app.py
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="UTA Retention Predictor", layout="centered")
st.title("🎓 Student Retention Probability Predictor")

st.markdown("Enter student information below to predict the probability of retention.")

# Generate inputs dynamically from model's expected features
def get_user_input():
    input_dict = {}
    for feature in model.feature_names_in_:
        input_dict[feature] = st.number_input(f"{feature}", value=0.0)
    return pd.DataFrame([input_dict])

# Input section
input_df = get_user_input()

# Prediction
if st.button("🔍 Predict Retention"):
    probability = model.predict_proba(input_df)[0][1]  # retention probability
    st.success(f"🎯 Predicted Retention Probability: **{probability * 100:.2f}%**")
