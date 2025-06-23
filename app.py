
import os
import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoders.joblib")

# Load model and encoders
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)

st.title("Student Retention Probability Predictor")

st.markdown("### Enter student information below:")

def user_input_features():
    input_data = {}
    for feature in model.feature_names_in_:
        if feature in label_encoders:
            options = label_encoders[feature].classes_.tolist()
            choice = st.selectbox(f"{feature}", options)
            input_data[feature] = label_encoders[feature].transform([choice])[0]
        else:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)
    return pd.DataFrame([input_data])

input_df = user_input_features()

if st.button("Predict Retention"):
    proba = model.predict_proba(input_df)[0, 1]
    st.success(f"Estimated Retention Probability: **{proba * 100:.2f}%**")
