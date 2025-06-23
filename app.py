# app.py
import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="UTA Retention Predictor", layout="centered")
st.title("🎓 UTA Student Retention Probability Predictor")

st.markdown("Fill in the student information below:")

# Categorical values mapping (based on training data)
categorical_options = {
    'Gender': ['Male', 'Female'],
    'CapFlag': ['N', 'Y'],
    'ExtraCurricularActivities': ['N', 'Y'],
    'PellEligibility': ['Y', 'N'],
    'FirstTermEnrolledCollege': [
        'College of Business',
        'College of Science',
        'Col Nurse & Health Innovation',
        'College of Liberal Arts',
        'Division of Student Success',
        'College of Engineering'
    ],
    'AdmitYear': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
}

# Input form
def get_user_input():
    input_data = {}
    for feature in model.feature_names_in_:
        if feature in categorical_options:
            choice = st.selectbox(f"{feature}", categorical_options[feature])
            # Encode as integer code to match training
            input_data[feature] = categorical_options[feature].index(choice)
        else:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)
    return pd.DataFrame([input_data])

input_df = get_user_input()

if st.button("🔍 Predict Retention"):
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"🎯 Predicted Retention Probability: **{probability * 100:.2f}%**")

