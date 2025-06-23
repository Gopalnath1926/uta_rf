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

# Define categorical options (from training)
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

# Use 2 columns for layout
col1, col2 = st.columns(2)

# Collect user input
def get_user_input():
    input_data = {}
    for i, feature in enumerate(model.feature_names_in_):
        col = col1 if i % 2 == 0 else col2
        if feature in categorical_options:
            choice = col.selectbox(f"{feature}", categorical_options[feature])
            input_data[feature] = categorical_options[feature].index(choice)
        else:
            input_data[feature] = col.number_input(f"{feature}", value=0.0)
    return pd.DataFrame([input_data])

input_df = get_user_input()

if st.button("🔍 Predict Retention"):
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"🎯 Predicted Retention Probability: **{probability * 100:.2f}%**")
