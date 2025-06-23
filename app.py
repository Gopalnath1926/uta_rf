# app.py
import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="UTA Retention Predictor", layout="centered")
st.title("🎓 Student Retention Probability Predictor")

st.markdown("Enter student information below to predict the probability of retention.")

# Define mappings for categorical fields
categorical_mappings = {
    "Gender": {"Female": 0, "Male": 1},
    # Add more mappings here if needed
}

# Define input UI
def get_user_input():
    input_dict = {}

    for feature in model.feature_names_in_:
        if feature in categorical_mappings:
            options = list(categorical_mappings[feature].keys())
            user_choice = st.selectbox(f"{feature}", options)
            input_dict[feature] = categorical_mappings[feature][user_choice]
        else:
            input_dict[feature] = st.number_input(f"{feature}", value=0.0)
    
    return pd.DataFrame([input_dict])

# Collect input
input_df = get_user_input()

# Predict
if st.button("🔍 Predict Retention"):
    proba = model.predict_proba(input_df)[0][1]
    st.success(f"🎯 Predicted Retention Probability: **{proba * 100:.2f}%**")
