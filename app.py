
import streamlit as st
import pandas as pd
import joblib

# Load model and column names
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Retention Probability Predictor")

st.markdown("Enter student details below:")

def user_input_features():
    input_data = {}
    for col in model_columns:
        if col.startswith("Gender_"):
            gender = st.selectbox("Gender", ["Male", "Female"])
            input_data["Gender_Male"] = 1 if gender == "Male" else 0
            input_data["Gender_Female"] = 1 if gender == "Female" else 0
            break
        elif col.startswith("CapFlag_"):
            cap = st.selectbox("Cap Flag", ["Y", "N"])
            input_data["CapFlag_Y"] = 1 if cap == "Y" else 0
            input_data["CapFlag_N"] = 1 if cap == "N" else 0
            break
        elif col.startswith("ExtraCurricularActivities_"):
            extra = st.selectbox("Extra Curricular Activities", ["Y", "N"])
            input_data["ExtraCurricularActivities_Y"] = 1 if extra == "Y" else 0
            input_data["ExtraCurricularActivities_N"] = 1 if extra == "N" else 0
            break
        elif col.startswith("PellEligibility_"):
            pell = st.selectbox("Pell Eligibility", ["Y", "N"])
            input_data["PellEligibility_Y"] = 1 if pell == "Y" else 0
            input_data["PellEligibility_N"] = 1 if pell == "N" else 0
            break
        elif col.startswith("FirstTermEnrolledCollege_"):
            colleges = [
                "Col Nurse & Health Innovation",
                "College of Business",
                "College of Engineering",
                "College of Liberal Arts",
                "College of Science",
                "Division of Student Success",
            ]
            college = st.selectbox("First Term Enrolled College", colleges)
            for c in colleges:
                encoded_col = f"FirstTermEnrolledCollege_{c}"
                input_data[encoded_col] = 1 if c == college else 0
            break

    # Add numeric fields
    input_data["AdmitYear"] = st.number_input("Admit Year", min_value=2000, max_value=2030, value=2023)
    input_data["SatTot02"] = st.number_input("SAT Total", value=1000)
    input_data["FirstTermGPA"] = st.number_input("First Term GPA", min_value=0.0, max_value=4.0, value=2.5)
    input_data["TotalFamilyIncome"] = st.number_input("Total Family Income", value=40000)
    input_data["HsGPA"] = st.number_input("High School GPA", min_value=0.0, max_value=4.0, value=3.0)

    return pd.DataFrame([input_data])

input_df = user_input_features()

if st.button("Predict Retention Probability"):
    # Add missing columns if any
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]  # Ensure column order
    prob = model.predict_proba(input_df)[0, 1]
    st.success(f"Estimated Retention Probability: **{prob * 100:.2f}%**")
