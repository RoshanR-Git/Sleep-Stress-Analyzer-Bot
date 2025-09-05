import streamlit as st
import joblib
import os

# Page configuration
st.set_page_config(page_title="Sleep Stress Analyzer Bot 🤖", page_icon="🤖", layout="centered")

MODEL_FILES_TO_TRY = [
    "sleep_stress_pipeline.pkl",
    "sleep_stress_model.pkl",
    "sleep_stress_model.joblib",
    "sleep_stress_pipeline.joblib"
]

def load_model_try(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                mdl = joblib.load(p)
                return mdl, p, None
            except Exception as e:
                return None, p, f"Found file but failed to load: {e}"
    return None, None, "No model file found in working directory."

model, model_path, load_error = load_model_try(MODEL_FILES_TO_TRY)

st.sidebar.header("Model status")
if model is not None:
    st.sidebar.success(f"Model loaded from: {model_path}")
    # show basic info
    try:
        st.sidebar.write(f"Model type: `{type(model).__name__}`")
        # If it's an sklearn pipeline, show steps
        if hasattr(model, "steps"):
            st.sidebar.write("Pipeline steps:")
            for name, step in model.steps:
                st.sidebar.write(f"- {name}: {type(step).__name__}")
    except Exception:
        pass
else:
    st.sidebar.error(f"Model not loaded: {load_error}")

import streamlit as st
import pandas as pd
import joblib

# Load trained model pipeline
model = joblib.load("sleep_stress_pipeline.pkl")



# Title
st.title("Sleep Stress Analyzer Bot 🤖")
st.write("This app predicts stress levels based on sleep and lifestyle factors.")

# Input form
with st.form("user_input_form"):
    st.subheader("Enter Your Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=10, max_value=100, value=25, step=1)
    occupation = st.selectbox(
        "Occupation",
        ["Software Engineer", "Doctor", "Sales Representative", "Teacher", "Nurse", "Other"]
    )
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=12.0, value=7.0, step=0.1)
    quality_of_sleep = st.slider("Quality of Sleep (1 = poor, 10 = excellent)", min_value=1, max_value=10, value=5)
    physical_activity = st.number_input("Physical Activity Level (minutes/day)", min_value=0, max_value=300, value=30, step=5)

    systolic = st.number_input("Blood Pressure - Systolic", min_value=80, max_value=200, value=120, step=1)
    diastolic = st.number_input("Blood Pressure - Diastolic", min_value=50, max_value=150, value=80, step=1)
    blood_pressure = f"{systolic}/{diastolic}"

    bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75, step=1)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=5000, step=100)

    sleep_disorder = st.selectbox("Sleep Disorder", ["None", "Insomnia", "Sleep Apnea"])

    submitted = st.form_submit_button("Predict Stress Level")

# Process inputs when submitted
if submitted:
    # Create input DataFrame (same columns as training dataset except ID & Stress Level)
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Sleep Duration": sleep_duration,
        "Quality of Sleep": quality_of_sleep,
        "Physical Activity Level": physical_activity,
        "BMI Category": bmi_category,
        "Blood Pressure": blood_pressure,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps,
        "Sleep Disorder": None if sleep_disorder == "None" else sleep_disorder
    }])

    # Ensure the features match what the model expects
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.success(f"😃 Your predicted Stress Level is: **{prediction}**")
# Footer with your name
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        width: 100%;
        text-align: right;
        padding: 10px;
        font-size: 14px;
        color: gray;
    }
    </style>
    <div class="footer">
        Developed by <b>Roshan R</b>
    </div>
    """,
    unsafe_allow_html=True
)
