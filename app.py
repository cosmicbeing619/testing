import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Ksat Predictor", layout="centered")
st.title("🌱 Soil Saturated Hydraulic Conductivity Predictor")

model_path = "ksat_model.joblib"

if os.path.exists(model_path):
    # Load the model
    model = joblib.load(model_path)

    st.subheader("🔢 Enter soil properties:")

    user_input_features = [
        "bulkdensity", "clay(%)", "medium",
        "organiccarbon", "sand(%)", "silt(%)", "veryfine"
    ]

    with st.form("ksat_input_form"):
        user_inputs = {}
        for feature in user_input_features:
            user_inputs[feature] = st.number_input(feature, value=0.0, format="%.4f")
        submitted = st.form_submit_button("Predict Ksat")

    if submitted:
        try:
            input_vector = {feat: 0 for feat in user_input_features}
            for feat in user_input_features:
                input_vector[feat] = user_inputs[feat]

            input_df = pd.DataFrame([input_vector])
            prediction = model.predict(input_df)[0]
            st.success(f"🌊 Predicted Ksat: {prediction:.6f} cm/hr")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.warning("⚠ Model not found. Please ensure 'best_rf_model.joblib' is in the project folder.")
