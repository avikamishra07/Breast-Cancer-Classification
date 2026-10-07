import streamlit as st
import numpy as np
import joblib

model = joblib.load("best_breast_cancer_model.pkl")
scaler = joblib.load("breast_cancer_scaler.pkl")
feature_columns = joblib.load("breast_cancer_features.pkl")
model_name = joblib.load("breast_cancer_model_name.pkl")

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️"
)

st.title("🎗️ Breast Cancer Tumor Prediction")
st.write("Enter the tumor features to predict whether the tumor is Benign or Malignant.")

st.divider()

inputs = []

for feature in feature_columns:
    value = st.number_input(
        feature,
        value=0.0
    )
    inputs.append(value)

st.divider()

if st.button("🔍 Predict Tumor", use_container_width=True):

    data = np.array(inputs).reshape(1, -1)

    if model_name in [
        "Logistic Regression",
        "KNN",
        "SVM"
    ]:
        data = scaler.transform(data)

    prediction = model.predict(data)[0]

    if prediction == 0:
        st.success("Prediction: Benign")
    else:
        st.error("Prediction: Malignant")

    st.info("Model Used: " + model_name)