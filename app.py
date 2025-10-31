import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="CKD Prediction", page_icon="🩺", layout="centered")

# ==========================================
# Load models and scaler
# ==========================================
try:
    scaler = pickle.load(open("models/scaler.pkl", 'rb'))
    model_rf = pickle.load(open("models/model_randomforest.pkl", 'rb'))
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")
    st.stop()

# ==========================================
# Prediction Function
# ==========================================
def predict_chronic_disease(model, age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc):
    df = pd.DataFrame({
        'age': [age],
        'bp': [bp],
        'sg': [sg],
        'al': [al],
        'hemo': [hemo],
        'sc': [sc],
        'htn': [1 if htn == 'yes' else 0],
        'dm': [1 if dm == 'yes' else 0],
        'cad': [1 if cad == 'yes' else 0],
        'appet': [1 if appet == 'good' else 0],
        'pc': [1 if pc == 'normal' else 0]
    })

    df[['age', 'bp', 'sg', 'al', 'hemo', 'sc']] = scaler.transform(df[['age', 'bp', 'sg', 'al', 'hemo', 'sc']])
    prediction = model.predict(df)
    return prediction[0]

# ==========================================
# Streamlit UI
# ==========================================
st.title("🩺 Chronic Kidney Disease Prediction")
st.markdown("Enter patient details below to predict whether the patient has CKD or not using **Random Forest Classifier**.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 120, 48)
    bp = st.number_input("Blood Pressure", 40, 200, 80)
    sg = st.number_input("Specific Gravity", 1.005, 1.050, 1.020)
    al = st.number_input("Albumin", 0.0, 5.0, 1.0)
    hemo = st.number_input("Hemoglobin", 5.0, 20.0, 15.4)
    sc = st.number_input("Serum Creatinine", 0.5, 10.0, 1.2)

with col2:
    htn = st.selectbox("Hypertension", ["yes", "no"])
    dm = st.selectbox("Diabetes", ["yes", "no"])
    cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
    appet = st.selectbox("Appetite", ["good", "poor"])
    pc = st.selectbox("Protein in Urine", ["normal", "abnormal"])

# ==========================================
# Prediction Button
# ==========================================
if st.button("Predict"):
    result = predict_chronic_disease(model_rf, age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc)
    if result == 1:
        st.error("⚠️ The patient **has Chronic Kidney Disease (CKD)** — Predicted by Random Forest.")
    else:
        st.success("🩺 The patient **does not have Chronic Kidney Disease (CKD)** — Predicted by Random Forest.")
