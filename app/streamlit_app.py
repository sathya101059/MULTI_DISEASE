import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Multi-Disease Prediction", layout="wide")

# Define feature lists
KIDNEY_FEATURES = [
    "age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc",
    "sod","pot","hemo","pcv","wc","rc","htn","dm","cad","appet","pe","ane"
]
LIVER_FEATURES = [
    "Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase",
    "Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens",
    "Albumin","Albumin_and_Globulin_Ratio"
]
PARKINSONS_FEATURES = [
    'MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)',
    'MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)',
    'Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR',
    'RPDE','DFA','spread1','spread2','D2','PPE'
]

# Load trained models
kidney_model = joblib.load("models/kidney_pipeline.pkl")
liver_model = joblib.load("models/liver_pipeline.pkl")
parkinsons_model = joblib.load("models/parkinsons_pipeline.pkl")

# Load metrics and feature importance JSON files
with open("models/model_metrics.json", "r") as f:
    metrics_data = json.load(f)

def load_feature_importances():
    data = {}
    try:
        with open("models/kidney_feature_importances.json") as f:
            data["Kidney Disease"] = json.load(f)
        with open("models/liver_feature_importances.json") as f:
            data["Liver Disease"] = json.load(f)
        with open("models/parkinsons_feature_importances.json") as f:
            data["Parkinson's Disease"] = json.load(f)
    except:
        pass
    return data

feature_importance_data = load_feature_importances()

# Default values map
defaults_map = {
    "Kidney": {k: 0 for k in KIDNEY_FEATURES},
    "Liver": {k: 0 for k in LIVER_FEATURES},
    "Parkinson's": {k: 0 for k in PARKINSONS_FEATURES}
}

# ------------------- FUNCTIONS -------------------
def predict_single(disease, input_data):
    if disease == "Kidney":
        X = pd.DataFrame([input_data])[KIDNEY_FEATURES]
        model = kidney_model
    elif disease == "Liver":
        X = pd.DataFrame([input_data])[LIVER_FEATURES]
        model = liver_model
    else:
        X = pd.DataFrame([input_data])[PARKINSONS_FEATURES]
        model = parkinsons_model

    pred = model.predict(X)[0]
    proba = float(model.predict_proba(X)[0][1])
    return pred, proba

def show_feature_importance(feature_dict, title):
    if not feature_dict:
        st.warning("Feature importance data not available.")
        return
    features = list(feature_dict.keys())
    importances = list(feature_dict.values())
    sorted_idx = np.argsort(importances)[::-1][:10]

    fig, ax = plt.subplots()
    ax.barh(np.array(features)[sorted_idx], np.array(importances)[sorted_idx])
    ax.set_title(title)
    ax.invert_yaxis()
    st.pyplot(fig)

# ------------------- SIDEBAR -------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Disease Prediction", "Model Performance"])

# ------------------- MAIN -------------------
st.title("Multi-Disease Prediction System")
st.markdown("### Predict **Kidney Disease**, **Liver Disease**, or **Parkinson's Disease** using individual inputs or batch CSV upload.")

if page == "Disease Prediction":
    disease = st.selectbox("Select Disease", ["Kidney", "Liver", "Parkinson's"])

    input_mode = st.radio("Select Input Mode", ["Manual Input", "Batch CSV Upload"])

    if input_mode == "Manual Input":
        st.subheader(f"Enter {disease} Details")
        user_input = {}
        if disease == "Kidney":
            for col in KIDNEY_FEATURES:
                user_input[col] = st.number_input(f"{col}", value=0.0)
        elif disease == "Liver":
            for col in LIVER_FEATURES:
                if col == "Gender":
                    user_input[col] = st.selectbox("Gender", ["Male", "Female"])
                    user_input[col] = 1 if user_input[col] == "Male" else 0
                else:
                    user_input[col] = st.number_input(f"{col}", value=0.0)
        else:
            for col in PARKINSONS_FEATURES:
                user_input[col] = st.number_input(f"{col}", value=0.0)

        if st.button("Predict"):
            pred, proba = predict_single(disease, user_input)
            label = "Detected" if pred == 1 else "Not Detected"
            risk = "High" if proba > 0.7 else "Medium" if proba > 0.4 else "Low"
            st.success(f"**Prediction:** {label}")
            st.info(f"**Probability:** {proba:.2f} | **Risk Level:** {risk}")

    else:
        uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])
        if uploaded_file:
            df_batch = pd.read_csv(uploaded_file)
            if disease == "Kidney":
                expected = KIDNEY_FEATURES
                model = kidney_model
            elif disease == "Liver":
                expected = LIVER_FEATURES
                model = liver_model
            else:
                expected = PARKINSONS_FEATURES
                model = parkinsons_model

            missing_cols = set(expected) - set(df_batch.columns)
            if missing_cols:
                st.error(f"Uploaded CSV is missing columns: {missing_cols}")
            else:
                preds = model.predict(df_batch[expected])
                probas = model.predict_proba(df_batch[expected])[:, 1]
                df_batch["Prediction"] = np.where(preds == 1, "Detected", "Not Detected")
                df_batch["Probability"] = probas
                st.dataframe(df_batch)
                csv = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

elif page == "Model Performance":
    st.header("Model Performance Summary")
    for disease, metric in metrics_data.items():
        st.subheader(disease)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metric['accuracy']:.3f}")
        col2.metric("Precision", f"{metric['precision']:.3f}")
        col3.metric("Recall", f"{metric['recall']:.3f}")
        col4.metric("F1-Score", f"{metric['f1_score']:.3f}")
        st.markdown("---")

    st.header("Feature Importance")
    for disease, feature_dict in feature_importance_data.items():
        show_feature_importance(feature_dict, f"{disease}: Top Features")
