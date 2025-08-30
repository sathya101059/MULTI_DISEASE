# 🧠 Multi-Disease Prediction System

A **Streamlit-based web application** that predicts the likelihood of three major diseases:
- ✅ **Kidney Disease**
- ✅ **Liver Disease**
- ✅ **Parkinson's Disease**

The system supports:
- **Individual input** for predictions
- **Batch CSV upload** for multiple predictions
- **Feature Importance Visualization**
- **Model Performance Dashboard**
- **Download Predictions** (CSV)

---

## 🚀 Features
- **Machine Learning Models** (Random Forest Pipelines for all 3 diseases)
- **Real-time predictions**
- **Batch Prediction via CSV**
- **Interactive UI built with Streamlit**
- **Feature Importance Charts**
- **Performance Metrics (Accuracy, Precision, Recall, F1-score)**

---

## 🛠️ Tech Stack
- Python 3.x
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Matplotlib
- Joblib
- Git LFS (for model files)

---

## 📂 Project Structure
Multi-disease-predict/
│
├── app/
│ └── streamlit_app.py # Main Streamlit app
│
├── models/
│ ├── kidney_pipeline.pkl
│ ├── liver_pipeline.pkl
│ ├── parkinsons_pipeline.pkl
│ ├── kidney_feature_importances.json
│ ├── liver_feature_importances.json
│ ├── parkinsons_feature_importances.json
│ └── model_metrics.json
│
├── data/
│ ├── kidney_batch.csv
│ ├── liver_batch.csv
│ ├── parkinsons_batch.csv
│
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md