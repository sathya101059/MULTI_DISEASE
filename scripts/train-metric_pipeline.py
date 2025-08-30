import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Create models directory
os.makedirs("models", exist_ok=True)

def save_confusion_matrix(cm, class_names, out_path, title):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def build_pipeline():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, slice(0, 1000))], remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    return pipeline

all_metrics = {}

# ----------------- Kidney -----------------
print("Training Kidney Disease Model...")
df_kidney = pd.read_csv("data/kidney_disease.csv")
df_kidney.drop(columns=['id'], errors='ignore', inplace=True)
df_kidney.replace('?', np.nan, inplace=True)

# Encode categorical
for col in df_kidney.select_dtypes(include='object').columns:
    df_kidney[col] = df_kidney[col].astype('category').cat.codes
    df_kidney[col] = df_kidney[col].replace(-1, np.nan)

df_kidney = df_kidney.apply(pd.to_numeric, errors='coerce')

# Corrected Kidney Features based on your CSV
KIDNEY_FEATURES = [
    "age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc",
    "sod","pot","hemo","pcv","wc","rc","htn","dm","cad","appet","pe","ane"
]


X_kidney = df_kidney[KIDNEY_FEATURES]
y_kidney = df_kidney['classification'].replace({0:0, 1:1, 2:1})

X_train, X_test, y_train, y_test = train_test_split(X_kidney, y_kidney, test_size=0.2, random_state=42)
kidney_pipeline = build_pipeline()
kidney_pipeline.fit(X_train, y_train)
joblib.dump(kidney_pipeline, "models/kidney_pipeline.pkl")
print("✅ Kidney pipeline saved.")

y_pred = kidney_pipeline.predict(X_test)
all_metrics["Kidney Disease"] = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred))
}
save_confusion_matrix(confusion_matrix(y_test, y_pred), ["No Disease","Disease"], "models/kidney_confusion.png", "Kidney Confusion Matrix")

# ----------------- Liver -----------------
print("Training Liver Disease Model...")
df_liver = pd.read_csv("data/indian_liver_patient.csv")
df_liver.replace('?', np.nan, inplace=True)
for col in df_liver.select_dtypes(include='object').columns:
    df_liver[col] = df_liver[col].astype('category').cat.codes
    df_liver[col] = df_liver[col].replace(-1, np.nan)
df_liver = df_liver.apply(pd.to_numeric, errors='coerce')

LIVER_FEATURES = [
    "Age",
    "Gender",  # encode Male=1, Female=0
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio"
]

X_liver = df_liver[LIVER_FEATURES]
y_liver = df_liver['Dataset'].replace({1:1, 2:0})

X_train, X_test, y_train, y_test = train_test_split(X_liver, y_liver, test_size=0.2, random_state=42)
liver_pipeline = build_pipeline()
liver_pipeline.fit(X_train, y_train)
joblib.dump(liver_pipeline, "models/liver_pipeline.pkl")
print("✅ Liver pipeline saved.")

y_pred = liver_pipeline.predict(X_test)
all_metrics["Liver Disease"] = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred))
}
save_confusion_matrix(confusion_matrix(y_test, y_pred), ["No Disease","Disease"], "models/liver_confusion.png", "Liver Confusion Matrix")

# ----------------- Parkinson -----------------
print("Training Parkinson's Disease Model...")
df_pd = pd.read_csv("data/parkinsons.csv")
PARKINSONS_FEATURES = [
    'MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)',
    'MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)',
    'Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR',
    'RPDE','DFA','spread1','spread2','D2','PPE'
]
X_pd = df_pd[PARKINSONS_FEATURES]
y_pd = df_pd['status']

X_train, X_test, y_train, y_test = train_test_split(X_pd, y_pd, test_size=0.2, random_state=42)
pd_pipeline = build_pipeline()
pd_pipeline.fit(X_train, y_train)
joblib.dump(pd_pipeline, "models/parkinsons_pipeline.pkl")
print("✅ Parkinson pipeline saved.")

y_pred = pd_pipeline.predict(X_test)
all_metrics["Parkinson's Disease"] = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred))
}
save_confusion_matrix(confusion_matrix(y_test, y_pred), ["No PD","PD"], "models/parkinsons_confusion.png", "Parkinson's Confusion Matrix")

# ----------------- Save all metrics -----------------
import json
with open("models/model_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
print("✅ All metrics saved to models/model_metrics.json")
