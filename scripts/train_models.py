# scripts/train_models.py

from preprocess_kidney import preprocess_kidney_data
from preprocess_liver import preprocess_liver_data
from preprocess_parkinsons import preprocess_parkinsons_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ----------------- Kidney Disease -----------------
print("\nTraining Kidney Disease Model...")
X_train, X_test, y_train, y_test = preprocess_kidney_data("data/kidney_disease.csv")

kidney_model = RandomForestClassifier(n_estimators=100, random_state=42)
kidney_model.fit(X_train, y_train)
y_pred_kidney = kidney_model.predict(X_test)

print("Kidney Disease Report:\n", classification_report(y_test, y_pred_kidney))
print("Kidney Train Accuracy:", kidney_model.score(X_train, y_train))
print("Kidney Test Accuracy:", kidney_model.score(X_test, y_test))


joblib.dump(kidney_model, "models/kidney_model.pkl")
print("✅ Kidney Disease Model saved.\n")

# ----------------- Liver Disease -----------------
print("Training Liver Disease Model...")
X_train, X_test, y_train, y_test = preprocess_liver_data("data/indian_liver_patient.csv")

# Apply SMOTE to handle class imbalance
from imblearn.over_sampling import SMOTE
print("Before SMOTE:", y_train.value_counts())
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("After SMOTE:", y_train.value_counts())


liver_model = RandomForestClassifier(n_estimators=100, random_state=42)
liver_model.fit(X_train, y_train)
y_pred_liver = liver_model.predict(X_test)

print("Liver Disease Report:\n", classification_report(y_test, y_pred_liver))

joblib.dump(liver_model, "models/liver_model.pkl")
print("✅ Liver Disease Model saved.\n")

# ----------------- Parkinson's Disease -----------------
print("Training Parkinson's Disease Model...")
X_train, X_test, y_train, y_test = preprocess_parkinsons_data("data/parkinsons.csv")

parkinsons_model = RandomForestClassifier(n_estimators=100, random_state=42)
parkinsons_model.fit(X_train, y_train)
y_pred_parkinsons = parkinsons_model.predict(X_test)

print("Parkinson's Disease Report:\n", classification_report(y_test, y_pred_parkinsons))

joblib.dump(parkinsons_model, "models/parkinsons_model.pkl")
print("✅ Parkinson's Disease Model saved.\n")
