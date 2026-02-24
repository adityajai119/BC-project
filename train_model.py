# ==============================
# BREAST CANCER CLASSIFICATION
# COMPLETE ML PROJECT
# ==============================

# 1️⃣ Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib


# 2️⃣ Load Dataset
df = pd.read_csv("data.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())


# 3️⃣ Data Cleaning
# Drop unnecessary columns if they exist
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

if "Unnamed: 32" in df.columns:
    df.drop("Unnamed: 32", axis=1, inplace=True)


# 4️⃣ Convert Target Variable
# M = Malignant (1)
# B = Benign (0)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

print("\nTarget Value Counts:")
print(df["diagnosis"].value_counts())


# 5️⃣ Define Features and Target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]


# 6️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7️⃣ Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 8️⃣ Train Logistic Regression Model
log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\n===== Logistic Regression Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_log))


# 9️⃣ Train Random Forest Model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n===== Random Forest Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


# 🔟 Save Best Model (Random Forest Recommended)
joblib.dump(rf_model, "breast_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model and Scaler saved successfully!")