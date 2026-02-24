# ==============================
# BREAST CANCER CLASSIFICATION
# MULTI-MODEL ML PROJECT
# ==============================

# 1️⃣ Import Libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib


# 2️⃣ Load Dataset
df = pd.read_csv("data.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())


# 3️⃣ Data Cleaning
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

if "Unnamed: 32" in df.columns:
    df.drop("Unnamed: 32", axis=1, inplace=True)


# 4️⃣ Convert Target Variable
# M = Malignant (1), B = Benign (0)
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


# 8️⃣ Train Logistic Regression
print("\n===== Logistic Regression =====")
lr_model = LogisticRegression(max_iter=5000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))


# 9️⃣ Train Random Forest
print("\n===== Random Forest =====")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


# 🔟 Train Gradient Boosting
print("\n===== Gradient Boosting =====")
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_gb))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))


# 1️⃣1️⃣ Save All Models and Scaler
joblib.dump(rf_model, "breast_cancer_rf.pkl")
joblib.dump(gb_model, "breast_cancer_gb.pkl")
joblib.dump(lr_model, "breast_cancer_lr.pkl")
joblib.dump(scaler, "scaler.pkl")

# Also save as breast_cancer_model.pkl for backward compatibility
joblib.dump(rf_model, "breast_cancer_model.pkl")

print("\n✅ All 3 models and scaler saved successfully!")
print("   → breast_cancer_rf.pkl  (Random Forest)")
print("   → breast_cancer_gb.pkl  (Gradient Boosting)")
print("   → breast_cancer_lr.pkl  (Logistic Regression)")
print("   → scaler.pkl")