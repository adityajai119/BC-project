import os
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "breast_cancer_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data.csv")


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError("Model or scaler file not found. Please run train_model.py first.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if not os.path.exists(DATA_PATH):
        raise RuntimeError("data.csv not found. It is required for feature metadata and metrics.")

    df = pd.read_csv(DATA_PATH)

    # Mirror the cleaning logic from train_model.py
    if "id" in df.columns:
        df = df.drop("id", axis=1)
    if "Unnamed: 32" in df.columns:
        df = df.drop("Unnamed: 32", axis=1)

    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    feature_names = list(X.columns)
    feature_means = X.mean().to_dict()

    return model, scaler, X, y, feature_names, feature_means


model, scaler, X_full, y_full, FEATURE_NAMES, FEATURE_MEANS = load_artifacts()

app = Flask(__name__)
CORS(app)


def _build_feature_array(payload_features: Dict[str, Any]) -> np.ndarray:
    missing = [f for f in FEATURE_NAMES if f not in payload_features]
    if missing:
        raise ValueError(f"Missing feature(s): {', '.join(missing)}")

    ordered = [float(payload_features[name]) for name in FEATURE_NAMES]
    return np.array(ordered, dtype=float).reshape(1, -1)


def _label_from_prediction(pred: int) -> str:
    # In train_model.py: M -> 1 (Malignant), B -> 0 (Benign)
    return "Malignant" if pred == 1 else "Benign"


@app.route("/api/feature-metadata", methods=["GET"])
def feature_metadata():
    features = [
        {
            "name": name,
            "default": float(FEATURE_MEANS.get(name, 0.0)),
            "type": "number",
        }
        for name in FEATURE_NAMES
    ]
    return jsonify(
        {
            "features": features,
            "targetName": "diagnosis",
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict_single():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or empty JSON body."}), 400

    features = data.get("features")
    if not isinstance(features, dict):
        return jsonify({"error": "Request body must contain a 'features' object."}), 400

    try:
        feature_array = _build_feature_array(features)
        scaled = scaler.transform(feature_array)
        pred = int(model.predict(scaled)[0])
        proba = model.predict_proba(scaled)[0]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - generic safety net
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    label = _label_from_prediction(pred)
    response = {
        "prediction": pred,
        "label": label,
        "probabilities": {
            "benign": float(proba[0]),
            "malignant": float(proba[1]),
        },
        "modelName": type(model).__name__,
    }
    return jsonify(response)


@app.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    """
    Accepts either:
    - multipart/form-data with a CSV file under key 'file', or
    - application/json with {"rows": [{feature: value, ...}, ...]}
    """
    if "file" in request.files:
        file = request.files["file"]
        try:
            df = pd.read_csv(file)
        except Exception as exc:
            return jsonify({"error": f"Could not read CSV: {exc}"}), 400
    else:
        data = request.get_json(silent=True) or {}
        rows: List[Dict[str, Any]] = data.get("rows") or []
        if not rows:
            return jsonify({"error": "No data provided. Upload a CSV file or provide a 'rows' array."}), 400
        df = pd.DataFrame(rows)

    missing_cols = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_cols:
        return jsonify({"error": f"Missing required feature columns: {', '.join(missing_cols)}"}), 400

    ordered_df = df[FEATURE_NAMES].astype(float)
    scaled = scaler.transform(ordered_df.values)
    preds = model.predict(scaled)
    probas = model.predict_proba(scaled)

    results = []
    for i, pred in enumerate(preds):
        label = _label_from_prediction(int(pred))
        results.append(
            {
                "index": int(i),
                "prediction": int(pred),
                "label": label,
                "probabilities": {
                    "benign": float(probas[i][0]),
                    "malignant": float(probas[i][1]),
                },
            }
        )

    return jsonify({"results": results})


@app.route("/api/metrics", methods=["GET"])
def metrics():
    """
    Evaluate the loaded model on the full dataset using the loaded scaler.
    """
    X_scaled = scaler.transform(X_full.values)
    y_pred = model.predict(X_scaled)

    acc = accuracy_score(y_full, y_pred)
    prec = precision_score(y_full, y_pred)
    rec = recall_score(y_full, y_pred)
    f1 = f1_score(y_full, y_pred)
    cm = confusion_matrix(y_full, y_pred).tolist()

    return jsonify(
        {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusionMatrix": cm,
            "modelName": type(model).__name__,
            "nSamples": int(len(y_full)),
        }
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

