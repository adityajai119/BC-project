import os, json
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_PATHS = {"rf": os.path.join(BASE_DIR, "breast_cancer_rf.pkl"), "gb": os.path.join(BASE_DIR, "breast_cancer_gb.pkl"), "lr": os.path.join(BASE_DIR, "breast_cancer_lr.pkl")}
MODEL_DISPLAY = {"rf": "Random Forest", "gb": "Gradient Boosting", "lr": "Logistic Regression"}

def load_artifacts():
    scaler = joblib.load(SCALER_PATH)
    models = {}
    for k, p in MODEL_PATHS.items():
        if os.path.exists(p): models[k] = joblib.load(p)
    if not models: raise RuntimeError("No models. Run train_model.py first.")
    df = pd.read_csv(DATA_PATH)
    for c in ["id", "Unnamed: 32"]:
        if c in df.columns: df = df.drop(c, axis=1)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    X = df.drop("diagnosis", axis=1); y = df["diagnosis"]
    return models, scaler, X, y, list(X.columns), X.mean().to_dict()

models, scaler, X_full, y_full, FNAMES, FMEANS = load_artifacts()
X_scaled = scaler.transform(X_full.values)

# Pre-compute PCA
pca_model = PCA(n_components=2)
pca_coords = pca_model.fit_transform(X_scaled)

# Pre-compute SHAP explainers (background sample for speed)
bg = shap.sample(pd.DataFrame(X_scaled, columns=FNAMES), 50)
shap_explainers = {}
for mk, m in models.items():
    try:
        if mk == "lr":
            shap_explainers[mk] = shap.LinearExplainer(m, bg)
        else:
            shap_explainers[mk] = shap.TreeExplainer(m)
    except Exception:
        shap_explainers[mk] = shap.KernelExplainer(m.predict_proba, bg)

app = Flask(__name__)
CORS(app)
DEFAULT = "rf"

def _label(p): return "Malignant" if p == 1 else "Benign"
def _arr(feats):
    missing = [f for f in FNAMES if f not in feats]
    if missing: raise ValueError(f"Missing: {', '.join(missing)}")
    return np.array([float(feats[n]) for n in FNAMES]).reshape(1, -1)

def _metrics(m, k):
    yp = m.predict(X_scaled)
    return {"modelKey": k, "modelName": MODEL_DISPLAY.get(k, ""), "accuracy": float(accuracy_score(y_full, yp)),
            "precision": float(precision_score(y_full, yp)), "recall": float(recall_score(y_full, yp)),
            "f1": float(f1_score(y_full, yp)), "confusionMatrix": confusion_matrix(y_full, yp).tolist(), "nSamples": len(y_full)}

# ── Existing Endpoints ──
@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify({"models": [{"key": k, "name": MODEL_DISPLAY.get(k, k)} for k in models], "default": DEFAULT})

@app.route("/api/feature-metadata", methods=["GET"])
def feature_metadata():
    feats = [{"name": n, "default": float(FMEANS.get(n, 0)), "type": "number"} for n in FNAMES]
    return jsonify({"features": feats, "targetName": "diagnosis"})

@app.route("/api/predict", methods=["POST"])
def predict_single():
    data = request.get_json(silent=True) or {}
    feats = data.get("features")
    if not isinstance(feats, dict): return jsonify({"error": "Need 'features' object."}), 400
    mk = data.get("model", DEFAULT).lower().strip()
    if mk not in models: return jsonify({"error": f"Unknown model '{mk}'."}), 400
    m = models[mk]
    try:
        arr = _arr(feats); scaled = scaler.transform(arr)
        pred = int(m.predict(scaled)[0]); proba = m.predict_proba(scaled)[0]
    except Exception as e: return jsonify({"error": str(e)}), 400
    # Confidence interval via bootstrap
    ci_low, ci_high = float(proba[1]), float(proba[1])
    try:
        preds_boot = []
        for _ in range(50):
            idx = np.random.choice(len(X_scaled), len(X_scaled), replace=True)
            from sklearn.base import clone
            mc = clone(m).fit(X_scaled[idx], y_full.iloc[idx])
            preds_boot.append(mc.predict_proba(scaled)[0][1])
        ci_low = float(np.percentile(preds_boot, 5))
        ci_high = float(np.percentile(preds_boot, 95))
    except Exception:
        pass
    # Basic contributions
    contributions = []
    if hasattr(m, "feature_importances_"):
        imps = m.feature_importances_
        devs = [(float(feats[n]) - FMEANS[n]) / (FMEANS[n] + 1e-9) for n in FNAMES]
        contribs = sorted([(FNAMES[i], float(imps[i] * devs[i])) for i in range(len(FNAMES))], key=lambda x: abs(x[1]), reverse=True)
        contributions = [{"feature": c[0], "contribution": round(c[1], 5)} for c in contribs[:10]]
    elif hasattr(m, "coef_"):
        coefs = m.coef_[0]
        contribs = sorted([(FNAMES[i], float(coefs[i] * scaled[0][i])) for i in range(len(FNAMES))], key=lambda x: abs(x[1]), reverse=True)
        contributions = [{"feature": c[0], "contribution": round(c[1], 5)} for c in contribs[:10]]
    return jsonify({"prediction": pred, "label": _label(pred), "probabilities": {"benign": float(proba[0]), "malignant": float(proba[1])},
                     "modelKey": mk, "modelName": MODEL_DISPLAY.get(mk, ""), "contributions": contributions,
                     "confidenceInterval": {"low": round(ci_low, 4), "high": round(ci_high, 4)}})

@app.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    mk = request.args.get("model", DEFAULT).lower().strip()
    if mk not in models: return jsonify({"error": f"Unknown '{mk}'."}), 400
    m = models[mk]
    if "file" not in request.files: return jsonify({"error": "No file."}), 400
    try: df = pd.read_csv(request.files["file"])
    except Exception as e: return jsonify({"error": str(e)}), 400
    missing = [f for f in FNAMES if f not in df.columns]
    if missing: return jsonify({"error": f"Missing: {', '.join(missing)}"}), 400
    scaled = scaler.transform(df[FNAMES].astype(float).values)
    preds = m.predict(scaled); probas = m.predict_proba(scaled)
    results = [{"index": i, "prediction": int(p), "label": _label(int(p)), "probabilities": {"benign": float(probas[i][0]), "malignant": float(probas[i][1])}} for i, p in enumerate(preds)]
    return jsonify({"results": results, "modelKey": mk, "modelName": MODEL_DISPLAY.get(mk, "")})

@app.route("/api/metrics", methods=["GET"])
def metrics():
    mk = request.args.get("model", DEFAULT).lower().strip()
    if mk not in models: return jsonify({"error": f"Unknown '{mk}'."}), 400
    return jsonify(_metrics(models[mk], mk))

@app.route("/api/compare-models", methods=["GET"])
def compare_models():
    return jsonify({"models": [_metrics(m, k) for k, m in models.items()]})

@app.route("/api/dataset-stats", methods=["GET"])
def dataset_stats():
    cc = y_full.value_counts()
    corr = X_full.corrwith(y_full).abs().sort_values(ascending=False)
    return jsonify({"classDistribution": {"benign": int(cc.get(0, 0)), "malignant": int(cc.get(1, 0))},
                     "totalSamples": len(y_full), "totalFeatures": len(FNAMES),
                     "topCorrelations": [{"feature": n, "correlation": round(float(v), 4)} for n, v in corr.head(10).items()]})

@app.route("/api/feature-importance", methods=["GET"])
def feature_importance():
    mk = request.args.get("model", DEFAULT).lower().strip()
    if mk not in models: return jsonify({"error": f"Unknown '{mk}'."}), 400
    m = models[mk]
    if not hasattr(m, "feature_importances_"): return jsonify({"error": "N/A for this model."}), 400
    imps = m.feature_importances_
    paired = sorted(zip(FNAMES, imps), key=lambda x: x[1], reverse=True)
    return jsonify({"importances": [{"feature": n, "importance": round(float(v), 5)} for n, v in paired], "modelKey": mk})

@app.route("/api/predict-all", methods=["POST"])
def predict_all():
    data = request.get_json(silent=True) or {}
    feats = data.get("features")
    if not isinstance(feats, dict): return jsonify({"error": "Need 'features' object."}), 400
    try: arr = _arr(feats); scaled = scaler.transform(arr)
    except Exception as e: return jsonify({"error": str(e)}), 400
    results = []
    for mk, m in models.items():
        pred = int(m.predict(scaled)[0]); proba = m.predict_proba(scaled)[0]
        results.append({"modelKey": mk, "modelName": MODEL_DISPLAY.get(mk, ""), "prediction": pred,
                         "label": _label(pred), "probabilities": {"benign": float(proba[0]), "malignant": float(proba[1])}})
    mal = sum(1 for r in results if r["prediction"] == 1)
    consensus = "Malignant" if mal > len(results) / 2 else "Benign"
    agreement = max(mal, len(results) - mal) / len(results) * 100
    return jsonify({"results": results, "consensus": consensus, "agreement": round(agreement, 1)})

@app.route("/api/cross-validation", methods=["GET"])
def cross_validation():
    cv = []
    for mk, m in models.items():
        scores = cross_val_score(m, X_scaled, y_full, cv=5, scoring="accuracy")
        cv.append({"modelKey": mk, "modelName": MODEL_DISPLAY.get(mk, ""), "scores": [round(float(s), 4) for s in scores],
                    "mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4)})
    return jsonify({"results": cv})

@app.route("/api/roc-data", methods=["GET"])
def roc_data():
    curves = []
    for mk, m in models.items():
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_full, proba); roc_auc = auc(fpr, tpr)
            step = max(1, len(fpr) // 50)
            curves.append({"modelKey": mk, "modelName": MODEL_DISPLAY.get(mk, ""),
                           "fpr": [round(float(v), 4) for v in fpr[::step]], "tpr": [round(float(v), 4) for v in tpr[::step]],
                           "auc": round(float(roc_auc), 4)})
    return jsonify({"curves": curves})

@app.route("/api/learning-curve", methods=["GET"])
def learning_curve_data():
    mk = request.args.get("model", DEFAULT).lower().strip()
    if mk not in models: return jsonify({"error": f"Unknown '{mk}'."}), 400
    sizes, ts, vs = learning_curve(models[mk], X_scaled, y_full, train_sizes=np.linspace(0.1, 1.0, 8), cv=5, scoring="accuracy", n_jobs=-1)
    points = [{"size": int(sizes[i]), "trainMean": round(float(ts[i].mean()), 4), "valMean": round(float(vs[i].mean()), 4)} for i in range(len(sizes))]
    return jsonify({"modelKey": mk, "modelName": MODEL_DISPLAY.get(mk, ""), "points": points})

@app.route("/api/correlation-matrix", methods=["GET"])
def correlation_matrix():
    top_n = min(int(request.args.get("n", 12)), 20)
    corr_diag = X_full.corrwith(y_full).abs().sort_values(ascending=False)
    top = list(corr_diag.head(top_n).index); corr = X_full[top].corr()
    matrix = [{"x": f1, "y": f2, "value": round(float(corr.loc[f1, f2]), 3)} for f1 in top for f2 in top]
    return jsonify({"features": top, "matrix": matrix})

@app.route("/api/scatter-data", methods=["GET"])
def scatter_data():
    f1 = request.args.get("f1", FNAMES[0]); f2 = request.args.get("f2", FNAMES[1])
    if f1 not in FNAMES or f2 not in FNAMES: return jsonify({"error": "Invalid features."}), 400
    points = [{"x": round(float(X_full.iloc[i][f1]), 4), "y": round(float(X_full.iloc[i][f2]), 4), "label": int(y_full.iloc[i])} for i in range(len(X_full))]
    return jsonify({"f1": f1, "f2": f2, "points": points})

# ── NEW Endpoints ──
@app.route("/api/shap-values", methods=["POST"])
def shap_values():
    data = request.get_json(silent=True) or {}
    feats = data.get("features")
    if not isinstance(feats, dict): return jsonify({"error": "Need 'features'."}), 400
    mk = data.get("model", DEFAULT).lower().strip()
    if mk not in shap_explainers: return jsonify({"error": f"No SHAP for '{mk}'."}), 400
    try:
        arr = _arr(feats); scaled = scaler.transform(arr)
        sv = shap_explainers[mk].shap_values(scaled)
        if isinstance(sv, list): sv = sv[1]  # class 1 (malignant)
        if sv.ndim > 1: sv = sv[0]
        paired = sorted(zip(FNAMES, sv.tolist()), key=lambda x: abs(x[1]), reverse=True)
        return jsonify({"shapValues": [{"feature": n, "value": round(float(v), 5)} for n, v in paired[:15]], "modelKey": mk})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/histogram-data", methods=["GET"])
def histogram_data():
    feat = request.args.get("feature", FNAMES[0])
    if feat not in FNAMES: return jsonify({"error": "Invalid feature."}), 400
    benign = X_full[y_full == 0][feat].values; malignant = X_full[y_full == 1][feat].values
    all_vals = np.concatenate([benign, malignant])
    bins = np.linspace(float(all_vals.min()), float(all_vals.max()), 25)
    b_hist, _ = np.histogram(benign, bins=bins); m_hist, _ = np.histogram(malignant, bins=bins)
    centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    return jsonify({"feature": feat, "bins": [{"center": round(float(c), 4), "benign": int(b_hist[i]), "malignant": int(m_hist[i])} for i, c in enumerate(centers)]})

@app.route("/api/pca-data", methods=["GET"])
def pca_data():
    points = [{"x": round(float(pca_coords[i, 0]), 4), "y": round(float(pca_coords[i, 1]), 4), "label": int(y_full.iloc[i])} for i in range(len(pca_coords))]
    return jsonify({"explained": [round(float(v), 4) for v in pca_model.explained_variance_ratio_], "points": points})

@app.route("/api/decision-boundary", methods=["POST"])
def decision_boundary():
    data = request.get_json(silent=True) or {}
    f1 = data.get("f1", FNAMES[0]); f2 = data.get("f2", FNAMES[1])
    mk = data.get("model", DEFAULT).lower().strip()
    if mk not in models or f1 not in FNAMES or f2 not in FNAMES: return jsonify({"error": "Invalid."}), 400
    i1, i2 = FNAMES.index(f1), FNAMES.index(f2)
    # Train a 2-feature model for visualization
    X2 = X_scaled[:, [i1, i2]]
    from sklearn.base import clone
    m2 = clone(models[mk]).fit(X2, y_full)
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    res = 40
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
    Z = m2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    grid = [{"x": round(float(xx[i, j]), 3), "y": round(float(yy[i, j]), 3), "p": round(float(Z[i, j]), 3)} for i in range(res) for j in range(res)]
    pts = [{"x": round(float(X2[i, 0]), 3), "y": round(float(X2[i, 1]), 3), "label": int(y_full.iloc[i])} for i in range(len(X2))]
    return jsonify({"grid": grid, "points": pts, "f1": f1, "f2": f2, "resolution": res})

@app.route("/api/retrain", methods=["POST"])
def retrain():
    data = request.get_json(silent=True) or {}
    mk = data.get("model", "rf").lower().strip()
    params = data.get("params", {})
    try:
        if mk == "rf":
            m = RandomForestClassifier(n_estimators=int(params.get("n_estimators", 100)), max_depth=params.get("max_depth"), min_samples_split=int(params.get("min_samples_split", 2)), random_state=42)
        elif mk == "gb":
            m = GradientBoostingClassifier(n_estimators=int(params.get("n_estimators", 100)), max_depth=int(params.get("max_depth", 3)), learning_rate=float(params.get("learning_rate", 0.1)), random_state=42)
        elif mk == "lr":
            m = LogisticRegression(C=float(params.get("C", 1.0)), max_iter=int(params.get("max_iter", 5000)), random_state=42)
        else:
            return jsonify({"error": "Unknown model."}), 400
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(X_scaled, y_full, test_size=0.2, random_state=42)
        m.fit(Xtr, ytr)
        yp_tr = m.predict(Xtr); yp_te = m.predict(Xte)
        return jsonify({"status": "ok", "modelKey": mk, "modelName": MODEL_DISPLAY.get(mk, ""),
                        "trainMetrics": {"accuracy": round(float(accuracy_score(ytr, yp_tr)), 4), "f1": round(float(f1_score(ytr, yp_tr)), 4)},
                        "testMetrics": {"accuracy": round(float(accuracy_score(yte, yp_te)), 4), "f1": round(float(f1_score(yte, yp_te)), 4), "precision": round(float(precision_score(yte, yp_te)), 4), "recall": round(float(recall_score(yte, yp_te)), 4)},
                        "confusionMatrix": confusion_matrix(yte, yp_te).tolist(), "params": params})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── POTTI Chatbot ──
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

POTTI_SYSTEM = """You are POTTI, a friendly and knowledgeable AI assistant for the Breast Cancer AI Studio.
You help users understand breast cancer diagnosis, the machine learning models used (Random Forest, Gradient Boosting, Logistic Regression),
and the features of the application (predictions, SHAP explainability, PCA, decision boundaries, model training).
Keep answers concise (2-4 sentences) unless asked for detail. Use a warm, encouraging tone.
If asked about medical advice, always clarify you're an AI assistant and recommend consulting a healthcare professional."""

@app.route("/api/chat", methods=["POST"])
def chat():
    if not GEMINI_KEY:
        return jsonify({"error": "GEMINI_API_KEY not set. Set it as an environment variable."}), 400
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    history = data.get("history", [])
    if not message:
        return jsonify({"error": "Empty message."}), 400
    try:
        model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=POTTI_SYSTEM)
        chat_history = [{"role": h["role"], "parts": [h["text"]]} for h in history]
        chat_session = model.start_chat(history=chat_history)
        response = chat_session.send_message(message)
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "modelsLoaded": list(models.keys()), "chatbotReady": bool(GEMINI_KEY)}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
