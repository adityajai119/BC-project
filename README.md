# 🩺 Breast Cancer AI Studio

![Breast Cancer AI Studio](https://img.shields.io/badge/Status-Active-brightgreen)
![React](https://img.shields.io/badge/Frontend-React%2BVite-%2361DAFB?logo=react)
![Flask](https://img.shields.io/badge/Backend-Flask-%23000?logo=flask)
![Gemini](https://img.shields.io/badge/AI-Gemini_2.5_Flash-%234285F4?logo=google)

A full-stack, state-of-the-art web application that leverages multiple Machine Learning models (Random Forest, Gradient Boosting, Logistic Regression) to assist in breast cancer diagnosis. The application features advanced predictive analytics, model explainability (SHAP), dynamic visualizations (Histograms, PCA, Decision Boundaries), a custom model training dashboard, and an integrated AI chatbot named **POTTI** powered by Google's Gemini 2.5 API.

---

## ✨ Key Features

### 🧠 Advanced Analytics & ML
- **Multi-Model Inference:** Select between Random Forest, Gradient Boosting, and Logistic Regression for tailored predictions.
- **SHAP Explainers:** Interactive waterfall charts that explain exactly *why* a model made its prediction, breaking down the contribution of each individual feature.
- **Data Explorer:** Visualize the dataset with real-time Feature Scatter Plots, Correlation Heatmaps, Class Distribution Histograms, and 2D Principal Component Analysis (PCA).
- **Decision Boundaries:** View dynamic, generated 2D grids illustrating where models shift their predictions from Benign to Malignant.
- **What-If Analysis:** Interactively adjust patient features with sliders and instantly see the real-time impact on the diagnosis probabilities.
- **Confidence Intervals:** 90% bootstrap confidence intervals generated alongside deterministic predictions to gauge model certainty.

### 🏋️ Model Training & Evaluation
- **Hyperparameter Tuning:** Retrain the Random Forest, Gradient Boosting, or Logistic Regression models in real-time by adjusting sliders for `Trees`, `Max Depth`, `Min Samples Split`, `Learning Rate`, and `C`.
- **Metrics Dashboard:** View resulting validation metrics including Accuracy, Precision, Recall, F1-Score, and a customized Confusion Matrix Heatmap.

### 👩‍⚕️ Clinical Tools
- **Single & Batch Patient Processing:** Upload CSVs for bulk testing or query individual patient data.
- **Model Consensus:** Side-by-side comparison of how all 3 models vote on a single set of inputs, complete with an "Agreement Score."
- **PDF Report Generation:** One-click download of a comprehensive, formatted `Patient_Report.pdf` detailing the prediction, feature values, and SHAP top contributors.

### 🤖 POTTI AI Chatbot
- A friendly, persistent floating assistant integrated with **Gemini 2.5 Flash**.
- Domain-aware: Knows about breast cancer diagnosis, the application’s models, SHAP, PCA, and platform features.
- Fluid UI: Includes typing indicators, user/AI message distinct styling, and smooth `framer-motion` transitions.

### 🎨 Premium UI/UX
- **Interactive Onboarding:** `react-joyride` guided tour highlighting key platform elements on initial load.
- **Glassmorphism Design:** Sleek translucent cards, gradient typography, and hover animations.
- **System Theming:** Native-feeling Dark and Light mode toggle with extensive CSS polish.
- **Notifications:** Informative toasts indicating background processes (e.g., "Model trained successfully").
- **Skeleton Loaders:** Prevents layout shift while API requests are resolving.

---

## 🛠️ Tech Stack

**Frontend:**
- React 19 + TypeScript
- Vite
- Material-UI (MUI components and system themes)
- Recharts (Interactive SVG visualizations)
- Framer Motion (Page and component transitions)
- React Joyride (Onboarding tour)
- jsPDF + AutoTable (PDF document generation)

**Backend:**
- Python 3 + Flask
- Scikit-Learn (Model definitions and evaluation)
- Pandas & NumPy (Data manipulation)
- SHAP (Model explainers)
- Google Generative AI SDK (Gemini API for POTTI chatbot)

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-ai-studio.git
cd breast-cancer-ai-studio
```

### 2. Backend Setup (Flask API)
Open a terminal and navigate to the backend directory:
```bash
cd backend
```

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configure Gemini AI API (POTTI Chatbot)
You need a Google Gemini API Key for the chatbot to work.
1. Go to [Google AI Studio](https://aistudio.google.com/) and grab your API key.
2. Inside the `backend` folder, you will find a `.env` file (or create one). Add your key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Start the Backend Server
Make sure you are still in the `backend` folder and your virtual environment is activated:
```bash
python app.py
```
*The Flask server will start on `http://localhost:5000` and pre-compute PCA components and SHAP explainers.*

### 5. Frontend Setup (React App)
Open a **new** terminal window and navigate to the frontend directory:
```bash
cd frontend
```

Install Node.js dependencies:
```bash
npm install
# Or if using legacy deps for React 19:
npm install --legacy-peer-deps
```

Start the Vite development server:
```bash
npm run dev
```
*The frontend will be available at `http://localhost:5173`. Open this URL in your browser.*

---

## 📁 Project Structure

```text
breast-cancer-ai-studio/
├── backend/
│   ├── app.py                   # Main Flask server & all API routes
│   ├── data.csv                 # Original Breast Cancer dataset
│   ├── train_model.py           # Helper script to initially generate .pkl files
│   ├── scaler.pkl               # Trained feature scaler
│   ├── breast_cancer_rf.pkl     # Pre-trained Random Forest model
│   ├── breast_cancer_gb.pkl     # Pre-trained Gradient Boosting model
│   ├── breast_cancer_lr.pkl     # Pre-trained Logistic Regression model
│   ├── requirements.txt         # Python dependencies
│   └── .env                     # Contains GEMINI_API_KEY
├── frontend/
│   ├── index.html               # Entry point
│   ├── package.json             
│   ├── vite.config.ts           
│   └── src/
│       ├── App.tsx              # Main routing, layout, and global providers
│       ├── App.css              # Global glassmorphism & light mode styles
│       ├── shared.tsx           # Global types, themes, API urls, navigation
│       ├── Chatbot.tsx          # POTTI Gemini floating widget component
│       ├── OverviewPage.tsx     # Dashboard with dataset and model statistics
│       ├── SinglePage.tsx       # Predict single patient + SHAP + PDF Report
│       ├── BatchPage.tsx        # Upload CSV for bulk predictions
│       ├── ComparePage.tsx      # Multi-model consensus voting UI
│       ├── ExplorerPage.tsx     # Histograms, PCA, Correlation, Decision Boundaries
│       ├── InsightsPage.tsx     # Top feature importances
│       ├── WhatIfPage.tsx       # Real-time feature adjustment sliders
│       └── TrainingPage.tsx     # In-memory hyperparameter retuning dashboard
└── README.md
```

---

## 🤝 Contributing
Contributions are always welcome. To contribute:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📝 License
Distributed under the MIT License.
