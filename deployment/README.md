# Deployment Guide

This folder contains the deployed AutoML-based anomaly detection system, including backend API and frontend user interface.

---

## 📁 Structure

- `backend/` – FastAPI server for model inference  
- `frontend/` – Streamlit UI for user interaction  
- `artifacts/` – Trained model and preprocessing files  
- `sample_request.json` – Example input format  

---

## ⚙️ Requirements

- Python 3.10+
- Java (required for H2O MOJO scoring)

### Install dependencies:

```bash
pip install -r requirements.txt
```
---
## 🚀Running the Application 

### 1. Start Backend (FastAPI)

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Then open manually in browser:

http://127.0.0.1:8000/docs

### 2. Start Frontend (Streamlit)

Open a new terminal:

```bash
cd frontend
streamlit run streamlit_app.py
```
The UI will open automatically in your browser.

---

## 📊 Input Data (VERY IMPORTANT)

The model expects raw UNSW-NB15 network features (before preprocessing).

### ✅ Use:
Raw UNSW-NB15 dataset (test set recommended)
CSV format
All original feature columns
### ❌ Do NOT use:
Preprocessed data (scaled / encoded)
PCA / Autoencoder / VAE outputs
Selected feature subsets
NSL-KDD directly (unless adapted)
Files containing:
label
attack_cat
id

## 🧪 Example Workflow
Load a small CSV file (1–10 rows) from UNSW-NB15
Upload via Streamlit UI
System performs:
preprocessing
feature selection
model inference
Output:
prob_attack
pred_label (0 = normal, 1 = attack)

## ⚠️ Common Issues & Fixes
### 1. Missing Python packages

Error:
```bash
ModuleNotFoundError: No module named 'joblib'
```
Fix:
```bash
pip install -r requirements.txt
```
### 2. Backend not starting

If running:
```bash
python main.py
```
does nothing → use:
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
### 3. Image / UI loads but no predictions

Check:

Backend is running on port 8000
Frontend is pointing to correct URL:
http://127.0.0.1:8000

### 4. Java / MOJO error

Error related to:

h2o-genmodel.jar

Fix:

Ensure Java is installed
Ensure artifacts/ folder contains:
.zip model file
h2o-genmodel.jar

### 5. scikit-learn version warning

Warning:

InconsistentVersionWarning

Fix (optional but recommended):
```bash
pip install scikit-learn==1.7.2
```

### 6. Dataset not working

If predictions fail:

Check all required columns exist
Ensure categorical fields exist:
proto
service
state
Missing numeric values → will default to 0
Missing categoricals → will default to "unknown"

## 🧠 Notes
Backend uses H2O MOJO for fast inference
Frontend communicates via API
System is designed for UNSW-NB15 schema

## 💡 Tip

Start with 1 row first, then scale up.








 

