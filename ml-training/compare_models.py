import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("\n🔬 Scientific Model Comparison Started...\n")

# ----------------------------------------------------------
# STEP 1: Load dataset
# ----------------------------------------------------------
df = pd.read_csv("engine_data2.csv")

FEATURES = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp"
]

TARGET = "Engine Condition"

X_raw = df[FEATURES]
y = df[TARGET]

# ----------------------------------------------------------
# STEP 2: Load scaler used during training
# ----------------------------------------------------------
scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))

# IMPORTANT: apply the SAME scaler (do NOT refit)
X = scaler.transform(X_raw)

# ----------------------------------------------------------
# STEP 3: Train/Test split (same protocol as training)
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------------
# STEP 4: Load trained models
# ----------------------------------------------------------
# Random Forest ONNX was used for production,
# but RF pickle is inside the ensemble only.
# So we re-load RF from the ensemble if needed.

ensemble = pickle.load(open("artifacts/model.pkl", "rb"))
rf = dict(ensemble.named_estimators_)["rf"]

xgb = pickle.load(open("artifacts/xgb_model.pkl", "rb"))

# ----------------------------------------------------------
# STEP 5: Predict probabilities
# ----------------------------------------------------------
rf_probs = rf.predict_proba(X_test)[:, 1]
xgb_probs = xgb.predict_proba(X_test)[:, 1]

# Binary predictions using 0.5 threshold
rf_preds = (rf_probs >= 0.5).astype(int)
xgb_preds = (xgb_probs >= 0.5).astype(int)

# ----------------------------------------------------------
# STEP 6: Evaluation function
# ----------------------------------------------------------
def evaluate(name, probs, preds):
    print(f"\n🔍 {name}")
    print("-" * 35)
    print("ROC-AUC   :", round(roc_auc_score(y_test, probs), 4))
    print("Precision :", round(precision_score(y_test, preds), 4))
    print("Recall    :", round(recall_score(y_test, preds), 4))
    print("F1-score  :", round(f1_score(y_test, preds), 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

# ----------------------------------------------------------
# STEP 7: Compare models
# ----------------------------------------------------------
evaluate("Random Forest", rf_probs, rf_preds)
evaluate("XGBoost", xgb_probs, xgb_preds)

print("\n✅ Scientific comparison completed successfully.\n")
