from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from ..features.l1_features import extract_l1_ratios

ROOT = Path(__file__).resolve().parents[2]
L1_CSV = ROOT / "data" / "processed" / "l1_anomaly.csv"
L2_CSV = ROOT / "data" / "processed" / "l2_threat.csv"
L1_MODEL = ROOT / "models" / "l1_iforest.joblib"
L2_MODEL = ROOT / "models" / "l2_svm_tfidf.joblib"

def l1_vector(payload: str) -> np.ndarray:
    f = extract_l1_ratios(payload)
    return np.array([[f["alnum_ratio"], f["badwords_ratio"], f["special_ratio"], f["illegal_special_ratio"]]], dtype=float)

def evaluate_l1_only():
    df = pd.read_csv(L1_CSV)
    model = joblib.load(L1_MODEL)

    y_true = df["label_anomaly"].astype(int).values
    y_pred = []
    for p in df["payload"].astype(str).tolist():
        pred = int(model.predict(l1_vector(p))[0])  # 1 normal, -1 anomaly
        y_pred.append(1 if pred == -1 else 0)
    y_pred = np.array(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred)
    return {
        "cm": cm,
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
    }

def evaluate_adl_with_l2():
    # For ADL evaluation we need anomaly labels and a threat-type classifier.
    # We'll simulate "is_attack" using L2 type: Valid -> benign; otherwise attack.
    model_l1 = joblib.load(L1_MODEL)
    model_l2 = joblib.load(L2_MODEL)

    # Build a mixed dataset for evaluation from L2 corpus if it contains Valid + attacks
    df = pd.read_csv(L2_CSV)
    df = df.dropna()
    if df["label_type"].nunique() < 2:
        raise SystemExit("L2 dataset must contain at least Valid and one attack type to evaluate ADL logic.")

    # ADL truth: attack if label_type != Valid
    y_true = (df["label_type"].astype(str) != "Valid").astype(int).values

    y_pred = []
    for raw in df["raw_text"].astype(str).tolist():
        # L1 runs on ratios derived from raw_text
        pred_l1 = int(model_l1.predict(l1_vector(raw))[0])
        is_anom = (pred_l1 == -1)
        if not is_anom:
            y_pred.append(0)  # allow
            continue
        # L2 decides
        t = str(model_l2.predict([raw])[0])
        y_pred.append(0 if t == "Valid" else 1)

    y_pred = np.array(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "cm": cm,
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "l2_report": classification_report(df["label_type"].astype(str), model_l2.predict(df["raw_text"].astype(str)), digits=4),
    }

def main():
    if not (L1_CSV.exists() and L1_MODEL.exists()):
        raise SystemExit("Run build_corpus and train_l1_iforest first.")
    if not (L2_CSV.exists() and L2_MODEL.exists()):
        raise SystemExit("Run build_corpus and train_l2_svm first.")

    l1 = evaluate_l1_only()
    print("=== L1 ONLY (IsolationForest) ===")
    print("Confusion matrix:\n", l1["cm"])
    print("Accuracy:", l1["acc"])
    print("Precision:", l1["prec"])
    print("Recall:", l1["rec"])

    adl = evaluate_adl_with_l2()
    print("\n=== ADL (L1 IsolationForest + L2 TFIDF+SVM) ===")
    print("Confusion matrix:\n", adl["cm"])
    print("Accuracy:", adl["acc"])
    print("Precision:", adl["prec"])
    print("Recall:", adl["rec"])

    print("\n=== L2 per-class report ===")
    print(adl["l2_report"])

if __name__ == "__main__":
    main()
