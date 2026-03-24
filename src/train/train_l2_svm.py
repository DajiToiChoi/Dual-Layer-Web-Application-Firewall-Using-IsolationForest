from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "data" / "processed" / "l2_threat.csv"
OUT_MODEL = ROOT / "models" / "l2_svm_tfidf.joblib"

def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Missing {IN_CSV}. Run: python -m src.datasets.build_corpus")

    df = pd.read_csv(IN_CSV)
    if not {"raw_text","label_type"}.issubset(df.columns):
        raise SystemExit("l2_threat.csv must have raw_text,label_type columns.")

    X = df["raw_text"].astype(str)
    y = df["label_type"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,4))),
        ("svm", SVC(kernel="rbf", C=10))
    ])

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print("L2 TF-IDF+SVM evaluation:")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, digits=4))

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT_MODEL)
    print("Saved:", OUT_MODEL)

if __name__ == "__main__":
    main()
