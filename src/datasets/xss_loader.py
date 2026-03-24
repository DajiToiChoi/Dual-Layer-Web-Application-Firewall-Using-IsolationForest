from pathlib import Path
import pandas as pd

def load_xss(raw_dir: Path) -> pd.DataFrame:
    """
    XSS datasets are often CSV with payload and label.
    Output: DataFrame(raw_text,label_type) with label_type='XSS' for attacks, 'Valid' for benign if present.
    """
    csv_files = list(raw_dir.rglob("*.csv"))
    if not csv_files:
        return pd.DataFrame(columns=["raw_text","label_type"])

    df = pd.concat([pd.read_csv(fp) for fp in csv_files], ignore_index=True)

    payload_col = next((c for c in df.columns if c.lower() in ("payload","text","data","request","raw_text","sentence")), None) or df.columns[0]

    label_col = None
    for c in df.columns:
        if c.lower() in ("label","class","y","is_xss","attack"):
            label_col = c
            break

    out = pd.DataFrame()
    out["raw_text"] = df[payload_col].astype(str)
    if label_col is None:
        out["label_type"] = "XSS"
    else:
        out["label_type"] = df[label_col].apply(lambda v: "Valid" if str(v).lower() in ("0","benign","normal","valid") else "XSS")
    return out
