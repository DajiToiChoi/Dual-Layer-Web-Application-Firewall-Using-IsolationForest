from pathlib import Path
import pandas as pd

def load_httpparams(raw_dir: Path) -> pd.DataFrame:
    """
    HTTPParams dataset (commonly on GitHub) may be in CSV format with columns such as:
      - 'payload' or 'text' or 'request'
      - 'label' or 'class' (0/1) or strings
    This loader tries to detect columns heuristically.
    Output: DataFrame(payload, label_anomaly, label_type_optional)
    """
    csv_files = [fp for fp in raw_dir.rglob("*.csv") if fp.is_file()]
    if not csv_files:
        return pd.DataFrame(columns=["payload","label_anomaly","label_type"])

    df = pd.concat([pd.read_csv(fp) for fp in csv_files], ignore_index=True)

    # find payload column
    payload_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ("payload","text","request","query","http_request","data"):
            payload_col = c
            break
    if payload_col is None:
        payload_col = df.columns[0]

    # find label column
    label_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ("label","class","y","is_attack","attack"):
            label_col = c
            break

    # label_type column (optional)
    type_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ("type","attack_type","category","label_type"):
            type_col = c
            break

    out = pd.DataFrame()
    out["payload"] = df[payload_col].astype(str)

    if label_col is None:
        # if no label, assume unknown -> treat as normal (for L1 training you should provide labels)
        out["label_anomaly"] = 0
    else:
        # normalize label into 0/1
        out["label_anomaly"] = df[label_col].apply(lambda v: 0 if str(v).lower() in ("0","normal","valid","benign","norm") else 1)

    if type_col:
        out["label_type"] = df[type_col].astype(str)
    else:
        # if we have attack_type column, reuse it as label_type for L2
        atk_col = next((c for c in df.columns if c.lower() in ("attack_type","category","type")), None)
        if atk_col:
            out["label_type"] = df[atk_col].astype(str).apply(lambda v: "Valid" if str(v).lower() in ("norm","normal","valid","benign","0") else str(v))
        else:
            out["label_type"] = ""
    return out
