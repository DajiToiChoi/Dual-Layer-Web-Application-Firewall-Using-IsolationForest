from pathlib import Path
import pandas as pd

def load_ecml_pkdd(raw_dir: Path) -> pd.DataFrame:
    """
    ECML/PKDD web attack dataset formats vary.
    This loader expects CSV(s) with at least:
      - a text/request column
      - a label column with attack type or 0/1
    Output: DataFrame(raw_text,label_type)
    """
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame(columns=["raw_text","label_type"])

    df = pd.concat([pd.read_csv(fp) for fp in csv_files], ignore_index=True)

    text_col = None
    for c in df.columns:
        if c.lower() in ("raw_text","text","request","payload","data"):
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[0]

    label_col = None
    for c in df.columns:
        if c.lower() in ("label_type","type","category","label","class"):
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[-1]

    out = pd.DataFrame()
    out["raw_text"] = df[text_col].astype(str)
    out["label_type"] = df[label_col].astype(str)
    return out
