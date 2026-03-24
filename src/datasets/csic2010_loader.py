from pathlib import Path
import pandas as pd

def _load_from_txt(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for fp in raw_dir.glob("*.txt"):
        name = fp.name.lower()
        if "normal" in name or "valid" in name:
            label = 0
        elif any(k in name for k in ("anomal", "attack", "malicious")):
            label = 1
        else:
            continue

        for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append({"payload": line, "label_anomaly": label})
    return pd.DataFrame(rows)


def _load_from_csv(raw_dir: Path) -> pd.DataFrame:
    """
    Support the CSV version (e.g., csic_database.csv) where columns include
    Method, URL, classification plus optional headers/fields.
    """
    csv_files = list(raw_dir.rglob("*.csv"))
    if not csv_files:
        return pd.DataFrame(columns=["payload", "label_anomaly"])

    df = pd.concat([pd.read_csv(fp) for fp in csv_files], ignore_index=True)

    # pick payload source: prefer URL column if present
    payload_cols = [c for c in df.columns if c.lower() in ("url", "request", "payload", "path")]
    if payload_cols:
        payload_col = payload_cols[0]
    else:
        payload_col = df.columns[-1]

    method_col = next((c for c in df.columns if c.lower() == "method"), None)

    def make_payload(row):
        url = str(row[payload_col])
        if method_col:
            return f"{row[method_col]} {url}"
        return url

    out = pd.DataFrame()
    out["payload"] = df.apply(make_payload, axis=1).astype(str)

    label_col = next((c for c in df.columns if c.lower() in ("classification", "label", "class", "attack")), None)
    if label_col:
        out["label_anomaly"] = df[label_col].apply(lambda v: 0 if str(v).lower() in ("0", "normal", "valid") else 1)
    else:
        out["label_anomaly"] = 0

    return out


def load_csic2010(raw_dir: Path) -> pd.DataFrame:
    """
    Flexible loader for CSIC2010 variants (txt or csv).
    Output: DataFrame(payload, label_anomaly)
    """
    from_txt = _load_from_txt(raw_dir)
    from_csv = _load_from_csv(raw_dir)
    if not from_txt.empty and not from_csv.empty:
        return pd.concat([from_txt, from_csv], ignore_index=True)
    if not from_txt.empty:
        return from_txt
    return from_csv
