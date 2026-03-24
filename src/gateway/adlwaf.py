from dataclasses import dataclass
from typing import Dict, Any, Optional
import joblib
import numpy as np
from ..features.l1_features import extract_l1_ratios

@dataclass
class ReqView:
    method: str
    path: str
    headers: Dict[str, str]
    body: str
    query: str = ""

class ADLWAF:
    def __init__(self, l1_model_path: str, l2_model_path: str):
        self.l1 = joblib.load(l1_model_path)  # IsolationForest
        self.l2 = joblib.load(l2_model_path)  # TF-IDF + SVM (Pipeline)

    def _l1_vector(self, payload: str) -> np.ndarray:
        feats = extract_l1_ratios(payload)
        return np.array([[feats["alnum_ratio"], feats["badwords_ratio"],
                          feats["special_ratio"], feats["illegal_special_ratio"]]], dtype=float)

    def l1_predict(self, req: ReqView) -> Dict[str, Any]:
        payload = f"{req.method} {req.path}?{req.query}\n{req.headers}\n\n{req.body}"
        X = self._l1_vector(payload)

        # IsolationForest: predict returns 1 (inlier/normal) or -1 (outlier/anomaly)
        pred = int(self.l1.predict(X)[0])

        # decision_function: higher = more normal
        raw = float(self.l1.decision_function(X)[0])  # typically around [-0.5, 0.5]
        # Convert to anomaly_score in [0,1] (heuristic scale)
        # clamp range for stability
        raw_clamped = max(-0.5, min(0.5, raw))
        anomaly_score = float(1 - (raw_clamped + 0.5) / 1.0)  # raw=0.5 -> 0, raw=-0.5 -> 1

        return {
            "is_anomaly": pred == -1,
            "anomaly_score": anomaly_score,
            "raw_decision": raw,
        }

    def l2_predict_type(self, req: ReqView) -> str:
        raw_text = f"{req.method} {req.path}?{req.query}\n{req.headers}\n\n{req.body}"
        return str(self.l2.predict([raw_text])[0])

    def inspect(self, req: ReqView) -> Dict[str, Any]:
        l1 = self.l1_predict(req)
        if not l1["is_anomaly"]:
            return {"blocked": False, "reason": "L1 normal", "l1": l1, "l2_type": ""}

        # If anomaly, run L2
        l2_type = self.l2_predict_type(req)
        if l2_type != "Valid":
            return {"blocked": True, "reason": f"L2 threat={l2_type}", "l1": l1, "l2_type": l2_type}
        return {"blocked": False, "reason": "Benign anomaly (L2=Valid)", "l1": l1, "l2_type": l2_type}
