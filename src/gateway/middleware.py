from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from sqlalchemy.orm import Session

from .database import SessionLocal
from .models import RequestLog
from .rate_limiter import is_rate_limited
from .alert import send_telegram_alert
from .config import L1_MODEL_PATH, L2_MODEL_PATH
from .adlwaf import ADLWAF, ReqView

_waf = None

def get_waf() -> ADLWAF:
    global _waf
    if _waf is None:
        _waf = ADLWAF(str(L1_MODEL_PATH), str(L2_MODEL_PATH))
    return _waf

class WAFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query_string = request.url.query

        # rate limit
        if is_rate_limited(client_ip, path):
            return JSONResponse(status_code=429, content={"detail": "Too many requests (rate limited)"})

        # read body
        body_bytes = await request.body()
        body_str = body_bytes.decode(errors="ignore")

        # inspect via ADLWAF
        waf = get_waf()
        reqv = ReqView(method=method, path=path, headers=dict(request.headers), body=body_str, query=query_string)
        decision = waf.inspect(reqv)

        blocked = bool(decision["blocked"])
        reason = str(decision["reason"])
        l1 = decision["l1"]
        l2_type = decision.get("l2_type", "")

        # log
        db: Session = SessionLocal()
        try:
            log = RequestLog(
                client_ip=client_ip,
                method=method,
                path=path,
                query_string=query_string,
                headers=str(dict(request.headers)),
                body=body_str[:2000],
                rule_score=0,
                l1_anomaly=bool(l1["is_anomaly"]),
                anomaly_score=float(l1["anomaly_score"]),
                l2_type=l2_type,
                is_blocked=blocked,
                block_reason=reason,
            )
            db.add(log)
            db.commit()
        finally:
            db.close()

        if blocked:
            send_telegram_alert(f"[WAF BLOCK] ip={client_ip} path={path} reason={reason}")
            return JSONResponse(status_code=403, content={"detail": "Request blocked by ADL-WAF", "reason": reason})

        # restore body for downstream
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request._receive = receive

        return await call_next(request)
