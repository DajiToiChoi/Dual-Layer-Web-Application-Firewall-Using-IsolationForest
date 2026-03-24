from fastapi import FastAPI, Depends, Header, HTTPException, Form, Body
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import RequestLog
from .middleware import WAFMiddleware, get_waf
from .config import DEMO_API_KEYS

Base.metadata.create_all(bind=engine)

app = FastAPI(title="ADL-WAF (IsolationForest + SVM) Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(WAFMiddleware)

def verify_api_key(x_api_key: str = Header(default=None)):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key")
    if x_api_key not in DEMO_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

@app.get("/")
async def root():
    return {
        "message": "Gateway running",
        "endpoints": ["/public", "/secure-data", "/submit", "/admin/logs", "/test", "/test/inspect"]
    }

@app.get("/public")
async def public():
    return {"message": "Public endpoint"}

@app.get("/secure-data")
async def secure_data(api_key: str = Depends(verify_api_key)):
    return {"message": "Secure data", "api_key": api_key}

@app.post("/submit")
async def submit(data: dict):
    return {"received": data}

@app.get("/admin/logs")
def admin_logs(limit: int = 50, db: Session = Depends(get_db)):
    logs = db.query(RequestLog).order_by(RequestLog.id.desc()).limit(limit).all()
    return [
        {
            "id": x.id,
            "time": x.created_at,
            "ip": x.client_ip,
            "path": x.path,
            "l1_anomaly": x.l1_anomaly,
            "anomaly_score": x.anomaly_score,
            "l2_type": x.l2_type,
            "blocked": x.is_blocked,
            "reason": x.block_reason,
        }
        for x in logs
    ]


@app.get("/test", response_class=HTMLResponse)
async def test_page():
    """Simple web form to manually test payloads against ADL-WAF."""
    return """
    <html>
      <head>
        <title>ADL-WAF Test Console</title>
        <style>
          body { font-family: system-ui, sans-serif; margin: 2rem; background: #0f172a; color: #e5e7eb; }
          h1 { color: #38bdf8; }
          label { display:block; margin-top:1rem; font-weight:600; }
          textarea, input[type=text] {
            width:100%; padding:0.5rem; border-radius:0.375rem;
            border:1px solid #4b5563; background:#020617; color:#e5e7eb;
          }
          textarea { min-height:120px; resize:vertical; }
          button {
            margin-top:1rem; padding:0.6rem 1.2rem; border-radius:9999px;
            border:none; cursor:pointer; background:linear-gradient(to right,#22c55e,#16a34a);
            color:#0b1120; font-weight:700;
          }
          .result { margin-top:1.5rem; padding:1rem; border-radius:0.5rem; background:#020617; border:1px solid #1f2937;}
          .blocked { border-color:#f97316; }
          .allowed { border-color:#22c55e; }
          code { font-family: ui-mono, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
        </style>
      </head>
      <body>
        <h1>ADL-WAF Test Console</h1>
        <p>Nhập đường dẫn, query và payload để kiểm tra model phát hiện tấn công. Nếu bị chặn, một cảnh báo Telegram sẽ được gửi (nếu bật trong <code>.env</code>).</p>
        <form id="test-form">
          <label>HTTP method</label>
          <input type="text" name="method" value="POST" />
          <label>Path (VD: /submit)</label>
          <input type="text" name="path" value="/submit" />
          <label>Query string (tùy chọn, VD: q=test)</label>
          <input type="text" name="query" value="" />
          <label>Body / Payload</label>
          <textarea name="body" placeholder="username=admin&password=123 OR 1=1"></textarea>
          <button type="submit">Test request</button>
        </form>
        <div id="result" class="result" style="display:none;"></div>
        <script>
        const form = document.getElementById('test-form');
        const resultDiv = document.getElementById('result');
        form.addEventListener('submit', async (e) => {
          e.preventDefault();
          const formData = new FormData(form);
          const payload = {
            method: formData.get('method') || 'POST',
            path: formData.get('path') || '/submit',
            query: formData.get('query') || '',
            body: formData.get('body') || ''
          };
          resultDiv.style.display = 'block';
          resultDiv.className = 'result';
          resultDiv.textContent = 'Đang gửi...';
          try {
            const res = await fetch('/test/inspect', {
              method: 'POST',
              headers: {'Content-Type':'application/json'},
              body: JSON.stringify(payload)
            });
            const data = await res.json();
            const cls = data.blocked ? 'result blocked' : 'result allowed';
            resultDiv.className = cls;
            resultDiv.innerHTML = '<strong>Kết quả:</strong><br>' +
              'blocked = ' + data.blocked + '<br>' +
              'reason = ' + (data.reason || '') + '<br>' +
              'L1.is_anomaly = ' + data.l1.is_anomaly + ', anomaly_score = ' + data.l1.anomaly_score.toFixed(4) + '<br>' +
              'L2.type = ' + (data.l2_type || '') + '<br>' +
              '<pre style="margin-top:0.75rem;white-space:pre-wrap;font-size:12px;">' + JSON.stringify(data, null, 2) + '</pre>';
          } catch (err) {
            resultDiv.className = 'result blocked';
            resultDiv.textContent = 'Lỗi: ' + err;
          }
        });
        </script>
      </body>
    </html>
    """


@app.post("/test/inspect")
async def test_inspect(payload: dict = Body(...)):
    """
    API thuần để test trực tiếp ADL-WAF (không áp dụng middleware rate-limit / block).
    Có thể gọi bằng fetch JSON từ trang /test.
    """
    # Lấy dữ liệu từ JSON body (từ trang /test hoặc client khác)
    method = (payload.get("method") or "POST").upper()
    path = payload.get("path") or "/submit"
    query = payload.get("query") or ""
    body = payload.get("body") or ""

    waf = get_waf()
    from .adlwaf import ReqView  # local import to avoid circulars at import time
    rv = ReqView(
        method=method,
        path=path,
        headers={},
        body=body,
        query=query,
    )
    decision = waf.inspect(rv)
    # Nếu bị block -> cũng gửi Telegram alert giống middleware
    if decision.get("blocked"):
        from .alert import send_telegram_alert
        send_telegram_alert(f"[WAF TEST BLOCK] path={rv.path} reason={decision.get('reason')}")
    return {
        "blocked": bool(decision.get("blocked")),
        "reason": decision.get("reason"),
        "l1": decision.get("l1"),
        "l2_type": decision.get("l2_type"),
    }
