import os
import requests
from dotenv import load_dotenv

load_dotenv()

def send_telegram_alert(message: str) -> None:
    if os.getenv("ENABLE_TELEGRAM_ALERT", "false").lower() != "true":
        return
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=data, timeout=5)
    except Exception:
        # Alerts must not break primary request processing
        pass
