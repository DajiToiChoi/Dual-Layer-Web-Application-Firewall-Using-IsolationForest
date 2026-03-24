import time
from collections import defaultdict
from .config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS

_request_history = defaultdict(list)

def is_rate_limited(client_ip: str, route: str) -> bool:
    key = (client_ip, route)
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    ts = _request_history[key]
    ts = [t for t in ts if t >= window_start]
    ts.append(now)
    _request_history[key] = ts

    return len(ts) > RATE_LIMIT_REQUESTS
