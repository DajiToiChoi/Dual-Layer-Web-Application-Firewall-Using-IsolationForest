from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float
from datetime import datetime
from .database import Base

class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    client_ip = Column(String, index=True)
    method = Column(String)
    path = Column(String)
    query_string = Column(String)
    headers = Column(String)
    body = Column(String)

    # Scores
    rule_score = Column(Integer, default=0)          # optional legacy
    l1_anomaly = Column(Boolean, default=False)      # L1 output
    anomaly_score = Column(Float, default=0.0)       # continuous

    # L2 prediction
    l2_type = Column(String, default="")
    is_blocked = Column(Boolean, default=False)
    block_reason = Column(String, default="")

    created_at = Column(DateTime, default=datetime.utcnow)
