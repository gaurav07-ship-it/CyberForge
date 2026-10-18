"""
Database Persistence Layer
Stores transactions, alerts, grouped cases, and analyst SAR audit decisions.
"""

import os
import json
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_PATH = "data/aml_cases.db"
os.makedirs("data", exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base = declarative_base()


class DBCase(Base):
    __tablename__ = "cases"

    case_id = Column(String(50), primary_key=True)
    status = Column(String(30), default="NEW")
    risk_score = Column(Float, nullable=False)
    total_volume_usd = Column(Float, default=0.0)
    transaction_count = Column(Integer, default=1)
    topology = Column(String(50), default="UNKNOWN")
    primary_narrative = Column(Text, default="")
    accounts_json = Column(Text, default="[]")
    banks_json = Column(Text, default="[]")
    analyst_decision = Column(String(50), nullable=True)
    analyst_notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    alerts = relationship("DBAlert", back_populates="case", cascade="all, delete-orphan")


class DBAlert(Base):
    __tablename__ = "alerts"

    alert_id = Column(String(50), primary_key=True)
    case_id = Column(String(50), ForeignKey("cases.case_id"))
    timestamp = Column(String(50))
    src_account = Column(String(50))
    dst_account = Column(String(50))
    amount_usd = Column(Float)
    payment_format = Column(String(30))
    risk_score = Column(Float)
    risk_tier = Column(String(20))
    narrative = Column(Text)
    drivers_json = Column(Text, default="[]")
    subgraph_json = Column(Text, default="{}")

    case = relationship("DBCase", back_populates="alerts")


Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
