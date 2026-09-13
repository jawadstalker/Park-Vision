import json
import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./park_vision.db")

_is_sqlite = DATABASE_URL.startswith("sqlite")
_engine_kwargs = {}
if _is_sqlite:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # Pooling only matters for a real server (Postgres); SQLite is a single
    # file and doesn't use a connection pool. pool_pre_ping avoids handing
    # out dead connections after a DB restart/failover.
    _engine_kwargs["pool_pre_ping"] = True
    _engine_kwargs["pool_size"] = int(os.environ.get("DB_POOL_SIZE", "10"))
    _engine_kwargs["max_overflow"] = int(os.environ.get("DB_MAX_OVERFLOW", "20"))

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class DetectionRecord(Base):
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True, nullable=False)
    mode = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    frame_width = Column(Integer)
    frame_height = Column(Integer)
    vehicle_count = Column(Integer)
    occupied_count = Column(Integer)
    empty_count = Column(Integer)
    processing_time_ms = Column(Float)
    vehicles_json = Column(Text)
    spots_json = Column(Text)


class CalibrationRecord(Base):
    __tablename__ = "calibrations"

    camera_id = Column(String, primary_key=True)
    matrix_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


def init_db():
    Base.metadata.create_all(bind=engine)


def save_detection(camera_id, mode, frame_width, frame_height, vehicles, spots, processing_time_ms):
    session = SessionLocal()
    try:
        occupied = sum(1 for spot in spots if spot["status"] == "occupied")
        empty = sum(1 for spot in spots if spot["status"] == "empty")
        record = DetectionRecord(
            camera_id=camera_id,
            mode=mode,
            frame_width=frame_width,
            frame_height=frame_height,
            vehicle_count=len(vehicles),
            occupied_count=occupied,
            empty_count=empty,
            processing_time_ms=processing_time_ms,
            vehicles_json=json.dumps(vehicles),
            spots_json=json.dumps(spots),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id
    finally:
        session.close()


def get_history(camera_id, limit=50):
    session = SessionLocal()
    try:
        records = (
            session.query(DetectionRecord)
            .filter(DetectionRecord.camera_id == camera_id)
            .order_by(DetectionRecord.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": record.id,
                "camera_id": record.camera_id,
                "mode": record.mode,
                "timestamp": record.timestamp.isoformat(),
                "frame_width": record.frame_width,
                "frame_height": record.frame_height,
                "vehicle_count": record.vehicle_count,
                "occupied_count": record.occupied_count,
                "empty_count": record.empty_count,
                "processing_time_ms": record.processing_time_ms,
                "vehicles": json.loads(record.vehicles_json),
                "spots": json.loads(record.spots_json),
            }
            for record in records
        ]
    finally:
        session.close()


def list_cameras_with_history():
    session = SessionLocal()
    try:
        rows = session.query(DetectionRecord.camera_id).distinct().all()
        return sorted(row[0] for row in rows)
    finally:
        session.close()


def save_calibration_record(camera_id, matrix_json):
    session = SessionLocal()
    try:
        record = session.get(CalibrationRecord, camera_id)
        if record is None:
            record = CalibrationRecord(camera_id=camera_id, matrix_json=matrix_json)
            session.add(record)
        else:
            record.matrix_json = matrix_json
        session.commit()
    finally:
        session.close()


def load_calibration_record(camera_id):
    session = SessionLocal()
    try:
        record = session.get(CalibrationRecord, camera_id)
        return record.matrix_json if record else None
    finally:
        session.close()


def list_calibration_camera_ids():
    session = SessionLocal()
    try:
        rows = session.query(CalibrationRecord.camera_id).all()
        return sorted(row[0] for row in rows)
    finally:
        session.close()
