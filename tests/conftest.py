import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.calibration as calibration_module
import app.database as database_module


@pytest.fixture(autouse=True)
def isolated_calibration_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_module, "CALIBRATION_DIR", str(tmp_path / "calibrations"))
    yield


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    test_engine = create_engine(f"sqlite:///{tmp_path}/test_park_vision.db")
    database_module.Base.metadata.create_all(bind=test_engine)
    test_session_local = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(database_module, "engine", test_engine)
    monkeypatch.setattr(database_module, "SessionLocal", test_session_local)
    yield
