import pytest

import app.calibration as calibration_module


@pytest.fixture(autouse=True)
def isolated_calibration_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_module, "CALIBRATION_DIR", str(tmp_path / "calibrations"))
    yield
