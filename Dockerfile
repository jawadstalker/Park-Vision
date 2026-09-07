FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY streamlit_app.py calibrate_tool.py ./

ENV CALIBRATION_DIR=/srv/calibrations
RUN mkdir -p ${CALIBRATION_DIR}
VOLUME ["/srv/calibrations"]

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
