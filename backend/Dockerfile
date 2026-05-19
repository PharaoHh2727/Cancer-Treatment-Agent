FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MEDICAL_AGENT_ROOT=/app
ENV MEDICAL_AGENT_DATA_DIR=/app/runtime_data
ENV CLAM_PYTHON=python

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1 \
        libglib2.0-0 \
        libopenjp2-7 \
        libopenslide0 \
        openslide-tools \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

EXPOSE 8000
CMD ["python", "-m", "backend.api_server"]
