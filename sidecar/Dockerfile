FROM python:3.11-slim
WORKDIR /app
COPY requirements-laya.txt /app/requirements.txt
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements.txt
COPY app.py /app/sidecar/app.py
COPY laya_runner.py /app/sidecar/laya_runner.py
RUN touch /app/sidecar/__init__.py
ENV PYTHONPATH=/app
CMD ["uvicorn", "sidecar.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
