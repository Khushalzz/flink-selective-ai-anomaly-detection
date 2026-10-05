FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN mkdir -p /app/dashboard
COPY api.py index.html styles.css app.js /app/dashboard/
ENV PYTHONPATH=/app
EXPOSE 4173
CMD ["uvicorn", "dashboard.api:app", "--host", "0.0.0.0", "--port", "4173", "--workers", "1"]
