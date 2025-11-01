# Dockerfile - Neura.AI v1.1
FROM python:3.11-slim
WORKDIR /app

# system deps for pandas/plotly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

CMD ["uvicorn", "xau_dashboard_v5:app", "--host", "0.0.0.0", "--port", "8000"]
