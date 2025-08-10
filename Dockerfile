# Use official Python image & prometheus image
FROM python:3.10-slim

# Set working directory
WORKDIR /app
# Install Prometheus
RUN apt-get update && apt-get install -y wget tar \
    && wget https://github.com/prometheus/prometheus/releases/download/v2.53.0/prometheus-2.53.0.linux-amd64.tar.gz \
    && tar -xzf prometheus-*.tar.gz \
    && mv prometheus-*/prometheus /usr/local/bin/prometheus \
    && rm -rf prometheus-*

# App files
COPY . .
COPY prometheus.yml /usr/local/bin/prometheus.yml
COPY start.sh /usr/local/bin/start.sh
RUN sed -i 's/\r$//' /usr/local/bin/start.sh && chmod +x /usr/local/bin/start.sh

# Python deps
RUN pip install --no-cache-dir nbformat mlflow nbclient pydantic fastapi uvicorn scikit-learn==1.2.2 joblib numpy==1.23.5 ipykernel pandera prometheus_fastapi_instrumentator pandas

EXPOSE 8090 5000 9090

# Be explicit about the absolute path
ENTRYPOINT ["/usr/local/bin/start.sh"]