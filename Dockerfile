# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir nbformat nbclient pydantic fastapi uvicorn scikit-learn==1.2.2 joblib numpy==1.23.5 ipykernel pandera prometheus_fastapi_instrumentator pandas

# Expose port for Main app
EXPOSE 8090

#expose Port for mlflow
EXPOSE 5000

#expose port for prometheus
EXPOSE 9090

# Start API
CMD ["uvicorn", "src.Main:app", "--host", "0.0.0.0", "--port", "8090"]