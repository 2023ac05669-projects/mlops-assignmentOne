#!/usr/bin/env bash
uvicorn src.Main:app --host 0.0.0.0 --port 8090 &
prometheus --config.file  /usr/local/bin/prometheus.yml --storage.tsdb.path /usr/local/bin --web.listen-address 0.0.0.0:9090 &
mlflow ui --backend-store-uri file:/app/mlruns --host 0.0.0.0 --port 5000
wait