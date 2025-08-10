from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from datetime import datetime
import sqlite3

import pandas as pd

from fastapi import FastAPI
from contextlib import asynccontextmanager
import nbformat
from nbclient import NotebookClient
from pathlib import Path
from src.schemas import PredictRequest, PredictResponse, HousingData

from prometheus_fastapi_instrumentator import Instrumentator
from time import perf_counter
from fastapi import Request, Response
from src.metrics import PREDICTION_COUNTER, PREDICTION_LATENCY


# FastAPI lifespan hook — runs on app startup and shutdown

# Function to execute the notebook
def run_notebook(input_path=Path(__file__).parent / ".." / "notebooks" / "MLOps.ipynb"):
    # Read the notebook file into memory as a NotebookNode object

    nb = nbformat.read(input_path, as_version=4)

    # Create a notebook execution client
    client = NotebookClient(
        nb,              # the notebook content
        timeout=1200,    # max time per cell (seconds)
        kernel_name="python3"  # which kernel to run it on
    )

    # Execute all cells in order, storing outputs back into nb
    client.execute()

    output_path=Path(__file__).parent / ".." / "notebooks" / "MLOps_ouput.ipynb"
    # Save the executed notebook (with outputs) to disk
    nbformat.write(nb, output_path)
    print("Reached here")

##logging setup

newRowsCounter = 0
pendingData = []

# Set up logging
logging.basicConfig(filename="logs/api.log", level=logging.INFO)

CSV_PATH = Path.cwd() / "data/raw/california_housing_raw.csv"

# SQLite DB for structured logging
conn = sqlite3.connect("logs/predictions.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    timestamp TEXT,
    input TEXT,
    prediction REAL
)
""")
conn.commit()

# Loading best model
model = joblib.load(Path.cwd()/"models/BestModel/betterModel.pkl")

app = FastAPI()

#add instrumentation endpoint
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

##API to add data and trigger model re-trainig for every 5 rows
@app.post("/addData")
def predict(body: HousingData):
    global newRowsCounter
    newRowsCounter += 1
    # Convert Pydantic model to DataFrame
    df_new = pd.DataFrame([body.model_dump()])

    # Append to CSV (no header, keep existing file)
    df_new.to_csv(CSV_PATH, mode="a", header=False, index=False)

    df = pd.read_csv(CSV_PATH)

    if newRowsCounter >=5:
        run_notebook()
        model = joblib.load(Path.cwd()/"models/BestModel/betterModel.pkl")
        newRowsCounter = 0
        return f"new Model Loaded: Shape: {df.shape}"
    return f"new Data Added : shape : {df.shape}"
        


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):

    t0 = perf_counter()
    try: 
        X = [[body.MedInc, body.HouseAge, body.AveRooms, body.AveBedrms,
            body.Population, body.AveOccup, body.Latitude, body.Longitude]]
        y = model.predict(X)[0]

        #logging the input and prediction
        log(body, y)
        status = "ok"
        return {"MedHouseVal": float(y)}
    except Exception:
        status = "error"
        raise
    finally:
        PREDICTION_COUNTER.labels(model_name="ModelInUse", status=status).inc()
        PREDICTION_LATENCY.observe(perf_counter() - t0)


@app.get("/modelMetrics")
def metrics():
    cursor.execute("SELECT COUNT(*), AVG(prediction) FROM predictions")
    count, avg = cursor.fetchone()
    return {
        "total_predictions": count,
        "average_prediction": round(avg or 0, 2)
    }


def log(data, prediction):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": data.dict(),
        "prediction": float(prediction)
    }

    # File log
    logging.info(str(log_entry))

    # SQLite log
    cursor.execute("INSERT INTO predictions VALUES (?, ?, ?)",
                   (log_entry["timestamp"], str(log_entry["input"]), log_entry["prediction"]))
    conn.commit()
