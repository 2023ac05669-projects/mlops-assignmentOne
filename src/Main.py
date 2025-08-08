from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from datetime import datetime
import sqlite3

from fastapi import FastAPI
from contextlib import asynccontextmanager
import nbformat
from nbclient import NotebookClient
from pathlib import Path


# FastAPI lifespan hook — runs on app startup and shutdown

# Function to execute the notebook
def run_notebook( input_path=Path(__file__).parent / ".." / "notebooks" / "MLOps.ipynb"):
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

# Run the notebook before the API starts serving requests
run_notebook()

##logging setup

# Set up logging
logging.basicConfig(filename="logs/api.log", level=logging.INFO)

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

# Load trained model (adjust the path as needed)
model = joblib.load("models/betterModel.pkl")
# Define request schema
class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
                             data.Population, data.AveOccup, data.Latitude, data.Longitude]])
    
    prediction = model.predict(input_array)[0]

    #logging the input and prediction
    log(data, prediction)
    return {"predicted_house_value": round(prediction, 3)}


@app.get("/metrics")
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
