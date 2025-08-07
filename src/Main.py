from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from datetime import datetime
import sqlite3


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
model_lr = joblib.load("models/linear_regression.pkl")
model_dt = joblib.load("models/decision_tree_model.pkl")
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

@app.post("/predictLR")
def predict(data: InputData):
    input_array = np.array([[data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
                             data.Population, data.AveOccup, data.Latitude, data.Longitude]])
    
    prediction = model_lr.predict(input_array)[0]

    #logging the input and prediction
    log(data, prediction)
    return {"predicted_house_value": round(prediction, 3)}


@app.post("/predictDR")
def predict(data: InputData):
    input_array = np.array([[data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
                             data.Population, data.AveOccup, data.Latitude, data.Longitude]])
    
    prediction = model_dt.predict(input_array)[0]
    
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
