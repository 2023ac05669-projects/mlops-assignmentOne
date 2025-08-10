# src/schemas.py
from pydantic import BaseModel, Field, conlist

# Single prediction (8 features)
class PredictRequest(BaseModel):
    MedInc: float = Field(..., gt=0)
    HouseAge: float = Field(..., ge=0)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., ge=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float
    Longitude: float

class PredictResponse(BaseModel):
    MedHouseVal: float


class HousingData(BaseModel):
    MedInc: float = Field(..., gt=0)
    HouseAge: float = Field(..., ge=0)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., ge=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float
    Longitude: float
    MedHouseVal: float