
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_model, predict_default


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model when the API starts."""
    global model
    model = load_model()
    yield
    model = None


app = FastAPI(
    title="Credit Default Prediction API",
    description=(
        "Predict credit card payment default risk "
        "using a trained Random Forest model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerInput(BaseModel):
    """Customer features required by the trained model."""

    model_config = ConfigDict(extra="forbid")

    LIMIT_BAL: float = Field(ge=0)
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int = Field(gt=0)

    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Credit Default Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded",
        )

    try:
        result = predict_default(
            customer.model_dump(),
            model=model,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
