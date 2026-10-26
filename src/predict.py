
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "credit_default_model.joblib"


def load_model():
    """Load the trained credit default prediction pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}. "
            "Run the model-training notebook first."
        )

    return joblib.load(MODEL_PATH)


def predict_default(customer_data: dict, model=None) -> dict:
    """
    Predict whether a customer may default on credit card payments.

    Args:
        customer_data: Dictionary containing the 23 model input features.
        model: Optional preloaded model for reuse or testing.

    Returns:
        Dictionary containing the predicted class and default probability.
    """
    if model is None:
        model = load_model()

    input_df = pd.DataFrame([customer_data])

    expected_features = list(model.feature_names_in_)
    missing_features = [
        feature for feature in expected_features
        if feature not in input_df.columns
    ]
    unexpected_features = [
        feature for feature in input_df.columns
        if feature not in expected_features
    ]

    if missing_features or unexpected_features:
        raise ValueError(
            f"Missing features: {missing_features}; "
            f"unexpected features: {unexpected_features}"
        )

    input_df = input_df[expected_features]

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    return {
        "prediction": prediction,
        "prediction_label": (
            "Default" if prediction == 1 else "No Default"
        ),
        "default_probability": round(probability, 4),
    }
