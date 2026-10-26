
import pytest

from src.predict import load_model, predict_default


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def valid_customer(model):
    """Create a valid input using the model's expected features."""
    return {
        feature: 0
        for feature in model.feature_names_in_
    }


def test_prediction_returns_valid_class(model, valid_customer):
    result = predict_default(valid_customer, model=model)

    assert result["prediction"] in (0, 1)
    assert result["prediction_label"] in ("Default", "No Default")


def test_default_probability_is_between_zero_and_one(
    model, valid_customer
):
    result = predict_default(valid_customer, model=model)

    assert 0.0 <= result["default_probability"] <= 1.0


def test_prediction_contains_expected_fields(model, valid_customer):
    result = predict_default(valid_customer, model=model)

    assert set(result.keys()) == {
        "prediction",
        "prediction_label",
        "default_probability",
    }


def test_missing_feature_is_rejected(model, valid_customer):
    invalid_customer = valid_customer.copy()
    invalid_customer.pop("LIMIT_BAL")

    with pytest.raises(ValueError, match="Missing features"):
        predict_default(invalid_customer, model=model)


def test_unexpected_feature_is_rejected(model, valid_customer):
    invalid_customer = valid_customer.copy()
    invalid_customer["unexpected_feature"] = 123

    with pytest.raises(ValueError, match="unexpected features"):
        predict_default(invalid_customer, model=model)
