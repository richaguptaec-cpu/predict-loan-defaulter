
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_default


st.set_page_config(
    page_title="Credit Default Risk Predictor",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Default Risk Predictor")
st.write(
    "Explore estimated credit card payment default risk "
    "using a trained Random Forest model."
)

st.info(
    "Portfolio demonstration only. Predictions are estimates, "
    "not financial advice or decisions about real customers."
)

with st.form("customer_form"):
    st.subheader("Customer information")

    col1, col2, col3 = st.columns(3)

    with col1:
        limit_bal = st.number_input(
            "Credit limit (LIMIT_BAL)",
            min_value=0.0,
            value=50000.0,
            step=5000.0,
        )
        sex = st.selectbox("SEX", [1, 2], index=1)
        education = st.selectbox("EDUCATION", [1, 2, 3, 4, 5, 6, 0], index=1)
        marriage = st.selectbox("MARRIAGE", [1, 2, 3, 0], index=0)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)

    with col2:
        st.markdown("**Repayment status**")
        pay_0 = st.number_input("PAY_0 (most recent)", -2, 8, 2)
        pay_2 = st.number_input("PAY_2", -2, 8, 2)
        pay_3 = st.number_input("PAY_3", -2, 8, 0)
        pay_4 = st.number_input("PAY_4", -2, 8, 0)
        pay_5 = st.number_input("PAY_5", -2, 8, 0)
        pay_6 = st.number_input("PAY_6", -2, 8, 0)

    with col3:
        st.markdown("**Bill amounts**")
        bill_amt1 = st.number_input("BILL_AMT1", value=45000.0)
        bill_amt2 = st.number_input("BILL_AMT2", value=42000.0)
        bill_amt3 = st.number_input("BILL_AMT3", value=40000.0)
        bill_amt4 = st.number_input("BILL_AMT4", value=38000.0)
        bill_amt5 = st.number_input("BILL_AMT5", value=35000.0)
        bill_amt6 = st.number_input("BILL_AMT6", value=30000.0)

    st.subheader("Recent payment amounts")
    pay_col1, pay_col2, pay_col3 = st.columns(3)

    with pay_col1:
        pay_amt1 = st.number_input("PAY_AMT1", min_value=0.0, value=0.0)
        pay_amt2 = st.number_input("PAY_AMT2", min_value=0.0, value=1000.0)

    with pay_col2:
        pay_amt3 = st.number_input("PAY_AMT3", min_value=0.0, value=2000.0)
        pay_amt4 = st.number_input("PAY_AMT4", min_value=0.0, value=2000.0)

    with pay_col3:
        pay_amt5 = st.number_input("PAY_AMT5", min_value=0.0, value=3000.0)
        pay_amt6 = st.number_input("PAY_AMT6", min_value=0.0, value=3000.0)

    submitted = st.form_submit_button(
        "Predict default risk",
        type="primary",
        use_container_width=True,
    )


if submitted:
    customer = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,
        "BILL_AMT1": bill_amt1,
        "BILL_AMT2": bill_amt2,
        "BILL_AMT3": bill_amt3,
        "BILL_AMT4": bill_amt4,
        "BILL_AMT5": bill_amt5,
        "BILL_AMT6": bill_amt6,
        "PAY_AMT1": pay_amt1,
        "PAY_AMT2": pay_amt2,
        "PAY_AMT3": pay_amt3,
        "PAY_AMT4": pay_amt4,
        "PAY_AMT5": pay_amt5,
        "PAY_AMT6": pay_amt6,
    }

    try:
        result = predict_default(customer)

        st.subheader("Prediction result")

        metric1, metric2 = st.columns(2)
        metric1.metric(
            "Predicted class",
            result["prediction_label"],
        )
        metric2.metric(
            "Estimated default probability",
            f'{result["default_probability"]:.1%}',
        )

        probability = result["default_probability"]
        st.progress(probability)

        if result["prediction"] == 1:
            st.warning(
                "The model predicts potential payment default. "
                "This is an estimate, not a certainty."
            )
        else:
            st.success(
                "The model predicts no payment default. "
                "This does not guarantee future repayment."
            )

    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Unable to generate prediction: {exc}")
