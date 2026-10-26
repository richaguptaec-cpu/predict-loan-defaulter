# Credit Default Risk Prediction

An end-to-end machine learning project that predicts the likelihood of credit card payment default using customer demographic, credit limit, repayment status, bill statement, and payment history features.

The project combines data exploration, model training, hyperparameter tuning, a reusable prediction module, a FastAPI REST API, an interactive Streamlit dashboard, and automated tests.

## Project Highlights

- Explored and analyzed 30,000 customer records.
- Built and compared Logistic Regression and Random Forest classifiers.
- Tuned Random Forest hyperparameters using stratified cross-validation.
- Developed a reusable prediction pipeline with input validation.
- Exposed predictions through a FastAPI REST API.
- Built an interactive Streamlit dashboard.
- Added automated unit and API tests using Pytest.

## Technology Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **API:** FastAPI, Uvicorn, Pydantic
- **Dashboard:** Streamlit
- **Testing:** Pytest, FastAPI TestClient, HTTPX
- **Development:** VS Code, Jupyter Notebook

## Dataset

This project uses the [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

The dataset contains 30,000 records and 23 model input features, including:

- Credit limit, age, education, sex, and marital status
- Monthly repayment status
- Bill statement amounts
- Previous payment amounts

**Target:** `default payment next month`

| Target | Records | Percentage |
|---|---:|---:|
| No default | 23,364 | 77.88% |
| Default | 6,636 | 22.12% |
| Total | 30,000 | 100% |

The dataset has no missing values or duplicate rows in the inspected records. The `ID` column is excluded from model training.

## Machine Learning Workflow

1. Data loading and exploratory data analysis
2. Target distribution and data quality checks
3. Feature preparation and stratified train-test split
4. Preprocessing with feature scaling and categorical encoding
5. Logistic Regression baseline
6. Random Forest model
7. Hyperparameter tuning using `GridSearchCV` and stratified five-fold cross-validation
8. Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
9. Model persistence using Joblib

The data was split into 24,000 training records and 6,000 test records.

## Model Evaluation

### Test-set comparison

| Metric | Logistic Regression | Random Forest | Tuned Random Forest |
|---|---:|---:|---:|
| Accuracy | 80.88% | 79.20% | 78.03% |
| Precision | 69.23% | 52.99% | 50.30% |
| Recall | 24.42% | 52.68% | 56.37% |
| F1-score | 36.10% | 52.83% | 53.16% |
| ROC-AUC | 71.00% | 76.80% | 77.17% |

The tuned Random Forest achieved the highest recall, F1-score, and ROC-AUC among these three evaluated models.

Recall is particularly relevant when identifying potential defaulters. However, increasing recall can also increase false positives, so the preferred operating threshold depends on the business context.

**Evaluation note:** These are the observed results from the current workflow. The test set was also used during exploratory threshold and model comparisons, so the metrics should not be considered a completely untouched final performance estimate.

## Application Architecture

```text
Customer Input
      |
      v
Streamlit Dashboard -----> FastAPI REST API
                                  |
                                  v
                         Prediction Module
                                  |
                                  v
                      Saved ML Pipeline
                    (Preprocessing + Model)
                                  |
                                  v
                    Prediction and Probability
```

The Streamlit dashboard can call the prediction module directly. The FastAPI service independently exposes predictions over HTTP.

## Project Structure

```text
predict-loan-defaulter/
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── data/
│   └── README.md
├── models/
│   └── credit_default_model.joblib
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   └── predict.py
├── tests/
│   ├── test_predict.py
│   └── test_api.py
├── .gitignore
├── README.md
└── requirements.txt
```

The trained model and original dataset are generated or downloaded locally and should not be committed to Git.

## Setup and Installation

### Prerequisites

- Python 3.14.4 or a compatible Python version
- Git
- A local copy of the dataset

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/predict-loan-defaulter.git
cd predict-loan-defaulter
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the dataset

Download the dataset from the [official UCI dataset page](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

Place the Excel file at:

```text
data/default of credit card clients.xls
```

The notebook uses this file for data exploration and model training.

### 5. Train and save the model

Open and run:

```text
notebooks/01_data_exploration.ipynb
```

Run the notebook cells in order, including model training and model saving. This creates:

```text
models/credit_default_model.joblib
```

### 6. Start the API

From the project root:

```bash
python -m uvicorn api.main:app --reload
```

Open the interactive API documentation at:

http://127.0.0.1:8000/docs

Check the service health at:

http://127.0.0.1:8000/health

### 7. Start the dashboard

In a separate terminal:

```bash
python -m streamlit run dashboard/app.py
```

Streamlit normally opens at:

http://localhost:8501

### 8. Run automated tests

From the project root:

```bash
python -m pytest -v
```

The current implementation has nine passing tests covering prediction output, input validation, the API health endpoint, and API prediction requests.

## API Usage

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Generate a prediction

```http
POST /predict
Content-Type: application/json
```

Send all 23 required model input features in the JSON request body. The interactive Swagger documentation at `/docs` provides the request schema.

Example response structure:

```json
{
  "prediction": 1,
  "prediction_label": "Default",
  "default_probability": 0.775
}
```

The values above are illustrative. Actual predictions depend on the submitted features and saved model.

## Screenshots

Screenshots will be added to document the application's capabilities:
- FastAPI Swagger documentation
<img width="1268" height="622" alt="image" src="https://github.com/user-attachments/assets/1fe6595a-4628-42bc-8d44-38a83a78d706" />
 
- Successful API prediction response
<img width="1183" height="299" alt="image" src="https://github.com/user-attachments/assets/e881ef97-ade4-4672-85de-c3aca6ca89c6" />

- Streamlit prediction dashboard
<img width="1270" height="481" alt="image" src="https://github.com/user-attachments/assets/d0d7df9e-ff55-4419-aed3-63b879082268" />

## Key Learnings

- Building end-to-end machine learning workflows
- Handling class imbalance and evaluating recall versus precision
- Applying preprocessing consistently during training and inference
- Tuning Random Forest hyperparameters with cross-validation
- Packaging model inference into a reusable Python module
- Developing and testing REST APIs with FastAPI
- Integrating machine learning predictions into Streamlit
- Automating tests with Pytest

## Limitations and Responsible Use

- The model predicts credit card payment default based on a historical dataset; it is not a validated personal-loan underwriting model.
- The dataset reflects historical patterns and may not represent current customer populations.
- Model probabilities are estimates and have not been established as calibrated real-world default probabilities.
- The model is a portfolio demonstration, not a production-ready financial decision system.
- Real-world use would require additional validation, fairness assessment, monitoring, security controls, and domain review.

## License and Attribution

Dataset: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

Review the dataset's terms and select an appropriate license for your own code before publishing. Do not imply that the dataset or its original contents are your own work.

---

**Built as a machine learning and software engineering portfolio project.**
