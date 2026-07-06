# Smart Lender API Reference Specification

This document details the HTTP endpoints provided by the Smart Lender Flask application, including their routing, parameters, expected schemas, and error responses.

---

## ⚡ Quick Endpoints Summary

| Endpoint | Method | Request Content-Type | Description |
| :--- | :---: | :--- | :--- |
| [`/`](#get-) | `GET` | N/A | Landing dashboard with pipeline stats |
| [`/predict`](#get-predict) | `GET` | N/A | Interactive evaluation client |
| [`/predict`](#post-predict) | `POST` | `application/json` or `application/x-www-form-urlencoded` | Score a single applicant |
| [`/batch-predict`](#post-batch-predict) | `POST` | `multipart/form-data` | Process batch applicant CSV queue |
| [`/download-results`](#get-download-results) | `GET` | N/A | Downloads last processed batch CSV |

---

## 🔍 Endpoint Specifications

### `GET /`
Serves the platform dashboard demonstrating validation pipeline accuracy metrics.

* **Response Status**: `200 OK`
* **Response Content-Type**: `text/html`

---

### `GET /predict`
Renders the frontend application interface (form/upload tab panes).

* **Response Status**: `200 OK`
* **Response Content-Type**: `text/html`

---

### `POST /predict`
Evaluates credit risk for a single applicant record.

* **Request Content-Type**: `application/json` or `application/x-www-form-urlencoded`
* **Parameters Schema**:

| Parameter | Type | Required | Values / Constraints | Description |
| :--- | :---: | :---: | :--- | :--- |
| `Gender` | String | No | `Male`, `Female` (Default: `Male`) | Applicant gender identity |
| `Married` | String | No | `Yes`, `No` (Default: `No`) | Marital status |
| `Dependents` | String | No | `0`, `1`, `2`, `3+` (Default: `0`) | Number of dependents |
| `Education` | String | No | `Graduate`, `Not Graduate` (Default: `Graduate`) | Academic qualification level |
| `Self_Employed` | String | No | `Yes`, `No` (Default: `No`) | Business/Self-employment status |
| `ApplicantIncome` | Number | Yes | $\ge 0$ (Default: `5000`) | Primary applicant monthly income in ₹ |
| `CoapplicantIncome`| Number | Yes | $\ge 0$ (Default: `0`) | Co-applicant monthly income in ₹ |
| `LoanAmount` | Number | Yes | $> 0$ (Default: `150`) | Requested loan amount in Thousands (₹) |
| `Loan_Amount_Term` | Number | Yes | Positive integer (Default: `360`) | Term duration in months |
| `Credit_History` | Number | Yes | `1.0` (Good score), `0.0` (Default/No score) | Applicant credit record standing |
| `Property_Area` | String | No | `Rural`, `Semiurban`, `Urban` (Default: `Semiurban`) | Property location classification |

#### JSON Request Example
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 6000,
  "CoapplicantIncome": 2000,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1.0,
  "Property_Area": "Semiurban"
}
```

#### JSON Response Example (`200 OK`)
If the client submits with header `Content-Type: application/json`, the server responds with a JSON payload:
```json
{
  "prediction": "Approved",
  "confidence": "80.8%",
  "risk_level": "Low",
  "risk_factors": [],
  "input_data": {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 6000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Semiurban"
  }
}
```

#### Risk Analysis Rules
The system evaluates and triggers three specific credit risk flags based on standard guidelines:
1. **Bad Credit History**: Triggers if `Credit_History` = `0.0` ("No credit history or poor credit record detected.").
2. **Low Income**: Triggers if `ApplicantIncome` < 2500 and `CoapplicantIncome` = 0 ("Low applicant income with no coapplicant support.").
3. **High Debt-to-Income (DTI)**: Triggers if the requested `LoanAmount` exceeds estimated capacity based on monthly income and loan term:
   \[\text{LoanAmount} > \text{ApplicantIncome} \times 0.05 \times \left(\frac{\text{Loan\_Amount\_Term}}{12}\right)\]
   ("High debt-to-income ratio (requested loan amount is high relative to income).")

---

### `POST /batch-predict`
Processes a queue of applicant records uploaded in CSV format.

* **Request Content-Type**: `multipart/form-data`
* **Form Parameters**:
  * `file`: The `.csv` applicant spreadsheet file.
* **Required CSV Header Columns**:
  `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`

#### JSON Response Example (`200 OK`)
```json
{
  "total": 10,
  "approved": 6,
  "rejected": 4,
  "approval_rate": "60.0%",
  "download_url": "/download-results",
  "results": [
    {
      "ApplicantIncome": 5849,
      "LoanAmount": 128.0,
      "Credit_History": 1.0,
      "Prediction": "Approved",
      "Confidence": "79.4%"
    },
    {
      "ApplicantIncome": 4583,
      "LoanAmount": 128.0,
      "Credit_History": 1.0,
      "Prediction": "Rejected",
      "Confidence": "53.2%"
    }
  ]
}
```

#### CSV Processing Steps
1. Validates that all 11 target columns are present; returns error if any are missing.
2. Fills missing elements using local dataset modal values (categorical) and median values (numerical).
3. Executes feature mapping and outlier log-transformations.
4. Performs standard scaling using the serialized fitted scaler.
5. Scores the entire dataset using the XGBoost model.
6. Appends `Prediction` and `Confidence` columns and writes the file locally to `temp/predictions_output.csv`.

---

### `GET /download-results`
Downloads the scored batch results generated in the last run of `/batch-predict`.

* **Response Status**: `200 OK` (delivers file `smart_lender_predictions.csv`) or `404 Not Found` (if no batch run has been executed).
* **Response Content-Type**: `text/csv`

---

## 🚫 Error Responses

When errors occur during data ingestion or model processing, the endpoints return a JSON payload detailing the failure.

### Missing Model Error (`500 Internal Server Error`)
Returned by `/predict` or `/batch-predict` if the Flask server was started without training the machine learning pipeline (`rdf.pkl` is absent):
```json
{
  "error": "Model not trained yet. Please run train.py first."
}
```

### Invalid File Format (`400 Bad Request`)
Returned by `/batch-predict` if the uploaded file is not a `.csv`:
```json
{
  "error": "Invalid file format. Please upload a CSV file."
}
```

### Missing Structural Headers (`400 Bad Request`)
Returned by `/batch-predict` if the CSV header is incomplete:
```json
{
  "error": "Missing columns in CSV: Credit_History, Property_Area"
}
```
