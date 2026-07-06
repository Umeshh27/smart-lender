# Smart Lender - AI-Powered Credit Risk Evaluation

Smart Lender is a machine learning-powered web application designed to predict the creditworthiness of loan applicants, enabling banks and financial institutions to make faster, data-driven loan approval decisions. The platform leverages classification algorithms to evaluate applicant data (such as income, credit score history, loan amounts, and dependency statuses) and determine the likelihood of loan repayment or default.

---

## 📚 Technical Documentation Directory

For deep-dives into the codebase, APIs, and ML architectures, reference our structured guides:

* 🏗️ **[System Architecture Guide](./docs/ARCHITECTURE.md)**: Deep-dive into data cleaning, mode/median imputation, feature mappings, SMOTE class balancing, scaling, and XGBoost training.
* ⚡ **[API Reference Specification](./docs/API.md)**: Full routing details, JSON request/response schemas, risk evaluation trigger thresholds, and CSV ingestion schemas.
* ⚙️ **[Developer & Setup Guide](./docs/DEVELOPER.md)**: Workspace configuration steps, dependency lists, retraining instructions, and troubleshooting.

---

## ⚡ Key Features

* **Single Applicant Credit Evaluator**: An interactive web form to input applicant details and instantly receive a loan approval decision, confidence score, and risk flags.
* **Batch Analyst Queue**: Drag-and-drop CSV uploader for financial analysts to evaluate multiple applicants concurrently during high-volume periods, returning summary statistics and a downloadable prediction report.
* **Advanced Risk Flagging**: Automatically detects missing credit histories, low applicant incomes, and high debt-to-income ratios.
* **Enterprise UI Theme**: Clean, responsive Light Mode financial dashboard layout (using standard Inter typography and soft slate variables) with no glowing neon or radium effects.

---

## 📂 Project Structure

```text
smart-lender/
├── train.py                  # Training pipeline (downloads dataset, preprocesses, trains 4 models, saves XGBoost)
├── app.py                    # Flask server handling web routing, single and batch API prediction requests
├── test_applicants.csv       # Sample batch evaluation CSV containing 10 test applicant records
├── train.csv                 # Source dataset (auto-downloaded from hosted Git repository)
├── docs/                     # Comprehensive documentation
│   ├── ARCHITECTURE.md       # Pipeline, SMOTE, and scaling architecture
│   ├── API.md                # Endpoint parameters and JSON schemas
│   └── DEVELOPER.md          # Setup and model retraining guide
├── static/
│   ├── css/
│   │   └── style.css         # Modern light-mode enterprise-level stylesheet
│   └── js/
│       └── main.js           # Handles tab controls, drag-and-drop file upload, and rendering
└── templates/
    ├── home.html             # Landing page with pipeline overview and model comparisons
    ├── predict.html          # Dual-tab page containing Single Form and Batch Uploader
    └── submit.html           # Prediction result page detailing approval status, confidence, and risk flags
```

---

## ⚙️ Quick Start

### 1. Install Dependencies
```bash
pip install scikit-learn xgboost flask imbalanced-learn pandas numpy requests
```

### 2. Generate Serialized Model
Train the classifiers and generate the serialized model file `rdf.pkl`:
```bash
python train.py
```

### 3. Launch Flask App
Start the development server locally:
```bash
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## 🧪 Verification Scenarios

### Scenario 1: Low-Risk Applicant Approval
* **Input Parameters**: Salaried Graduate, Married, 0 Dependents, High Income (₹6,000 applicant / ₹2,000 co-applicant), Good Credit History, Semiurban property.
* **Expected Result**: **Approved** (Low Risk, ~80.8% Confidence).

### Scenario 2: High-Risk Applicant Detection
* **Input Parameters**: Unmarried Self-Employed Non-Graduate, Low Income (₹1,800 applicant / ₹0 co-applicant), High Loan Amount (₹150K requested), No Credit Score, Rural property.
* **Expected Result**: **Rejected** (High Risk, ~91.6% Confidence, flags low income and missing credit score).

### Scenario 3: Bulk CSV Evaluation
* **Input**: Switch to the **Batch Evaluation** tab and upload the [test_applicants.csv](./test_applicants.csv) file from the project root.
* **Expected Result**: Dashboard renders summary showing **10 evaluated applicants**, **6 approved**, **4 rejected** (60.0% approval rate) and provides a downloadable CSV containing decision tags.
