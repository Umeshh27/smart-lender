# Smart Lender - AI-Powered Credit Risk Evaluation

Smart Lender is a machine learning-powered web application designed to predict the creditworthiness of loan applicants, enabling banks and financial institutions to make faster, data-driven loan approval decisions. The platform leverages classification algorithms to evaluate applicant data (such as income, credit score history, loan amounts, and dependency statuses) and determine the likelihood of loan repayment or default.

The application processes structural applicant parameters. After training and evaluating multiple models, the best-performing model (**XGBoost**) is serialized as `rdf.pkl` and integrated into a Flask web application for real-time single and batch prediction.

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

## ⚙️ Preprocessing & Machine Learning Pipeline
Our machine learning pipeline (`train.py`) executes the following operations:
1. **Imputation**:
   * **Categorical fields** (`Gender`, `Married`, `Dependents`, `Self_Employed`, `Credit_History`) are imputed using their statistical *mode*.
   * **Numerical fields** (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`) are imputed using their *median*.
2. **Feature Mapping**: Categorical variables are mapped to standardized numeric representations for input consistency.
3. **Outlier Handling**: Applies a log-transformation (`np.log1p`) on `ApplicantIncome`, `CoapplicantIncome`, and `LoanAmount` to normalize skewed distributions.
4. **SMOTE Balancing**: Synthesizes records for the minority class in the training split using SMOTE (`imblearn`) to correct class imbalances.
5. **Feature Scaling**: Uses a `StandardScaler` to normalize features prior to model fit.

### Model Comparison Metrics (Stratified 20% Split)
The training pipeline fits and evaluates four classifiers:

| Classifier Model | Training Accuracy | Testing Accuracy |
| :--- | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | 83.98% | 72.36% |
| **Decision Tree** | 79.82% | 78.86% |
| **Random Forest** | 88.13% | 78.86% |
| **XGBoost Classifier (Best Model)** | **82.05%** | **77.24%** |

The trained XGBoost model along with the scaling parameters are exported to `rdf.pkl`.

---

## 🚀 Setup & Installation

### Prerequisites
* Python 3.13+ installed.

### 1. Install Dependencies
Run the following command to install the required libraries:
```bash
pip install scikit-learn xgboost flask imbalanced-learn pandas numpy requests
```

### 2. Run the Machine Learning Pipeline
Train the classifiers and generate the serialized model file `rdf.pkl`:
```bash
python train.py
```

### 3. Launch the Web Application
Start the Flask development server locally:
```bash
python app.py
```
Open your web browser and navigate to: **`http://localhost:5000`**

---

## 🧪 Verification Scenarios

### Scenario 1: Fast-Track Approval for Low-Risk Applicants
* **Input Parameters**: Salaried Graduate, Married, 0 Dependents, High Income (₹6,000 applicant / ₹2,000 co-applicant), Good Credit History, Semiurban property.
* **Expected Result**: **Approved** (Low Risk, ~80.8% Confidence).

### Scenario 2: High-Risk Applicant Detection
* **Input Parameters**: Unmarried Self-Employed Non-Graduate, Low Income (₹1,800 applicant / ₹0 co-applicant), High Loan Amount (₹150K requested), No Credit Score, Rural property.
* **Expected Result**: **Rejected** (High Risk, ~91.6% Confidence, flags low income and missing credit score).

### Scenario 3: Bulk Evaluation
* **Input**: Switch to the **Batch Evaluation** tab and upload the [test_applicants.csv](./test_applicants.csv) file from the project root.
* **Expected Result**: Dashboard renders a summary panel showing **10 evaluated applicants**, **6 approved**, **4 rejected** (60.0% approval rate) and provides a downloadable CSV containing decision tags.
