# Smart Lender Developer Setup & Contribution Guide

This guide details steps to configure, run, retrain, and debug the Smart Lender credit risk evaluator codebase.

---

## ⚙️ Prerequisites & Environment Configuration

Ensure you have **Python 3.13+** installed on your workstation.

### 1. Setup a Virtual Environment
We recommend utilizing a virtual environment to prevent dependency conflicts with your system Python path:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Required Dependencies
With your virtual environment activated, install the required packages:
```bash
pip install scikit-learn xgboost flask imbalanced-learn pandas numpy requests
```

---

## 🏃 Running the Pipeline

The project is structured with an offline machine learning model generation script and a live web server.

### Phase 1: Train & Serialize the Model
To train the classifier algorithms and save the pipeline assets, execute the training script:
```bash
python train.py
```
**What happens under the hood:**
1. The script downloads `train.csv` from the hosted repository if it is not present in the root directory.
2. The dataset is imputed, encoded, log-transformed, balanced via SMOTE, and standardized.
3. Four machine learning algorithms are trained and evaluated on stratified splits.
4. The best-performing model (`XGBoost`) along with its associated `StandardScaler` are packaged and saved to `rdf.pkl` in the root folder.

### Phase 2: Start the Web Server
After creating `rdf.pkl`, you can launch the Flask server:
```bash
python app.py
```
Open your browser and navigate to **`http://localhost:5000`** to access the dashboard.

---

## 📂 Codebase File Layout

```text
smart-lender/
├── train.py                  # Model training and comparison script
├── app.py                    # Web server, request processing, risk checking
├── train.csv                 # Source dataset (auto-downloaded from hosted Git repository)
├── test_applicants.csv       # Sample batch evaluation CSV containing 10 test records
├── rdf.pkl                   # Output binary from train.py (Git ignored or tracked)
├── docs/                     # Detailed project specifications
│   ├── ARCHITECTURE.md       # Preprocessing and Pipeline details
│   ├── API.md                # Server HTTP routes and schema documentation
│   └── DEVELOPER.md          # Setup and training guide (This file)
├── static/                   # Static dashboard assets
│   ├── css/
│   │   └── style.css         # Minimal light-mode enterprise CSS stylesheet
│   └── js/
│       └── main.js           # Handles drag-and-drop uploads, UI tabs, and API polling
└── templates/                # Jinja2 HTML Page Layouts
    ├── home.html             # Dashboard homepage with models stats table
    ├── predict.html          # Tabs containing Form and Batch CSV upload
    └── submit.html           # Post-evaluation output details page
```

---

## 🧪 Testing and Verification

To verify that the model works correctly, test it with the following scenarios on the frontend (`/predict` page):

### 1. Verification of Single Prediction: Low Risk Approval
* **Input Parameters**: Salaried Graduate, Married, 0 Dependents, High Income (₹6,000 / month applicant / ₹2,000 co-applicant), Good Credit History (`1.0`), Semiurban property.
* **Expected Result**: **Approved** (Low Risk, ~80.8% Confidence).

### 2. Verification of Single Prediction: High Risk Rejection
* **Input Parameters**: Unmarried Self-Employed Non-Graduate, Low Income (₹1,800 / month applicant / ₹0 co-applicant), High Loan Amount (₹150K requested), No Credit Score (`0.0`), Rural property.
* **Expected Result**: **Rejected** (High Risk, ~91.6% Confidence, with flags highlighting low income and lack of credit score).

### 3. Verification of Batch Prediction
* Upload the sample CSV file [test_applicants.csv](../test_applicants.csv) via the drag-and-drop batch area.
* **Expected Result**: Summary dashboard updates instantly, rendering **10 evaluated applicants**, **6 approved**, **4 rejected** (60.0% approval rate) and provides a downloadable CSV containing evaluation tags.

---

## 🛠️ Troubleshooting

### Error: `rdf.pkl not found. Please train the model first.`
* **Cause**: You tried starting the server or accessing predictions before generating the serialized model payload.
* **Fix**: Shut down the server (`Ctrl+C`) and run:
  ```bash
  python train.py
  ```
  Verify that `rdf.pkl` has been created in your root project folder before executing `python app.py`.

### Error: `ModuleNotFoundError: No module named 'imblearn'`
* **Cause**: `imbalanced-learn` is missing from the current Python virtual environment.
* **Fix**: Run:
  ```bash
  pip install imbalanced-learn
  ```
