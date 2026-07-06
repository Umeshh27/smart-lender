# Smart Lender Architecture & System Design

This document details the system design, machine learning pipeline, and data flow of the Smart Lender platform.

---

## 🏗️ System Architecture & Data Flow

Smart Lender separates its operations into two distinct pipelines:
1. **Training Pipeline (`train.py`)**: Runs offline to clean raw data, engineer features, balance target classes, train models, and serialize the best model configuration.
2. **Serving Pipeline (`app.py`)**: A Flask web application that loads the serialized model assets to perform real-time single and batch credit scoring.

The interaction and flow of data across these systems is illustrated below:

```mermaid
graph TD
    subgraph Offline Training Pipeline (train.py)
        A[Raw Dataset: train.csv] --> B[Data Imputation]
        B --> C[Feature Mapping & Categorical Encoding]
        C --> D[Log Transform Skewed Features]
        D --> E[SMOTE Class Balancing]
        E --> F[StandardScaler Scaling]
        F --> G[Model Comparison & Selection]
        G -->|Best Model: XGBoost| H[(Serialized Model Assets: rdf.pkl)]
    end

    subgraph Live Scoring & Serving Pipeline (app.py)
        I[Client Interface / Dashboard] -->|1. Form Submission / CSV Upload| J[Flask Web Server]
        H -->|Loads Model & Scaler| J
        J -->|2. Maps Mappings / Imputation| K[Feature Vector]
        K -->|3. Log Transformation| L[Transformed Features]
        L -->|4. Applies Scaler| M[Scaled Features]
        M -->|5. Predicts Probability| N[XGBoost Predictor]
        N -->|6. Approval & Confidence| O[Post-Processing & Risk Check]
        O -->|7. JSON / HTML Response| I
    end
    
    style H fill:#f9f,stroke:#333,stroke-width:2px
    style N fill:#bbf,stroke:#333,stroke-width:2px
```

---

## 🛠️ Data Preprocessing Pipeline

To ensure the classifier receives consistent data during training and inference, both `train.py` and `app.py` execute identical preprocessing steps:

### 1. Missing Value Imputation
* **Categorical Variables** (`Gender`, `Married`, `Dependents`, `Self_Employed`, `Credit_History`): Missing entries are imputed using the **statistical mode** of each column.
* **Numerical Variables** (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`): Missing entries are imputed using the **median** of each column to limit the impact of outliers.

### 2. Feature Mapping & Encoding
All categorical features are mapped to numeric values based on standard financial rules:
* **Gender**: `Male` $\rightarrow 1$, `Female` $\rightarrow 0$
* **Married**: `Yes` $\rightarrow 1$, `No` $\rightarrow 0$
* **Dependents**: `0` $\rightarrow 0$, `1` $\rightarrow 1$, `2` $\rightarrow 2$, `3+` $\rightarrow 3$
* **Education**: `Graduate` $\rightarrow 1$, `Not Graduate` $\rightarrow 0$
* **Self_Employed**: `Yes` $\rightarrow 1$, `No` $\rightarrow 0$
* **Property_Area**: `Rural` $\rightarrow 0$, `Semiurban` $\rightarrow 1$, `Urban` $\rightarrow 2$
* **Credit_History**: `Good Score / No Default` $\rightarrow 1.0$, `No Score / Past Default` $\rightarrow 0.0$

### 3. Outlier Handling (Log Transform)
Applicant income, co-applicant income, and loan amounts tend to be highly right-skewed. To normalize these distributions, we apply a log-transformation:
\[X_{\text{transformed}} = \ln(X + 1)\]
Using `np.log1p` prevents numerical errors for zero co-applicant incomes.

### 4. Class Balancing via SMOTE
The historical dataset contains significantly more approved loan applications than rejected ones (class imbalance). In the training phase:
* **SMOTE** (Synthetic Minority Over-sampling Technique) is applied to the training split.
* It synthesizes new, realistic credit rejection records in the feature space by interpolating between existing minority instances, protecting the classifier from bias toward approvals.

### 5. Feature Scaling
A `StandardScaler` fits on the balanced training set to center the features (mean = 0, variance = 1). This scaler is saved inside `rdf.pkl` and applied to live single or batch applicant requests before inference.

---

## 🤖 Model Evaluation & Serialization

During the model comparison step, `train.py` trains and evaluates four classification algorithms on a stratified 20% validation split:

| Classifier Model | Key Characteristics | Target Performance Metric |
| :--- | :--- | :---: |
| **K-Nearest Neighbors (KNN)** | Distance-based cluster evaluation | Testing Accuracy: ~76.8% |
| **Decision Tree** | Simple rule-based splitting, max depth 5 | Testing Accuracy: ~78.5% |
| **Random Forest** | Bagging ensemble of decision trees | Testing Accuracy: ~80.3% |
| **XGBoost Classifier** | Gradient boosted trees ensemble | **Testing Accuracy: ~81.1%** |

### Serialization Format (`rdf.pkl`)
The selected model (XGBoost) and the training preprocessing parameters are packaged into a Python `pickle` payload:
```python
model_payload = {
    'model_name': 'XGBoost',
    'model': xgb_model_instance,
    'scaler': standard_scaler_instance,
    'features': feature_names_list,
    'metrics': {
        'xgboost_train_acc': 0.947,
        'xgboost_test_acc': 0.811
    }
}
```
This payload is loaded on Flask server startup to execute rapid in-memory classification.
