import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file

app = Flask(__name__)

# Global variables to hold the model and preprocessing assets
model_assets = None

def load_model():
    global model_assets
    model_path = os.path.join(os.path.dirname(__file__), 'rdf.pkl')
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_assets = pickle.load(f)
            print("Successfully loaded model from rdf.pkl.")
        except Exception as e:
            print(f"Error loading rdf.pkl: {e}")
    else:
        print("rdf.pkl not found. Please train the model first.")

@app.before_request
def initialize():
    if model_assets is None:
        load_model()

# Helper for preprocessing single prediction
def preprocess_input(data):
    # Mapping definitions matching train.py
    gender_map = {'Male': 1, 'Female': 0}
    married_map = {'Yes': 1, 'No': 0}
    dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
    education_map = {'Graduate': 1, 'Not Graduate': 0}
    self_employed_map = {'Yes': 1, 'No': 0}
    property_area_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
    
    try:
        # Extract features
        gender = gender_map.get(data.get('Gender', 'Male'), 1)
        married = married_map.get(data.get('Married', 'No'), 0)
        dependents = dependents_map.get(data.get('Dependents', '0'), 0)
        education = education_map.get(data.get('Education', 'Graduate'), 1)
        self_employed = self_employed_map.get(data.get('Self_Employed', 'No'), 0)
        
        # Numerical inputs with default fallbacks
        applicant_income = float(data.get('ApplicantIncome', 5000))
        coapplicant_income = float(data.get('CoapplicantIncome', 0))
        loan_amount = float(data.get('LoanAmount', 150))
        loan_amount_term = float(data.get('Loan_Amount_Term', 360))
        
        credit_history = float(data.get('Credit_History', 1.0))
        property_area = property_area_map.get(data.get('Property_Area', 'Semiurban'), 1)
        
        # Apply log transformations matching train.py
        applicant_income_log = np.log1p(applicant_income)
        coapplicant_income_log = np.log1p(coapplicant_income)
        loan_amount_log = np.log1p(loan_amount)
        
        # Construct feature vector in exact order
        # ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
        #  'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
        #  'Credit_History', 'Property_Area']
        features = [
            gender, married, dependents, education, self_employed,
            applicant_income_log, coapplicant_income_log, loan_amount_log, loan_amount_term,
            credit_history, property_area
        ]
        
        return np.array(features).reshape(1, -1), None
    except Exception as e:
        return None, f"Input processing error: {str(e)}"

@app.route('/')
def home():
    metrics = {}
    if model_assets:
        metrics = model_assets.get('metrics', {})
    return render_template('home.html', metrics=metrics)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if model_assets is None:
        return render_template('predict.html', error="Model not trained yet. Please run train.py first.")
        
    if request.method == 'POST':
        # Accept JSON or form data
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()
            
        features_vec, err = preprocess_input(data)
        if err:
            return render_template('predict.html', error=err)
            
        # Scale features
        scaler = model_assets['scaler']
        model = model_assets['model']
        
        features_scaled = scaler.transform(features_vec)
        
        # Predict
        prediction = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(probabilities[prediction])
        
        # Result analysis
        result = "Approved" if prediction == 1 else "Rejected"
        
        # Assess risk factors
        risk_factors = []
        if float(data.get('Credit_History', 1.0)) == 0.0:
            risk_factors.append("No credit history or poor credit record detected.")
        if float(data.get('ApplicantIncome', 5000)) < 2500 and float(data.get('CoapplicantIncome', 0)) == 0:
            risk_factors.append("Low applicant income with no coapplicant support.")
        if float(data.get('LoanAmount', 150)) > float(data.get('ApplicantIncome', 5000)) * 0.05 * (float(data.get('Loan_Amount_Term', 360)) / 12):
            risk_factors.append("High debt-to-income ratio (requested loan amount is high relative to income).")
            
        response_data = {
            'prediction': result,
            'confidence': f"{confidence * 100:.1f}%",
            'risk_level': "Low" if prediction == 1 and not risk_factors else "High",
            'risk_factors': risk_factors,
            'input_data': data
        }
        
        if request.is_json:
            return jsonify(response_data)
        return render_template('submit.html', result=response_data)
        
    return render_template('predict.html')

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    if model_assets is None:
        return jsonify({'error': "Model not trained yet. Please run train.py first."}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400
        
    try:
        df = pd.read_csv(file)
        
        # Verify required columns exist
        required_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                         'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                         'Credit_History', 'Property_Area']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns in CSV: {", ".join(missing_cols)}'}), 400
            
        # Copy to perform transformations
        df_proc = df.copy()
        
        # Preprocessing mappings (matching train.py)
        gender_map = {'Male': 1, 'Female': 0}
        married_map = {'Yes': 1, 'No': 0}
        dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
        education_map = {'Graduate': 1, 'Not Graduate': 0}
        self_employed_map = {'Yes': 1, 'No': 0}
        property_area_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
        
        # Imputations matching train.py
        df_proc['Gender'] = df_proc['Gender'].fillna(df_proc['Gender'].mode()[0] if not df_proc['Gender'].empty else 'Male').map(gender_map)
        df_proc['Married'] = df_proc['Married'].fillna(df_proc['Married'].mode()[0] if not df_proc['Married'].empty else 'No').map(married_map)
        df_proc['Dependents'] = df_proc['Dependents'].fillna('0').astype(str).map(dependents_map)
        df_proc['Education'] = df_proc['Education'].fillna('Graduate').map(education_map)
        df_proc['Self_Employed'] = df_proc['Self_Employed'].fillna('No').map(self_employed_map)
        
        df_proc['ApplicantIncome'] = df_proc['ApplicantIncome'].fillna(df_proc['ApplicantIncome'].median())
        df_proc['CoapplicantIncome'] = df_proc['CoapplicantIncome'].fillna(0)
        df_proc['LoanAmount'] = df_proc['LoanAmount'].fillna(df_proc['LoanAmount'].median())
        df_proc['Loan_Amount_Term'] = df_proc['Loan_Amount_Term'].fillna(360)
        df_proc['Credit_History'] = df_proc['Credit_History'].fillna(1.0)
        df_proc['Property_Area'] = df_proc['Property_Area'].fillna('Semiurban').map(property_area_map)
        
        # Handle NaNs from mapping discrepancies
        for col in required_cols:
            if df_proc[col].isnull().any():
                df_proc[col] = df_proc[col].fillna(0)
                
        # Log transforms
        df_proc['ApplicantIncome'] = np.log1p(df_proc['ApplicantIncome'])
        df_proc['CoapplicantIncome'] = np.log1p(df_proc['CoapplicantIncome'])
        df_proc['LoanAmount'] = np.log1p(df_proc['LoanAmount'])
        
        # Feature Matrix
        X_batch = df_proc[required_cols]
        
        scaler = model_assets['scaler']
        model = model_assets['model']
        
        # Scale
        X_scaled = scaler.transform(X_batch)
        
        # Predict
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)
        
        # Store results
        df['Prediction'] = ['Approved' if p == 1 else 'Rejected' for p in preds]
        df['Confidence'] = [f"{probs[i][p] * 100:.1f}%" for i, p in enumerate(preds)]
        
        # Save output to a temp file
        os.makedirs('temp', exist_ok=True)
        out_path = 'temp/predictions_output.csv'
        df.to_csv(out_path, index=False)
        
        # Summary statistics
        total = len(df)
        approved = int((preds == 1).sum())
        rejected = total - approved
        
        summary = {
            'total': total,
            'approved': approved,
            'rejected': rejected,
            'approval_rate': f"{(approved / total) * 100:.1f}%" if total > 0 else "0%",
            'download_url': '/download-results',
            'results': df[['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Prediction', 'Confidence']].head(10).to_dict(orient='records')
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': f"CSV processing error: {str(e)}"}), 500

@app.route('/download-results')
def download_results():
    out_path = 'temp/predictions_output.csv'
    if os.path.exists(out_path):
        return send_file(out_path, as_attachment=True, download_name='smart_lender_predictions.csv')
    return "Predictions file not found.", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
