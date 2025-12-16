from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = joblib.load("models/random_forest_model.pkl")

# Assume these are the columns used during training
feature_names = [
    'GENDER', 'Car_Owner', 'Propert_Owner', 'CHILDREN', 'Annual_income',
    'Type_Income', 'EDUCATION', 'Marital_status', 'Housing_type',
    'Birthday_count', 'Employed_days', 'Mobile_phone', 'Work_Phone',
    'Phone', 'EMAIL_ID', 'Type_Occupation', 'Family_Members'
]

# Helper to preprocess input like you did for training
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Example encoding for categorical variables (customize as per your training code)
    mapping = {
        'GENDER': {'M': 0, 'F': 1},
        'Car_Owner': {'N': 0, 'Y': 1},
        'Propert_Owner': {'N': 0, 'Y': 1},
        'Type_Income': {'Working': 0, 'Commercial associate': 1, 'Pensioner': 2, 'State servant': 3, 'Student': 4},
        'EDUCATION': {'Higher education': 0, 'Secondary / secondary special': 1, 'Incomplete higher': 2, 'Lower secondary': 3, 'Academic degree': 4},
        'Marital_status': {'Married': 0, 'Single / not married': 1, 'Civil marriage': 2, 'Separated': 3, 'Widow': 4},
        'Housing_type': {'House / apartment': 0, 'With parents': 1, 'Municipal apartment': 2, 'Rented apartment': 3, 'Office apartment': 4, 'Co-op apartment': 5},
        'Type_Occupation': {'Laborers': 0, 'Core staff': 1, 'Accountants': 2, 'Managers': 3, 'Drivers': 4, 'Sales staff': 5}
    }

    for col, map_dict in mapping.items():
        if col in df.columns:
            df[col] = df[col].map(map_dict)

    return df[feature_names]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = {key: request.form[key] for key in feature_names}
        form_data = {k: float(v) if v.replace('.', '', 1).isdigit() else v for k, v in form_data.items()}
        input_df = preprocess_input(form_data)
        prediction = model.predict(input_df)[0]
        result = "🔴 Defaulted" if prediction == 1 else "🟢 Not Defaulted"
        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
