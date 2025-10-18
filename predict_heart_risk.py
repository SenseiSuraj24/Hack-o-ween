import pandas as pd
import numpy as np
import pickle
import json

# --- Load the saved components ---
try:
    with open('heart_disease_model.pkl', 'rb') as f:
        MODEL = pickle.load(f)
    with open('mean_imputer.pkl', 'rb') as f:
        MEAN_IMPUTER = pickle.load(f)
    with open('mode_imputer.pkl', 'rb') as f:
        MODE_IMPUTER = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        SCALER = pickle.load(f)
    with open('model_features.pkl', 'rb') as f:
        MODEL_FEATURES = pickle.load(f)
except FileNotFoundError:
    print("Error: Required model files are missing. Ensure all .pkl files are in the same directory.")
    # Exit or raise an error in a real application
    MODEL = None
    
# Define the expected feature columns based on your training data
# This list MUST match the columns used during training
# Note: 'LengthOfStay', 'Medical Condition', 'random_notes', 'noise_col' are excluded.
# 'Gender' is replaced by 'Gender_Male' (or whatever the saved feature name is)
NUMERICAL_COLS = [
    'Age', 'Glucose', 'Blood Pressure', 'BMI', 'Oxygen Saturation',
    'Cholesterol', 'Triglycerides', 'HbA1c', 'Physical Activity', 
    'Diet Score', 'Stress Level', 'Sleep Hours'
]
CATEGORICAL_COLS = ['Gender'] # Only 'Gender' was categorical
ENCODED_GENDER_COL = 'Gender_Male'

def predict_heart_risk(patient_data: dict) -> dict:
    """
    Predicts the heart issue risk for a single patient.

    Args:
        patient_data (dict): A dictionary of patient features.
                             Example: {'Age': 55, 'Gender': 'Female', 
                                       'Glucose': 120.5, 'Blood Pressure': 140.0, ...}

    Returns:
        dict: A dictionary containing the prediction (0 or 1) and the probability.
    """
    if MODEL is None:
        return {"error": "Model not loaded.", "prediction": -1, "probability": 0.0}

    # 1. Convert dictionary to DataFrame
    df_new = pd.DataFrame([patient_data])

    # 2. Impute missing values (even if they are NaN in the input)
    # Note: Use the fitted imputers/scalers, NOT refitting them.
    df_new[NUMERICAL_COLS] = MEAN_IMPUTER.transform(df_new[NUMERICAL_COLS])
    df_new[CATEGORICAL_COLS] = MODE_IMPUTER.transform(df_new[CATEGORICAL_COLS])

    # 3. One-Hot Encode 'Gender'
    # Create the column first and set its value based on the Gender input
    df_new[ENCODED_GENDER_COL] = np.where(df_new['Gender'] == 'Male', 1, 0)
    
    # Drop the original 'Gender' column and other one-hot columns (e.g. 'Gender_Female' which is implicitly 0)
    df_new = df_new.drop(columns=CATEGORICAL_COLS) 

    # 4. Scale Numerical Features
    df_new[NUMERICAL_COLS] = SCALER.transform(df_new[NUMERICAL_COLS])

    # 5. Reorder and Select Features
    # This step is CRITICAL: the feature columns and their order must exactly match the training data
    df_final = df_new[MODEL_FEATURES]

    # 6. Make Prediction
    # Prediction: 0 (No Issue) or 1 (Issue, based on Hypertension)
    prediction = MODEL.predict(df_final)[0]
    
    # Prediction probability (confidence score for class 1)
    probability = MODEL.predict_proba(df_final)[:, 1][0]

    return {
        "prediction": int(prediction),
        "probability_of_heart_issue": round(probability, 4)
    }

# --- Example of Usage ---
# You would get this data from a web form or API call
new_patient_data = {
    'Age': 60, 
    'Gender': 'Male', 
    'Glucose': 150.0, 
    'Blood Pressure': 165.0, # High BP
    'BMI': 35.0, 
    'Oxygen Saturation': 95.0, 
    'Cholesterol': 280.0, 
    'Triglycerides': 220.0, 
    'HbA1c': 8.5, 
    'Smoking': 1, # Smoker
    'Alcohol': 0, 
    'Physical Activity': 2.5, 
    'Diet Score': 3.0, 
    'Family History': 1, 
    'Stress Level': 7.0, 
    'Sleep Hours': 5.0
}

# The patient is missing an 'Age' reading:
patient_with_missing_data = {
    'Age': np.nan, # Missing value will be imputed with the mean from the training data
    'Gender': 'Female', 
    'Glucose': 90.0, 
    'Blood Pressure': 120.0, 
    'BMI': 25.0, 
    'Oxygen Saturation': 98.0, 
    'Cholesterol': 180.0, 
    'Triglycerides': 100.0, 
    'HbA1c': 5.5, 
    'Smoking': 0, 
    'Alcohol': 0, 
    'Physical Activity': 8.0, 
    'Diet Score': 7.0, 
    'Family History': 0, 
    'Stress Level': 2.0, 
    'Sleep Hours': 8.0
}

# To see the output, you can run:
# print(json.dumps(predict_heart_risk(new_patient_data), indent=4))
# print(json.dumps(predict_heart_risk(patient_with_missing_data), indent=4))