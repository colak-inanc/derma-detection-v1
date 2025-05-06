import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from md.predict_router import predict_diabetes, predict_disease, predict_cbc

diabetes_input = {
    'Age': 45,
    'Gender': 1,
    'BMI': 27.5,
    'Chol': 180,
    'TG': 140,
    'HDL': 50,
    'LDL': 100,
    'Cr': 1.0,
    'BUN': 15
}


disease_input = {
    'patient_id': 101,
    'age': 40,
    'gender': 'Male',
    'symptom_1': 'Headache',
    'symptom_2': 'Fever',
    'symptom_3': 'Fatigue',
    'heart_rate_bpm': 90,
    'body_temperature_c': 38.0,
    'blood_pressure_mmhg': 120,
    'oxygen_saturation_%': 96,
    'severity': 'Mild'
}


cbc_input = {
    'WBC': 6.1,
    'RBC': 4.7,
    'HGB': 13.5,
    'HCT': 41.0,
    'MCV': 88.0,
    'MCH': 30.0,
    'MCHC': 34.0,
    'PLT': 250
}

diabetes_result = predict_diabetes(diabetes_input)
disease_result = predict_disease(disease_input)
cbc_result = predict_cbc(cbc_input)

print("Diabetes Prediction:", diabetes_result)
print("Disease Prediction:", disease_result)
print("CBC Clustering Result (Cluster ID):", cbc_result)
