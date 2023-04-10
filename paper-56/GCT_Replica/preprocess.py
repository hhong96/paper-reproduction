import os
import sys
import pandas as pd
import numpy as np
import pickle

# Set data directory
DATA_DIR = 'data'

def load_data_files():
    patient = pd.read_csv(os.path.join(DATA_DIR, 'patient.csv'))
    admissiondx = pd.read_csv(os.path.join(DATA_DIR, 'admissiondx.csv'))
    diagnosis = pd.read_csv(os.path.join(DATA_DIR, 'diagnosis.csv'))
    treatment = pd.read_csv(os.path.join(DATA_DIR, 'treatment.csv'))
    return patient, admissiondx, diagnosis, treatment

def preprocess_patient_data(patient):
    patient = patient[['patientunitstayid', 'gender', 'age', 'ethnicity', 'unittype', 'unitadmitsource', 'unitvisitnumber','patienthealthsystemstayid','unitdischargestatus']]
    return patient

def preprocess_admissiondx_data(admissiondx):
    admissiondx = admissiondx[['patientunitstayid', 'admitdxpath']]
    return admissiondx

def preprocess_diagnosis_data(diagnosis):
    diagnosis = diagnosis[['patientunitstayid', 'diagnosisstring','diagnosisid','icd9code']]
    return diagnosis

def preprocess_treatment_data(treatment):
    treatment = treatment[['patientunitstayid', 'treatmentstring','treatmentid']]
    return treatment

def main():
    # Load CSV data
    patient, admissiondx, diagnosis, treatment = load_data_files()

    # Preprocess data
    patient = preprocess_patient_data(patient)
    admissiondx = preprocess_admissiondx_data(admissiondx)
    diagnosis = preprocess_diagnosis_data(diagnosis)
    treatment = preprocess_treatment_data(treatment)

    # Merge data
    patient_admissiondx = pd.merge(patient, admissiondx, on='patientunitstayid', how='left')
    patient_admissiondx_diagnosis = pd.merge(patient_admissiondx, diagnosis, on='patientunitstayid', how='left')
    sample_fraction = 0.001  # Change this value to control the fraction of data you want to keep
    patient_admissiondx_diagnosis_sample = patient_admissiondx_diagnosis.sample(frac=sample_fraction)
    treatment_sample = treatment.sample(frac=sample_fraction)

    patient_admissiondx_diagnosis_treatment = pd.merge(patient_admissiondx_diagnosis_sample, treatment_sample, on='patientunitstayid', how='left')

    # Save preprocessed data
    patient_admissiondx_diagnosis_treatment.to_csv(os.path.join(DATA_DIR, 'preprocessed_data.csv'), index=False)

    print("Preprocessed data saved to:", os.path.join(DATA_DIR, 'preprocessed_data.csv'))

if __name__ == '__main__':
    main()

