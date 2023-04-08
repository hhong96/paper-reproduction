import torch
import numpy as np
import pickle
import networkx as nx


patient_df = pd.read_csv('patient.csv')
admissiondx_df = pd.read_csv('admissionDx.csv')
diagnosis_df = pd.read_csv('diagnosis.csv')
treatment_df = pd.read_csv('treatment.csv')

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def create_adj_matrix(data, patient_df, admissiondx_df, diagnosis_df, treatment_df):
    G = nx.Graph()

    # Add patients as nodes
    for _, row in patient_df.iterrows():
        patient_id = row['patientunitstayid']
        G.add_node(patient_id, node_type='patient')

    # Add admission diagnoses as nodes and connect them to patients
    for _, row in admissiondx_df.iterrows():
        patient_id = row['patientunitstayid']
        admissiondx_id = row['admissiondxid']
        G.add_node(admissiondx_id, node_type='admissiondx')
        G.add_edge(patient_id, admissiondx_id)

    # Add diagnoses as nodes and connect them to patients
    for _, row in diagnosis_df.iterrows():
        patient_id = row['patientunitstayid']
        diagnosis_id = row['diagnosisid']
        G.add_node(diagnosis_id, node_type='diagnosis')
        G.add_edge(patient_id, diagnosis_id)

    # Add treatments as nodes and connect them to patients
    for _, row in treatment_df.iterrows():
        patient_id = row['patientunitstayid']
        treatment_id = row['treatmentid']
        G.add_node(treatment_id, node_type='treatment')
        G.add_edge(patient_id, treatment_id)

    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(G)

    # Convert the adjacency matrix to a dense numpy array
    adj_matrix = adj_matrix.toarray()

    return adj_matrix

'''train_adj_matrix = create_adj_matrix(train_data, patient_df, admissiondx_df, diagnosis_df, treatment_df)
val_adj_matrix = create_adj_matrix(val_data, patient_df, admissiondx_df, diagnosis_df, treatment_df)
test_adj_matrix = create_adj_matrix(test_data, patient_df, admissiondx_df, diagnosis_df, treatment_df)
'''

def accuracy(output, target):
    pred = output.argmax(dim=1)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / target.size(0)

def create_dataloader(data, adj_matrix, labels, batch_size):
    dataset = TensorDataset(torch.tensor(data), torch.tensor(adj_matrix), torch.tensor(labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


