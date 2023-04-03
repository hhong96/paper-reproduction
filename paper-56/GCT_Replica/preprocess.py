import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.model_selection import train_test_split

def read_data(filepath):
    data = pd.read_csv(filepath)
    return data

def generate_features(data):
    # Generate input features
    pass

def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    return train_data, val_data, test_data

def save_preprocessed_data(output_dir, train_data, val_data, test_data):
    with open(output_dir + '/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(output_dir + '/val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open(output_dir + '/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    data = read_data(args.data)
    features = generate_features(data)
    train_data, val_data, test_data = split_data(features)
    save_preprocessed_data(args.output_dir, train_data, val_data, test_data)
