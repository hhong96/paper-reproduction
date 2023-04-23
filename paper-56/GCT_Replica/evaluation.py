import tensorflow as tf
import numpy as np
import networkx as nx
import pickle
import sklearn.metrics as metrics
from model import GraphConvolution, MultiHeadAttention, GCTModel

# Load the data
feature_matrix = np.load("feature_matrix.npy")
adj_matrix = np.load("adj_matrix.npy")
labels = np.load("labels.npy", allow_pickle=True)

# Convert string labels to integers
unique_labels = np.unique(labels)
label_to_int = {label: i for i, label in enumerate(unique_labels)}
int_labels = np.array([label_to_int[label] for label in labels])

with open("graph.gpickle", "rb") as f:
    G = pickle.load(f)

indices = np.arange(feature_matrix.shape[0])
np.random.shuffle(indices)

id_to_index_map = {node: idx for idx, node in enumerate(G.nodes)}
index_to_id_map = {idx: node for idx, node in enumerate(G.nodes)}

node_to_label_map = {index_to_id_map[index]: label for index, label in zip(range(len(int_labels)), int_labels)}

valid_indices = [idx for idx in indices if index_to_id_map[idx] in node_to_label_map]

# Split the dataset
train_indices = valid_indices[:int(0.8 * len(valid_indices))]
val_indices = valid_indices[int(0.8 * len(valid_indices)):int(0.9 * len(valid_indices))]
test_indices = valid_indices[int(0.9 * len(valid_indices)):]

test_nodes = [index_to_id_map[idx] for idx in test_indices]
test_G = G.subgraph(test_nodes)
test_adj_matrix = nx.adjacency_matrix(test_G).astype(np.float32).toarray()
test_features = feature_matrix[test_indices].astype(np.float32)
test_labels = np.array([node_to_label_map[node] for node in test_nodes])

# Load the saved model
num_classes = len(np.unique(int_labels))
gcn_units = 64
attn_heads = 8
attn_head_dim = 64

model = GCTModel(num_classes, gcn_units, attn_heads, attn_head_dim)
# Create dummy inputs for model initialization
dummy_features = np.zeros((1, feature_matrix.shape[1]), dtype=np.float32)
dummy_adj_matrix = np.eye(1, dtype=np.float32)

# Call the model with dummy inputs to create variables
_ = model(dummy_features, dummy_adj_matrix)

model.load_weights('trained_gct_model_weights.h5')

# Evaluate the model on test data
test_predictions = model(test_features, test_adj_matrix)

# Get the predicted labels for the test set
test_pred_labels = np.argmax(test_predictions, axis=1)

# Calculate the test accuracy
test_acc = metrics.accuracy_score(test_labels, test_pred_labels)
print(f"Test Accuracy: {test_acc:.4f}")

# Calculate the test AUCPR
test_aucpr = metrics.average_precision_score(test_labels, test_pred_labels)
print(f"Test AUCPR: {test_aucpr:.4f}")

# Calculate the test AUROC
test_auroc = metrics.roc_auc_score(test_labels, test_pred_labels)
print(f"Test AUROC: {test_auroc:.4f}")
