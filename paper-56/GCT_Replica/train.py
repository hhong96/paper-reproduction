import tensorflow as tf
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from model import GraphConvolution, MultiHeadAttention, GCTModel
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

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
train_indices = valid_indices[:int(0.8 * len(valid_indices))]
val_indices = valid_indices[int(0.8 * len(valid_indices)):]

train_nodes = [index_to_id_map[idx] for idx in train_indices]
val_nodes = [index_to_id_map[idx] for idx in val_indices]

train_G = G.subgraph(train_nodes)
val_G = G.subgraph(val_nodes)

train_adj_matrix = nx.adjacency_matrix(train_G).astype(np.float32).toarray()
val_adj_matrix = nx.adjacency_matrix(val_G).astype(np.float32).toarray()

train_features = feature_matrix[train_indices].astype(np.float32)
train_labels = np.array([node_to_label_map[node] for node in train_nodes])

val_features = feature_matrix[val_indices].astype(np.float32)
val_labels = np.array([node_to_label_map[node] for node in val_nodes])
val_features = feature_matrix[val_indices].astype(np.float32)
#val_adj_matrix = adj_matrix[val_indices]

# Hyperparameters
num_classes = len(np.unique(int_labels))
gcn_units = 64
attn_heads = 8
attn_head_dim = 64
learning_rate = 0.001
epochs = 10
batch_size = 32

# Build the GCT model
model = GCTModel(num_classes, gcn_units, attn_heads, attn_head_dim)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
# Train the model
# Train the model
val_AUCPRs = []
val_AUROCs=[]
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print("Train features shape:", train_features.shape)
    print("Train adjacency matrix shape:", train_adj_matrix.shape)
    print("Train labels shape:", train_labels.shape)

    # Create a mapping between the original graph indices and the subgraph indices for train_G
    train_nodes_to_subgraph_idx = {node: idx for idx, node in enumerate(train_G.nodes)}

    # Training
    train_dataset = tf.data.Dataset.from_tensor_slices((train_indices, train_labels)).shuffle(len(train_indices)).batch(batch_size)
    for step, (batch_indices, labels) in enumerate(train_dataset):
        #print("batch indice:", batch_indices)
        #print("feature matrix shape",feature_matrix.shape)

        # Add this check
        features = feature_matrix[batch_indices.numpy()]


        print("Features array shape:", features.shape)
        train_losses = []

        with tf.GradientTape() as tape:
            # Convert batch_indices from the original graph to subgraph indices for train_G
            batch_indices_np = batch_indices.numpy()
            subgraph_batch_indices = [train_nodes_to_subgraph_idx[index_to_id_map[idx]] for idx in batch_indices_np]
            batch_adj_matrix = train_adj_matrix[np.ix_(subgraph_batch_indices, subgraph_batch_indices)]
            predictions = model(features, batch_adj_matrix)

            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            train_loss = tf.reduce_mean(loss).numpy()
            # Append the loss value for the current batch to the train_losses list
            train_losses.append(train_loss)
            loss = tf.reduce_mean(loss)
            

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc = np.mean(np.argmax(predictions, axis=1) == labels)
        #print(f"Step {step + 1} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")
        print(f"Step {step + 1} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")

    # Validation
    val_batch_size = 64
    val_indices = np.random.choice(len(val_features), size=val_batch_size, replace=False)
    val_batch_features = val_features[val_indices]
    val_batch_adj_matrix = val_adj_matrix[np.ix_(val_indices, val_indices)]
    val_predictions = model(val_batch_features, val_batch_adj_matrix)
    val_losses = []
    val_loss = tf.keras.losses.sparse_categorical_crossentropy(val_labels[val_indices], val_predictions)
    val_loss = tf.reduce_mean(val_loss).numpy()
    val_losses.append(val_loss)

    val_acc = np.mean(np.argmax(val_predictions, axis=1) == val_labels[val_indices])
    #print(f"Validation - Loss: {val_loss:.4f} - Accuracy: {val_acc:.4f}")
    print("epoch:",epoch)
    print(f"Validation  - Accuracy: {val_acc:.4f}")

    # Get the predicted labels for the validation set
    val_pred_labels = np.argmax(val_predictions, axis=1)

    # Calculate the validation accuracy
    val_acc = metrics.accuracy_score(val_labels[val_indices], val_pred_labels)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Calculate the validation AUCPR
    val_aucpr = metrics.average_precision_score(val_labels[val_indices], val_pred_labels)
    val_AUCPRs.append(val_aucpr)
    print(f"Validation AUCPR: {val_aucpr:.4f}")

    # Calculate the validation AUROC
    val_auroc = metrics.roc_auc_score(val_labels[val_indices], val_pred_labels)
    val_AUROCs.append(val_auroc)
    print(f"Validation AUROC: {val_auroc:.4f}")


print(f"average Validation AUCPR: {np.mean(val_AUCPRs):.4f}")
print(f"average Validation AUROC: {np.mean(val_AUROCs):.4f}")
# define epochs and losses
num_epochs = range(1, epochs+1)
train_losses = [train_loss] * epochs
val_losses = [val_loss]*epochs
# plot the losses
plt.plot(num_epochs, train_losses, 'bo', label='Training loss')
plt.plot(num_epochs, val_losses, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


num_epochs = range(1, epochs+1)
train_losses = [val_AUROCs] * epochs
val_losses = [val_AUCPRs]*epochs
# plot the losses
plt.plot(num_epochs, train_losses, 'bo', label='Training loss')
plt.plot(num_epochs, val_losses, 'b', label='Validation loss')
plt.title('val_AUROCs and val_AUCPRs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()