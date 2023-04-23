import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import numpy as np
import os
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pickle
DATA_DIR = 'data'
PREPROCESSED_DATA_FILE = 'preprocessed_data.csv'

'''
class GraphConvolution(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='glorot_uniform',
                                      name='kernel')

    def call(self, inputs, adj_matrix):
        output = tf.matmul(tf.matmul(adj_matrix, inputs), self.kernel)
        return self.activation(output)
'''

class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, num_units, activation=tf.nn.relu):
        super(GraphConvolution, self).__init__()
        self.num_units = num_units
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.num_units),
                                       initializer='glorot_uniform',
                                       name='kernel')
        self.bias = self.add_weight(shape=(self.num_units,),
                                    initializer='zeros',
                                    name='bias')
    def call(self, inputs, adj_matrix):
        inputs = tf.cast(inputs, tf.float32)
        output = tf.matmul(adj_matrix, inputs)
        output = tf.matmul(output, self.kernel)
        return output


class SimplifiedGCT(Model):
    def __init__(self, num_classes, **kwargs):
        super(SimplifiedGCT, self).__init__(**kwargs)
        self.graph_conv1 = GraphConvolution(64, activation='relu')
        self.graph_conv2 = GraphConvolution(32, activation='relu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, adj_matrix):
        x = self.graph_conv1(inputs, adj_matrix)
        x = self.graph_conv2(x, adj_matrix)
        x = self.flatten(x)
        output = self.dense(x)
        return output
import tensorflow as tf
from tensorflow.keras import layers



class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, head_dim, activation=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.activation = activation

    def build(self, input_shape):
        self.Wq = self.add_weight(shape=(input_shape[-1], self.num_heads * self.head_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Wk = self.add_weight(shape=(input_shape[-1], self.num_heads * self.head_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Wv = self.add_weight(shape=(input_shape[-1], self.num_heads * self.head_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)

    def call(self, inputs):
        q = tf.matmul(inputs, self.Wq)
        k = tf.matmul(inputs, self.Wk)
        v = tf.matmul(inputs, self.Wv)

        q = tf.stack(tf.split(q, self.num_heads, axis=-1), axis=1)
        k = tf.stack(tf.split(k, self.num_heads, axis=-1), axis=1)
        v = tf.stack(tf.split(v, self.num_heads, axis=-1), axis=1)

        attn_weights = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(self.head_dim)))
        output = tf.matmul(attn_weights, v)
        output = tf.concat(tf.unstack(output, axis=1), axis=-1)

        if self.activation:
            output = self.activation(output)
        return output


class GCTModel(tf.keras.Model):
    def __init__(self, num_classes, gcn_units, attn_heads, attn_head_dim):
        super(GCTModel, self).__init__()
        self.graph_conv = GraphConvolution(gcn_units)
        self.multi_head_attention = MultiHeadAttention(attn_heads, attn_head_dim)
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, adj_matrix):
        x = self.graph_conv(inputs, adj_matrix)
        x = self.multi_head_attention(x)
        x = tf.keras.layers.Flatten()(x) # Use this line instead of self.flatten(x)
        x = self.fc(x)
        return x
    
    @tf.function
    def call_with_default_adj(self, inputs):
        return self.call(inputs, self.default_adj_matrix)


def load_preprocessed_data():
    data = pd.read_csv(os.path.join(DATA_DIR, PREPROCESSED_DATA_FILE))
    data['diagnosisstring'].fillna('', inplace=True)

    return data

def create_graph(data):
    G = nx.Graph()
    for index, row in data.iterrows():
        patient_id = row['patientunitstayid']
        diagnosis = row['diagnosisstring']
        treatment = row['treatmentstring']
        unitdischargestatus = row['unitdischargestatus']

        # Add the index, diagnosisstring, and unitdischargestatus attributes to the node
        G.add_node(patient_id, diagnosisstring=diagnosis, treatment=treatment, index=index, unitdischargestatus=unitdischargestatus)
        
        # Add edges between patients with similar diagnoses
        for other_patient_id, other_row in data[data['diagnosisstring'] == diagnosis].iterrows():
            if patient_id != other_patient_id:
                G.add_edge(patient_id, other_patient_id)
        # Add edges between patients with similar treatments
        for other_patient_id, other_row in data[data['treatmentstring'] == treatment].iterrows():
            if patient_id != other_patient_id:
                G.add_edge(patient_id, other_patient_id)

    return G


# Function to create feature and adjacency matrices
def create_matrices(data):
    G = create_graph(data)
    
    # Create the feature matrix
    diagnoses = [data.get('diagnosisstring', 'default_value') for _, data in G.nodes(data=True)]
    vectorizer = CountVectorizer()
    feature_matrix = vectorizer.fit_transform(diagnoses).toarray()

    # Create the adjacency matrix
    adj_matrix = nx.adjacency_matrix(G)
    adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])

    # Create labels
    label_mapping = {'Alive': 0, 'Expired': 1}
    valid_nodes = [node for node in G.nodes if G.nodes[node].get('unitdischargestatus') in label_mapping]
    labels = np.array([label_mapping[G.nodes[node]['unitdischargestatus']] for node in valid_nodes])

    num_nodes_with_status = sum(1 for node in G.nodes if G.nodes[node].get('unitdischargestatus') is not None)
    print("Number of nodes with unitdischargestatus:", num_nodes_with_status)



    np.save("feature_matrix.npy", feature_matrix)
    np.save("adj_matrix.npy", adj_matrix)
    np.save("labels.npy", labels)
    with open("graph.gpickle", "wb") as f:
        pickle.dump(G, f)
    return feature_matrix, adj_matrix,labels,G

# Modify the main function
def main():
    # Load preprocessed data from process_eicu.py
    data = load_preprocessed_data()

    # Create the feature and adjacency matrices
    feature_matrix, adj_matrix,labels,graph = create_matrices(data)

    # Create the model
    num_classes = 1 # Set the number of classes
    model = SimplifiedGCT(num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model with your preprocessed data
    # model.fit(feature_matrix, adj_matrix, ...)

if __name__ == '__main__':
    main()

