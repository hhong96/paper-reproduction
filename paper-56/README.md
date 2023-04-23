# BD4H Paper Reproduction

This experiment was developed for Georgia Tech CSE-6250 group project.


## Paper

- Graph convolutional transformer: Learning the graphical structure of electronic health records
This repository contains the implementation of the Graph Convolutional Transformer (GCT) Model. GCT is a graph-based deep learning model that combines Graph Convolutional Networks (GCN) and Multi-Head Attention to perform classification tasks on graph-structured data.

## Dataset

 the Medical Information Mart for Intensive Care III (MIMIC-III) dataset was used. The dataset contains de-identified health data of patients admitted to the intensive care unit (ICU) at the Beth Israel Deaconess Medical Center between 2001 and 2012```

- Request access to the eICU dataset from eICU website.
- Note that you are required to participate in the CITI training.
- Download the patient, admissionDx, diagnosis, treatment CSV files.



## Installation

- Clone the repository

    ``` git clone```


- Create a virtual environment
    
    ``` conda create -n bd4h python=3.7 && conda activate bd4h ```


- Install the dependencies

    ``` pip install -r requirements.txt ```


- Run the experiment

    ``` python preprocess.py ```
    ``` python model.py ```
    ``` python train.py ```
    ``` python evaluation.py ```


## Description
- preprocess.py: This file contains the loading, transformation and merging of csv files. The output is a combined csv files with all the elements needed to build and train the model. 
- model.py: This file contains the implementation of the GCT model, including the Graph Convolutional layer and the Multi-Head Attention layer.

- train.py: This file is used to train the GCT model on the provided dataset. It loads the data, splits it into training and validation sets, and trains the model using the specified hyperparameters. After training, the model's weights are saved in the HDF5 format.

- evaluation.py: This file is used to evaluate the performance of the trained GCT model on the test dataset. It loads the saved model weights, performs predictions on the test data, and calculates various evaluation metrics such as accuracy, AUCPR, and AUROC.

## Dependencies

- numpy==1.21.2
- pandas==1.3.3
- scikit-learn==0.24.2
- torch==1.9.1
- tensorflow>=2.0.0,<2.7
- networkx
- h5py


## Appendix

## Implementation Details
Introduction:
Graph Convolutional Transformers (GCT) is a novel approach to learning the graphical structure of Electronic Health Records (EHRs) and effectively predicting patient outcomes. The GCT model combines Graph Convolutional Networks (GCNs) and Transformers to model complex, hierarchical relationships within EHR data. The model aims to capture both local and global information within the dataset and achieves competitive performance on several benchmark EHR tasks.

Scope of reproducibility:
In this reproducibility study, we aim to assess the performance of the GCT model by closely following the methodology presented in the original research. Our goal is to determine if the reported results can be reproduced using the provided code, data, and configurations. Additionally, we will explore the robustness of the model by testing its performance on different datasets and varying hyperparameters.

Methodology:
Model descriptions:
The GCT model is a combination of Graph Convolutional Networks (GCNs) and Transformers. The model leverages the strengths of both techniques: GCNs for local structure learning and Transformers for global context learning. It consists of three main components: the Graph Convolutional layer, the Transformer layer, and the readout layer. The Graph Convolutional layer learns the local information within the graph, while the Transformer layer captures the long-range dependencies. Finally, the readout layer aggregates the learned features to make the final prediction.

Data descriptions:
In the original GCT paper, the authors used the Medical Information Mart for Intensive Care III (MIMIC-III) dataset for their experiments. The dataset contains de-identified health data of patients admitted to the intensive care unit (ICU) at the Beth Israel Deaconess Medical Center between 2001 and 2012. The data includes patient demographics, diagnoses, procedures, treatments, and laboratory test results, among other information.

Computational implementation:
The GCT model was implemented using the TensorFlow deep learning framework. The code provided in the GitHub repository includes data preprocessing, model training, and evaluation scripts. The authors also provided the necessary hyperparameters for training the model and achieving competitive performance on the benchmark tasks.

File Explanation: 
The preprocess.py is used to preprocess the eicu data and convert it into a format that is available for analysis. It reads in the csv files in cluding patient, admissiondx, diagnosis and treatment, then convert and merge them into a single preprocessed data file. 
Then model.py would load the single preprocessed data file, and process them into graph, feature and adjacency matrices, that are used for the GCT model. It also creates the GCT model with layers and sets it up for training.   
Train.py file would then build and train the model with a series of hyperparameters. It would also do things like load in the matrices and convert string labels to integers. It goes through a number of training and validation and produce the average validation AUCPR/AUCROC graphs and visualize the training and validation losses as well. 
Evaluation.py file would then run a similar process like in train.py, but on test data and produce the test metrics like test accuracy, AUCPR and AUROC. 
