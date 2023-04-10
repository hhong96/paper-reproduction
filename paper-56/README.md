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