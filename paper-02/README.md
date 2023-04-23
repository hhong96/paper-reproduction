
# BD4H Paper Reproduction

This experiment was developed for Georgia Tech CSE-6250 group project.


## Paper

- Wang, Z., & Sun, J. (2022, August 7). Survtrace: Transformers for survival analysis with competing events. University of Illinois Urbana-Champaign.

## Dataset

METABRIC dataset was used for the experiment. The data set can be retrieved directly from PyCox module.
```
from pycox.datasets import metabric
df = metabric.read_df()
```


## Installation

- Clone the repository

    ``` git clone```


- Create a virtual environment
    
    ``` conda create -n bd4h python=3.7 && conda activate bd4h ```


- Install the dependencies

    ``` pip install -r requirements.txt ```


- Run the experiment

    ``` python run.py ```


## Description

- config.py : specification of the architecture and hyperparameters of the model.
- dataset.py : data preprocessing to train survival analysis model.
- train_utils.py : training SurvTRACE for survival analysis using PyTorch.
- evaluate_utils.py : evaluation of the model on the test data with Concordance index with IPCW and Brier score.
- model.py : PyTorch implementation of a single event survival analysis model
- modeling_bert.py : BERT transformer customized for processing time-to-event data.
- run.py : Entry point of the experiment. It runs the training and evaluation of the model.


## Dependencies

easydict==1.10
matplotlib==3.7.1
numpy==1.23.5
pandas==1.5.3
pycox==0.2.3
pytorch_pretrained_bert==0.6.2
scikit_learn==1.2.1
scikit_survival==0.0.0
torch==2.0.0

## Appendix

- Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.