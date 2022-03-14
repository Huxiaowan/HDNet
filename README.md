# The source code of the ID2182 paper: HDNet: High-resolution Dual-domain Learning  for Spectral Compressive Imaging

## Requirement
Python=3.5+
PyTorch=1.0+
gcc (GCC) 4.8.5  
CUDA 8.0  
OS: Ubuntu 16.04
CUDA: 9.0/10.0
pillow 
matplotlib 


## Prepare training data 
> Simulation Data
1. Download HSI training data
2. Specify the data path in train_HDNet.py

    ```python
    data_path = "/data/train_data/"
    mask_path = "/data/simu_data/"
    test_path = "/data/simu_data/Truth/" 
    ```


## Train
### Training simulation model
1) Put hyperspectral datasets (Ground truth) into corrsponding path. For our setting, the training data and validation datashould be scaled to 0-65535 and 0-1, respectively, with a size of 256×256×28.  
2) Run **train_HDNet.py**.
### Training real data model  
1) Put hyperspectral datasets (Ground truth) into corrsponding path. For our setting, the training data and validation datashould be scaled to 0-1 with a size of 660×660×28.  
2) Run **train_HDNet.py**.

