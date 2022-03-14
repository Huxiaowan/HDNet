# HDNet: High-resolution Dual-domain Learning  for Spectral Compressive Imaging
The source code of the CVPR2022 paper: **HDNet: High-resolution Dual-domain Learning  for Spectral Compressive Imaging**

## Architecture
![architecture](https://github.com/Huxiaowan/HDNet/blob/main/figures/architecture.png)
The architecture of HDNet. Spatial-spectral domain learning (SDL) includes HR spectral attention, HR spatial attention, and efficient feature fusion (EFF). In frequency domain learning (FDL), the 2D Discrete Fourier Transform (DFT) is used to obtain the HSI frequency spectrum. The adaptive weight θ(u,v) of each frequency coordinate (u,v) is dynamically determined by the frequency distance.

## Result
![table](https://github.com/Huxiaowan/HDNet/blob/main/figures/table_result.png)
The PSNR in dB (left entry in each cell) and SSIM (right entry in each cell) results of the test methods on 10 scenes.

![visual](https://github.com/Huxiaowan/HDNet/blob/main/figures/visiual_result.png)
Simulated HSI reconstruction comparisons of Scene 7 with 4 (out of 28) spectral channels. We show the spectral curves (topmedium) corresponding to the selected green boxes of the RGB image. Our HDNet reconstructs more visually pleasant detailed contents.

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

