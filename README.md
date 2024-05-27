# CAWANet

<img src="https://github.com/ZGWzzu/CAWANet/blob/main/docs/CAWANet_.jpg" width = 800px>


# Dataset
We used data sets adopted by FDSNet: [MSD,NEU-Seg, MSD.](https://github.com/jianzhang96/fdsnet?tab=readme-ov-file)

# Environment
The training was on an NVIDIA 3090 GPU, and the FPS was tested on an NVIDIA 1080Ti GPU   
Python 3.8.18 PyTorch 2.1.2 CUDA 11.8  
   
einops==0.7.0  
numpy==1.24.4  
Pillow==10.0.1  
torch==2.1.2  
torchvision==0.16.2  

You can create an environment by typing a command：  

```
conda create -n CAWANet python==3.8.18
activate CAWANet
```

Download the required package:  
```
pip install -r requirements.txt
```

# Usage
First download the dataset, and then put the dataset into /CAWANet/. Run python test_MSD.py
# Model weight

 Dataset | pth | mIoU | FPS 
 --- | --- | ---|---
 MSD | [CAWA_MSD](https://pan.baidu.com/s/13hZBmSwXGeX8R-f9IkYihw) Extract code：8bnv | 90.1 | 186.5
 NEU | [CAWA_NEU](https://pan.baidu.com/s/1xiCEGYb_Typ8E_55UExVYA) Extract code：hu1s | 77.9 | 420.0
 MT | [CAWA_MT](https://pan.baidu.com/s/1K3WAQsDaywEIRDtTOSjWUw) Extract code：3qtq| 79.5 | 414.7

# Results

<img src="https://github.com/ZGWzzu/CAWANet/blob/main/docs/MSD222.jpg" width = 800px>
<img src="https://github.com/ZGWzzu/CAWANet/blob/main/docs/neu111.jpg" width = 800px>
<img src="https://github.com/ZGWzzu/CAWANet/blob/main/docs/MT_SORT.jpg" width = 800px>
