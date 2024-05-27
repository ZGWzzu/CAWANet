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
We divided the test code for the three data sets into test_NEU.py, test_MSD.py, and test_MT.py. We set the parameters for each data set in the corresponding py file, just download the corresponding data set, and then put the data set in the /CAWANet/ directory. Then just run python test_MSD.py
We divided the test code for the three data sets into test_NEU.py, test_MSD.py, and test_MT.py. We set the parameters for each data set in the corresponding py file and just do the following to run it:  
1. Download the appropriate data set and place the data set in the /CAWANet/ directory.  
2. Download the weight files for the corresponding data set, and then place the weight files in the /CAWANet/ directory.  
3. Run python test_x.py  
# Model weight

 Dataset | Baidu Cloud/pth |Google Drive/pth | mIoU | FPS 
 --- | --- | ---|---|---
 MSD | [CAWA_MSD](https://pan.baidu.com/s/13hZBmSwXGeX8R-f9IkYihw) Extract code：8bnv |[CAWA_MSD](https://drive.google.com/file/d/1J593V_P8v2ZrzDAIelYDpNCL4exOibDc/view?usp=drive_link)| 90.1 | 186.5
 NEU | [CAWA_NEU](https://pan.baidu.com/s/1xiCEGYb_Typ8E_55UExVYA) Extract code：hu1s | [CAWA_NEU](https://drive.google.com/file/d/1qrMWbcMR1D69JAnFUgbz40GeXORsktD7/view?usp=drive_link)|77.9 | 420.0
 MT | [CAWA_MT](https://pan.baidu.com/s/1K3WAQsDaywEIRDtTOSjWUw) Extract code：3qtq|[CAWA_MT](https://drive.google.com/file/d/1J593V_P8v2ZrzDAIelYDpNCL4exOibDc/view?usp=drive_link)| 79.5 | 414.7

# Results

<img src="https://github.com/ZGWzzu/CAWANet/blob/main/docs/MSD222.jpg" width = 800px>
<img src="https://github.com/ZGWzzu/CAWANet/blob/main/docs/neu111.jpg" width = 800px>
<img src="https://github.com/ZGWzzu/CAWANet/blob/main/docs/MT_SORT.jpg" width = 800px>
