<script src="https://cdn.rawgit.com/zenorocha/clipboard.js/v2.0.0/dist/clipboard.min.js"></script>
<style>
  .copy-btn {
    background: #333;
    border: none;
    color: white;
    cursor: pointer;
    padding: 5px 10px;
    position: absolute;
    right: 10px;
    top: 10px;
  }
  
  .copy-btn:hover {
    background: #555;
  }
</style>

# CAWANet_test
<img src="https://github.com/ZGWzzu/CAWANet/blob/main/docs/CAWANet.jpg" width = 800px>

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

```conda create -n env python==3.6```   


<div class="language-python highlighter-rouge">
  <pre class="highlight"><code><span class="k">def</span> <span class="nf">hello_world</span><span class="p">(</span><span class="n">name</span><span class="p">):</span>
    <span class="k">print</span><span class="p">(</span><span class="s">'Hello, '</span> <span class="o">+</span> <span class="n">name</span><span class="p">)</span>
</code></pre>
</div>
<button class="copy-btn" data-clipboard-target=".language-python">Copy</button>

# Usage
First download the dataset, and then put the dataset into /CAWANet/. Run python test_MSD.py
# Model weight

 Dataset | pth | mIoU | FPS 
 --- | --- | ---|---
 MSD | [CAWA_MSD](https://pan.baidu.com/s/13hZBmSwXGeX8R-f9IkYihw) 提取码：8bnv | 90.1 | 186.5
 NEU | [CAWA_NEU](https://pan.baidu.com/s/1xiCEGYb_Typ8E_55UExVYA) 提取码：hu1s | 77.9 | 420.0
 MT | [CAWA_MT](https://pan.baidu.com/s/1K3WAQsDaywEIRDtTOSjWUw) 提取码：3qtq| 79.5 | 414.7

# Results

<img src="https://github.com/ZGWzzu/CAWANet/blob/main/docs/MSD222.jpg" width = 800px>
<img src="https://github.com/ZGWzzu/CAWANet/blob/main/docs/neu111.jpg" width = 800px>
<img src="https://github.com/ZGWzzu/CAWANet/blob/main/docs/MT_SORT.jpg" width = 800px>
