import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import torch.utils.data
from tqdm import tqdm
from bisenetv1 import BiSeNetV1

model = BiSeNetV1(4).to('cuda')

model.eval()
example_inputs = torch.randn((1, 3,512,512)).to('cuda')

for i in tqdm(range(200), desc='warmup....'):
    model(example_inputs)

time_arr = []
start_time = time.time()
for i in tqdm(range(1000), desc='test latency....'):

    model(example_inputs)

end_time = time.time()
time_arr.append(end_time - start_time)

std_time = np.std(time_arr)
infer_time_per_image = np.sum(time_arr) / (1000 * 1)
fps = 1 / infer_time_per_image
print(f'fps:{1 / infer_time_per_image:.1f}')
print(fps)