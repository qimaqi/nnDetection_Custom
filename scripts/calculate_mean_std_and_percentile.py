import numpy as np
from multiprocessing import Pool
from functools import partial
import os 
from tqdm import tqdm


pretrain_data_path = '/usr/bmicnas02/data-biwi-01/lung_detection/pretrain_data/preprocessed_nlst'
preprocessed_npy_files = os.listdir(pretrain_data_path)
preprocessed_npy_files = [f for f in preprocessed_npy_files if f.endswith('.npy')]
preprocessed_npy_files = [os.path.join(pretrain_data_path, f) for f in preprocessed_npy_files]
# caluclate mean and std
# calculate percentile005 and percentile995

def calculate_data_info(data_path):
    data = np.load(data_path)
    mean = np.mean(data)
    std = np.std(data)
    percentile005 = np.percentile(data, 0.5)
    percentile995 = np.percentile(data, 99.5)
    return mean, std, percentile005, percentile995

# with Pool(10) as p:
#     data_info = p.map(partial(calculate_data_info, pretrain_data_path), preprocessed_npy_files)

with Pool(10) as p:
    data_info = list(tqdm(p.imap(partial(calculate_data_info), preprocessed_npy_files), total=len(preprocessed_npy_files)))

# average mean, std, percentile005, percentile995
mean = np.mean([data[0] for data in data_info])
std = np.mean([data[1] for data in data_info])
percentile005 = np.mean([data[2] for data in data_info])
percentile995 = np.mean([data[3] for data in data_info])

print(f"mean: {mean}, std: {std}, percentile005: {percentile005}, percentile995: {percentile995}")