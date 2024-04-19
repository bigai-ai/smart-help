import torch
from torch.utils.data import Dataset, DataLoader
import h5py

from tqdm import tqdm
import numpy as np

from copy import deepcopy
import sys
sys.path.append("/home/zhihao/A2SP/rllib_A2SP")


with h5py.File('trajectories_early_end_image.h5', 'r+') as f:
    # 获取对应索引的数据和标签
    data = f['data']
    image = f['image']

    data_new = data[:10000]
    image_new = image[:10000]

    # new_data = data[:10000]
    # new_image = image[:10000]

    # data.resize(new_data.shape)

    data_len = len(data)
    print(data_len)
    # with tqdm(total=data_len, desc="data_num") as pbar:
    #     for i in range(data_len):
    #         for j in range(40):
    #             # print(data[i, j])
    #             # print(data[i][j])
    #             # print(data[i][j][0][0])
    #             if np.sum(data[i, j, 0, 0]) == 0:
    #                 # print("empty")
    #                 # print(data[i, j])
    #                 # print("i, j-1", data[i, j-1])
    #                 # print("i, j-2", data[i, j-3])
    #                 data[i, j] = data[i, j-1]
    #                 image[i, j] = image[i, j-1]
    #                 # print(data[i, j])
    #         pbar.update(1)

