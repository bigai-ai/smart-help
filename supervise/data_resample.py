import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.append("/home/zhihao/文档/GitHub/rllib_A2SP")

from model.agent_encoder_without_type_v2 import AgentEncoderWithoutType
from model.object_encoder_v3 import ObjectEncoder
from model.opponent_modeling_v3 import OpponentModeling
import numpy as np

from env.constants import AGENT_TYPE_NUMBER, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN
from env.subtask import SUBTASK_NUM

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
import h5py
from tqdm import tqdm

import glob
import json
from env.subtask import subtask_list
from env.goal import goal_list
from action import action_list
import pandas as pd
import tensorflow as tf

class TrajectoryDataset(Dataset):
    def __init__(self, file_path, train=True, train_ratio=0.9):
        self.file_path = file_path
        self.files = glob.glob(os.path.join(file_path, "*"))
        self.files.sort()
        if train:
            self.files = self.files[:int(len(self.files) * train_ratio)]
        else:
            self.files = self.files[int(len(self.files) * train_ratio):]

    def __getitem__(self, index):
        with h5py.File(self.files[index], 'r') as f:
            # 获取对应索引的数据和标签
            sequence = f['data'][:]
            image = f['image'][:]
            return torch.from_numpy(sequence), torch.from_numpy(image)

    def __len__(self):
        # 数据集的长度为'matrices'数据集的长度
        return len(self.files)


# 创建数据集
dataset_train = TrajectoryDataset('/home/zhihao/文档/GitHub/rllib_A2SP/dataset_new', train_ratio=1)
dataset_test = TrajectoryDataset('/home/zhihao/文档/GitHub/rllib_A2SP/dataset_new', train=False)

batch_size = 1

# 创建数据加载器
dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

epoch_num = 1000
batch_num = int(len(dataset_train) / batch_size)

# JSON 会将int key 转变为 str
tmp_dict = json.load(open("./object_index2name.json", "r"))
object_index2name = {}
for key, value in tmp_dict.items():
    object_index2name[int(key)] = value

window_size = 5

count_dict = {}

with tqdm(total=len(dataloader), desc="Processing items") as pbar:
    for batch_data, batch_image in dataloader:
        # 创建一个h5文件并写入数据
        data = torch.zeros((batch_data.shape[0], window_size, 33, 17))
        image = torch.zeros((batch_data.shape[0], window_size, 3, 224, 224))
        human_count = 0
        for i in range(15):
            data[:, :window_size-1] = data[:, 1:].clone()
            image[:, :window_size-1] = image[:, 1:].clone()
            # print(batch_image.shape)
            # 可见label 位
            if batch_data[0, :, i, 31, 10] == 1:
                human_count = window_size
            else:
                human_count -= 1
            # 如何window 里没有human，则把subtask label 改为unknown
            if human_count <= 0:
                batch_data[0, :, i, 30] = torch.from_numpy(np.asarray([8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            data[:, window_size - 1] = batch_data[0, :, i]
            tens = tf.transpose(tf.image.resize_with_pad(batch_image[:, i], 224, 224), perm=[0, 3, 1, 2]).numpy()
            image[:, window_size - 1] = batch_image[:, i].float()
            goal_name = subtask_list[int(data[0, window_size-1, 30, 0])]
            if count_dict.get(goal_name) is None:
                count_dict[goal_name] = 0
            else:
                count_dict[goal_name] += 1
            if not os.path.exists(f'./resample_new/{goal_name}'):
                os.makedirs(f'./resample_new/{goal_name}')
            
            with h5py.File(f'./resample_new/{goal_name}/{count_dict[goal_name]}.h5', 'w') as f:
                # 创建数据集
                data_set = f.create_dataset('data', (1, window_size, 33, 17), data=data, maxshape=(None, window_size, 33, 17))
                image_set = f.create_dataset('image', (1, window_size, 3, 224, 224), data=image, maxshape=(None, window_size, 3, 224, 224))
        pbar.update(1)