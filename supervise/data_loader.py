import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append("/home/zhihao/A2SP/rllib_A2SP")


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
import random


def cross_entropy(tensor1, tensor2):
    tmp = tensor1 * torch.log(tensor2 + 1e-8)
    return -torch.sum(tmp)

def criterion(input_matrix, subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict):

    subtask_predict = subtask_predict
    tar_index_1_predict = tar_index_1_predict
    tar_index_2_predict = tar_index_2_predict
    type_predict = type_predict

    batch_size = input_matrix.shape[0]

    subtask_name = input_matrix[:, 30, 0].to(torch.int).cuda()
    tar_index_1 = input_matrix[:, 30, 1].to(torch.int).cuda()
    tar_index_2 = input_matrix[:, 30, 2].to(torch.int).cuda()
    # type_name = input_matrix[:, 31, 0].int()
    # main_agent id: 1!!!
    main_type = input_matrix[:, 32, 0:6]

    subtask_predict_index = torch.argmax(subtask_predict, dim=1)
    subtask_predict_success_num = torch.sum(subtask_predict_index == subtask_name)
    tar_index_1_predict_index = torch.argmax(tar_index_1_predict, dim=1)
    tar_index_1_predict_success_num = torch.sum(tar_index_1_predict_index == tar_index_1)
    tar_index_2_predict_index = torch.argmax(tar_index_2_predict, dim=1)
    tar_index_2_predict_success_num = torch.sum(tar_index_2_predict_index == tar_index_2)
    type_distance = torch.norm(type_predict - main_type)
    total_num = subtask_name.shape[0]
    subtask_acc = subtask_predict_success_num / total_num
    tar_index_1_acc = tar_index_1_predict_success_num / total_num
    tar_index_2_acc = tar_index_2_predict_success_num / total_num
    height_distance = torch.mean(torch.abs(type_predict[:, 0] - main_type[:, 0])) / total_num
    weight_distance = torch.mean(torch.abs(type_predict[:, 1] - main_type[:, 1])) / total_num
    open_distance = torch.mean(torch.abs(type_predict[:, 2] - main_type[:, 2])) / total_num
    close_distance = torch.mean(torch.abs(type_predict[:, 3] - main_type[:, 3])) / total_num
    toggle_on_distance = torch.mean(torch.abs(type_predict[:, 4] - main_type[:, 4])) / total_num
    toggle_off_distance = torch.mean(torch.abs(type_predict[:, 5] - main_type[:, 5])) / total_num

    goal_name_probablity = torch.zeros(batch_size, SUBTASK_NUM).cuda()
    goal_name_probablity[np.arange(len(subtask_name)), subtask_name] = 1
    tar_index_1_probability = torch.zeros(batch_size, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN).cuda()
    tar_index_1_probability[np.arange(len(tar_index_1)), tar_index_1] = 1
    tar_index_2_probability = torch.zeros(batch_size, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN).cuda()
    tar_index_2_probability[np.arange(len(tar_index_2)), tar_index_2] = 1

    total_loss = 0
    total_loss = total_loss + cross_entropy(subtask_predict, goal_name_probablity)
    total_loss = total_loss + cross_entropy(tar_index_1_predict, tar_index_1_probability)
    total_loss = total_loss + cross_entropy(tar_index_2_predict, tar_index_2_probability)
    total_loss = total_loss + torch.norm(type_predict - main_type)

    return total_loss, subtask_acc, tar_index_1_acc, tar_index_2_acc, weight_distance, height_distance, open_distance, close_distance, toggle_on_distance, toggle_off_distance


class TrajectoryDataset(Dataset):
    def __init__(self, file_path, train=True, train_ratio=0.9, use_ratio=1.0):
        self.file_path = file_path
        self.len = len(glob.glob(os.path.join(file_path, f"{subtask_list[0]}/*")))
        self.files = {}
        for subtask in subtask_list:
            self.files[subtask] = glob.glob(os.path.join(file_path, f"{subtask}/*"))
            self.files[subtask].sort()
            if train:
                self.files[subtask] = self.files[subtask][:int(len(self.files[subtask]) * train_ratio * use_ratio)]
            else:
                self.files[subtask] = self.files[subtask][int(len(self.files[subtask]) * train_ratio * use_ratio):int(len(self.files[subtask]) * use_ratio)]
            self.len = max(self.len, len(self.files[subtask]))
        # self.files = glob.glob(os.path.join(file_path, "*"))
        # self.files.sort()
        # if train:
        #     self.files = self.files[:int(len(self.files) * train_ratio*use_ratio)]
        # else:
        #     self.files = self.files[int(len(self.files) * train_ratio*use_ratio):int(len(self.files)*use_ratio)]

    def __getitem__(self, index):
        subtask = random.choice(subtask_list)
        index = int(index / self.len * len(self.files[subtask]))
        with h5py.File(self.files[subtask][index], 'r') as f:
            # 获取对应索引的数据和标签
            sequence = f['data'][0]
            return torch.from_numpy(sequence)

    def __len__(self):
        # 数据集的长度为'matrices'数据集的长度
        return self.len

# 创建数据集
dataset_train = TrajectoryDataset('/home/zhihao/A2SP/rllib_A2SP/data_resample_symbolic/', train_ratio=1)
dataset_test = TrajectoryDataset('/home/zhihao/A2SP/rllib_A2SP/data/', train=False)

batch_size = 1

# 创建数据加载器
dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
# dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

print(len(dataset_train))

# with h5py.File('trajectories.h5', 'r') as f:
#     # 获取对应索引的数据和标签
#     sequence = f['data'][0]
#     print(sequence.shape)
#     print(f['data'].shape)
#     print(f['data'].shape[0])

# count = 1
# # 现在可以在训练循环中使用dataloader
# for data in dataloader:
#     sequences = data
#     print(count)
#     count += 1
#     # 使用sequences和labels进行训练

# model.load_state_dict(torch.load("/home/zhihao/A2SP/rllib_A2SP/model/oppent_modeling_single_3.pth"))
epoch_num = 1000
batch_num = int(len(dataset_train) / batch_size)

data_dict = {}

for i in range(30):
    data_dict[f"obj_{i+1}"] = []

data_dict["goal"] = []
data_dict["agent_1"] = []
data_dict["agent_2"] = []

# JSON 会将int key 转变为 str
tmp_dict = json.load(open("./object_index2name.json", "r"))
object_index2name = {}
for key, value in tmp_dict.items():
    object_index2name[int(key)] = value

for batch in dataloader:
    for j in range(5):
        data = batch[0, j]
        for i in range(30):
            if data[i, 10] != 0:
                data_dict[f"obj_{i+1}"].append(f"{object_index2name[int(data[i, 0])]} on {object_index2name[int(data[i, 1])]}")
                # data_dict[f"obj_{i+1}"].append(f"{object_index2name[int(data[i, 0])]}, parent_recpt: {object_index2name[int(data[i, 1])]}, isPickUp: {data[i, 2]}, isOpen: {data[i, 3]}, isCooked: {data[i, 4]}, isToogled: {data[i, 5]}, is_visible_1: {data[i, 6]}, is_visible_2: {data[i, 7]}, height: {data[i, 8]}, weight: {data[i, 9]}")
            else:
                data_dict[f"obj_{i+1}"].append("empty")
        data_dict["goal"].append(f"{subtask_list[int(batch[0, 4, 30, 0])]}, {object_index2name[int(batch[0, 4, 30, 1])]}, {object_index2name[int(batch[0, 4, 30, 2])]}")
        data_dict["agent_1"].append(f"AgentType: {data[31, 0: 6]}, action: {action_list[int(data[31, 6])]}, {object_index2name[int(data[31, 7])]}, Success: {data[31, 8]} With {object_index2name[int(data[31, 9])]}")
        data_dict["agent_2"].append(f"AgentType: {data[32, 0: 6]}, action: {action_list[int(data[32, 6])]}, {object_index2name[int(data[32, 7])]}, Success: {data[32, 8]} With {object_index2name[int(data[32, 9])]}")
        
    
    # break

# 将字典转换为DataFrame
df = pd.DataFrame(data_dict)

# 指定CSV文件路径
csv_file = 'data.csv'

# 将DataFrame保存为CSV文件
df.to_csv(csv_file, index=False)