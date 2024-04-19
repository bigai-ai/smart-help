import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append("/home/zhihao/文档/GitHub/rllib_A2SP")


from model.agent_encoder_new import AgentEncoder
from model.obj_encoder_new import ObjectEncoder
from model.opponent_modeling_new import OpponentModeling
import numpy as np

from env.constants import AGENT_TYPE_NUMBER, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN
from env.subtask import SUBTASK_NUM, subtask_list

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
import h5py
from tqdm import tqdm

import glob
from constants import ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN

import json
import wandb
import random



class Classifier_OppenentModeling(nn.Module):
    def __init__(self):
        super(Classifier_OppenentModeling, self).__init__()

        self.type_encoder = nn.Sequential(        
            nn.Embedding(ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        self.object_encoder = ObjectEncoder(type_encoder=self.type_encoder)
        self.agent_encoder = AgentEncoder(object_encoder=self.type_encoder)
        
        self.opponent_modeling = OpponentModeling()

    def forward(self, input_matrix):

        obj_obs = input_matrix[:, :, :30, :]

        objs_type_id = obj_obs[:, :, :, 0].to(torch.int).cuda()
        objs_parent_receptacle_id = obj_obs[:, :, :, 1].to(torch.int).cuda()
        # batch_size, 30, obj_obs_len -> batch_size, 30, 3
        objs_properties = obj_obs[:, :, :, 2: 6 + 2]
        objs_height = obj_obs[:, :, :, 6 + 2]
        objs_weight = obj_obs[:, :, :, 7 + 2]
        objs_pos = obj_obs[:, :, :, 10:13]
        objs_dis = obj_obs[:, :, :, 13]
        src_key_padding_mask = obj_obs[:, :, -1]
        objs_observation_embedding = self.object_encoder(
            objs_type_id=objs_type_id,
            objs_parent_receptacle_id=objs_parent_receptacle_id,
            objs_properties=objs_properties,
            objs_height = objs_height,
            objs_weight = objs_weight, 
            objs_pos = objs_pos, 
            objs_dis = objs_dis, 
            src_key_padding_mask=src_key_padding_mask, 
        )
        objs_observation_embedding = objs_observation_embedding.squeeze(0)

        # helper_agent: 0
        # main_agent: 1
        # we can not directly know the type of main_agent
        main_agent_embedding = self.agent_encoder(
            agents_action = input_matrix[:, :, 32, 6:9], 
            held_objs_type = input_matrix[:, :, 32, 9], 
            agents_pos = input_matrix[:, :, 32, :3], 
            agents_rot = input_matrix[:, :, 32, 3:6], 
        )

        # print(objs_observation_embedding.shape, main_agent_embedding.shape)

        subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict = self.opponent_modeling.estimate_subtask_and_type(objs_observation_embedding, main_agent_embedding)

        return subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict


# def cross_entropy(tensor1, tensor2):
#     tmp = tensor1 * torch.log(tensor2 + 1e-8)
#     return -torch.sum(tmp)

def criterion(input_matrix, subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict, window_size):
    batch_size = input_matrix.shape[0]

    subtask_name = input_matrix[:, window_size-1, 30, 0].to(torch.int).cuda()
    tar_index_1 = input_matrix[:, window_size-1, 30, 1].to(torch.int).cuda()
    tar_index_2 = input_matrix[:, window_size-1, 30, 2].to(torch.int).cuda()
    # subtask_name.required_grad = False
    # tar_index_1.required_grad = False
    # tar_index_2.required_grad = False
    # type_name = input_matrix[:, 31, 0].int()
    # main_agent id: 1!!!
    main_type = input_matrix[:, window_size-1, 32, 11:17]
    # main_type.required_grad = False

    subtask_predict_index = torch.argmax(subtask_predict, dim=1)
    subtask_predict_success_num = torch.sum(subtask_predict_index == subtask_name)
    tar_index_1_predict_index = torch.argmax(tar_index_1_predict, dim=1)
    tar_index_1_predict_success_num = torch.sum(tar_index_1_predict_index == tar_index_1)
    tar_index_2_predict_index = torch.argmax(tar_index_2_predict, dim=1)
    tar_index_2_predict_success_num = torch.sum(tar_index_2_predict_index == tar_index_2)
    total_num = subtask_name.shape[0]
    subtask_acc = subtask_predict_success_num / total_num
    tar_index_1_acc = tar_index_1_predict_success_num / total_num
    tar_index_2_acc = tar_index_2_predict_success_num / total_num
    height_distance = torch.mean(torch.abs(type_predict[:, 0] - main_type[:, 0]))
    weight_distance = torch.mean(torch.abs(type_predict[:, 1] - main_type[:, 1]))
    open_distance = torch.mean(torch.abs(type_predict[:, 2] - main_type[:, 2]))
    close_distance = torch.mean(torch.abs(type_predict[:, 3] - main_type[:, 3]))
    toggle_on_distance = torch.mean(torch.abs(type_predict[:, 4] - main_type[:, 4]))
    toggle_off_distance = torch.mean(torch.abs(type_predict[:, 5] - main_type[:, 5]))

    goal_name_probablity = torch.zeros(batch_size, SUBTASK_NUM).cuda()
    goal_name_probablity[np.arange(len(subtask_name)), subtask_name] = 1
    tar_index_1_probability = torch.zeros(batch_size, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN).cuda()
    tar_index_1_probability[np.arange(len(tar_index_1)), tar_index_1] = 1
    tar_index_2_probability = torch.zeros(batch_size, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN).cuda()
    tar_index_2_probability[np.arange(len(tar_index_2)), tar_index_2] = 1

    total_loss = F.binary_cross_entropy(subtask_predict, goal_name_probablity)
    total_loss = total_loss + F.binary_cross_entropy(tar_index_1_predict, tar_index_1_probability)
    total_loss = total_loss + F.binary_cross_entropy(tar_index_2_predict, tar_index_2_probability) / 3
    total_loss = total_loss + (torch.sum((type_predict - main_type) ** 2)) / 3

    wandb.log(
        {
            "loss1": F.binary_cross_entropy(subtask_predict, goal_name_probablity), 
            "loss2": F.binary_cross_entropy(tar_index_1_predict, tar_index_1_probability),
            "loss3": F.binary_cross_entropy(tar_index_2_predict, tar_index_2_probability) / 3,
            "loss4": (torch.sum((type_predict - main_type) ** 2)) / 3,
        }
    )

    # print("for debug loss", torch.max(subtask_predict), torch.min(subtask_predict), torch.max(tar_index_1_predict), torch.min(tar_index_1_predict), torch.max(tar_index_2_predict), torch.min(tar_index_2_predict), torch.max(type_predict), torch.min(type_predict))
    # print("for debug loss", torch.max(goal_name_probablity), torch.min(goal_name_probablity), torch.max(tar_index_1_probability), torch.min(tar_index_1_probability), torch.max(tar_index_2_probability), torch.min(tar_index_2_probability), torch.max(main_type), torch.min(main_type))
    # print("for debug loss", torch.max(total_loss), torch.min(total_loss))

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

    def __getitem__(self, index):
        subtask = random.choice(subtask_list)
        index = int(index / self.len * len(self.files[subtask]))
        with h5py.File(self.files[subtask][index], 'r') as f:
            # 获取对应索引的数据和标签
            data = f['data'][:]
            return torch.from_numpy(data)

    def __len__(self):
        # 数据集的长度为'matrices'数据集的长度
        return self.len

if __name__ == "__main__":
    wandb.init(
        project="Op_dep"
    )

    # 创建数据集
    dataset_train = TrajectoryDataset('/home/zhihao/文档/GitHub/rllib_A2SP/resample_new/')
    dataset_test = TrajectoryDataset('/home/zhihao/文档/GitHub/rllib_A2SP/resample_new/', train=False)

    batch_size = 32

    # 创建数据加载器
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    print(len(dataset_train))

    model = Classifier_OppenentModeling()
    # model.load_state_dict(torch.load("/home/zhihao/A2SP/rllib_A2SP/model/oppent_modeling_single_0.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    epoch_num = 1000
    batch_num = int(len(dataset_train) / batch_size)
    truncate_backprop = 5
    window_size = 5

    if torch.cuda.is_available():
        model = model.cuda()

    with tqdm(total=epoch_num, desc="epoch") as pbar:
        for epoch in range(epoch_num):
            with tqdm(total=batch_num, desc="batch") as pbar_small:
                for batch in dataloader:
                    if torch.cuda.is_available():
                        data = batch.cuda()
                    else:
                        exit()

                    data = data.squeeze()
                    if len(data.shape) == 3:
                        data = data.unsqueeze(0)

                    # 前向传播
                    subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict = model(data)
                    # print(subtask_predict.shape, tar_index_1_predict.shape, tar_index_2_predict.shape, type_predict.shape)

                    # 计算损失
                    total_loss, subtask_acc, tar_index_1_acc, tar_index_2_acc, weight_distance, height_distance, open_distance, close_distance, toggle_on_distance, toggle_off_distance = criterion(data, subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict, window_size)

                    # 反向传播和参数更新
                    optimizer.zero_grad()
                    total_loss.backward(retain_graph=True)
                    # total_loss.backward()

                    # 进行梯度裁剪
                    # clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    wandb.log(
                        {
                            "loss": total_loss,
                            "subtask_acc": subtask_acc,
                            "tar_index_1_acc": tar_index_1_acc,
                            "tar_index_2_acc": tar_index_2_acc,
                            "weight_distance": weight_distance,
                            "height_distance": height_distance,
                            "open_distance": open_distance,
                            "close_distance": close_distance,
                            "toggle_on_distance": toggle_on_distance,
                            "toggle_off_distance": toggle_off_distance,
                        }
                    )
                    pbar_small.update(1)
            with torch.no_grad():
                count = 1
                all_subtask_acc = 0
                all_tar_index_1_acc = 0
                all_tar_index_2_acc = 0
                all_weight_distance = 0
                all_height_distance = 0
                all_open_distance = 0
                all_close_distance = 0
                all_toggle_on_distance = 0
                all_toggle_off_distance = 0
                for batch in dataloader_test:
                    # for i in range(15):
                    #     if i < window_size:
                    #         data = torch.zeros((batch.shape[0], window_size, 33, 11))
                    #         data[:, window_size-1-i:] = batch[:, :i+1]
                    #     else:
                    #         data = batch[:, i-window_size+1: i+1]
                    #     if torch.cuda.is_available():
                    #         data = data.cuda()
                    # 前向传播
                    if torch.cuda.is_available():
                        data = batch.cuda()

                    data = data.squeeze()
                    if len(data.shape) == 3:
                        data = data.unsqueeze(0)

                    subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict = model(data)

                    # 计算损失
                    total_loss, subtask_acc, tar_index_1_acc, tar_index_2_acc, weight_distance, height_distance, open_distance, close_distance, toggle_on_distance, toggle_off_distance = criterion(data, subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict, window_size)

                    all_subtask_acc = all_subtask_acc / count * (count - 1) + subtask_acc / count
                    all_tar_index_1_acc = all_tar_index_1_acc / count * (count - 1) + tar_index_1_acc / count
                    all_tar_index_2_acc = all_tar_index_2_acc / count * (count - 1) + tar_index_2_acc / count
                    all_weight_distance = all_weight_distance / count * (count - 1) + weight_distance / count
                    all_height_distance = all_height_distance / count * (count - 1) + height_distance / count
                    all_open_distance = all_open_distance / count * (count - 1) + open_distance / count
                    all_close_distance = all_close_distance / count * (count - 1) + close_distance / count
                    all_toggle_on_distance = all_toggle_on_distance / count * (count - 1) + toggle_on_distance / count
                    all_toggle_off_distance = all_toggle_off_distance / count * (count - 1) + toggle_off_distance / count

                    count += 1
            wandb.log(
                {
                    "test_subtask_acc": all_subtask_acc, 
                    "test_tar_index_1_acc": all_tar_index_1_acc,
                    "test_tar_index_2_acc": all_tar_index_2_acc,
                    "test_weight_distance": all_weight_distance,
                    "test_height_distance": all_height_distance,
                    "test_open_distance": all_open_distance,
                    "test_close_distance": all_close_distance,
                    "test_toggle_on_distance": all_toggle_on_distance,
                    "test_toggle_off_distance": all_toggle_off_distance,
                }
            )
            if epoch % 10 == 9:
                torch.save(model.state_dict(), f"/home/zhihao/文档/GitHub/rllib_A2SP/model/op_{epoch}.pth")
            pbar.update(1)