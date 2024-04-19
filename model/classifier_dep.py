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


class Classifier_OppenentModeling_v2(nn.Module):
    def __init__(self):
        super(Classifier_OppenentModeling_v2, self).__init__()

        # if os.path.exists("./object_index2name.json") and os.path.getsize("./object_index2name.json") > 0:
        #     # JSON 会将int key 转变为 str
        #     tmp_dict = json.load(open("./object_index2name.json", "r"))
        #     self.object_index2name = {}
        #     for key, value in tmp_dict.items():
        #         self.object_index2name[int(key)] = value
        # else:
        #     raise Exception("object_index2name.json not found")

        self.type_encoder = nn.Sequential(        
            nn.Embedding(ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.type_encoder.load_state_dict(torch.load("model/type_encoder_dep.pth"))
        for params in self.type_encoder.parameters():
            params.requires_grad = False

        self.object_encoder = ObjectEncoder(type_encoder=self.type_encoder)
        self.object_encoder.load_state_dict(torch.load("model/object_encoder_dep.pth"))
        for params in self.object_encoder.parameters():
            params.requires_grad = False
        self.agent_encoder = AgentEncoder(object_encoder=self.type_encoder)
        self.agent_encoder.load_state_dict(torch.load("model/agent_encoder_without_type_dep.pth"))
        for params in self.agent_encoder.parameters():
            params.requires_grad = False
        
        self.opponent_modeling = OpponentModeling()
        self.opponent_modeling.load_state_dict(torch.load("model/oppenent_modeling_dep.pth"))
        for params in self.opponent_modeling.parameters():
            params.requires_grad = False

        # https://cloud.tsinghua.edu.cn/d/1a9af7bd2457494fb02a/

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
            agents_action = input_matrix[:, :, 31, 6:9], 
            held_objs_type = input_matrix[:, :, 31, 9], 
            agents_pos = input_matrix[:, :, 31, :3], 
            agents_rot = input_matrix[:, :, 31, 3:6], 
        )

        # print(objs_observation_embedding.shape, main_agent_embedding.shape)

        subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict = self.opponent_modeling.estimate_subtask_and_type(objs_observation_embedding, main_agent_embedding)

        return subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict

