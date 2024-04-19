from typing import Dict, List
from ray.rllib.utils.framework import TensorType
import torch.nn as nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from gymnasium import spaces
import torch.nn.functional as F

import json
import os

from env.goal import GOAL_NUM
from env.subtask import SUBTASK_NUM
import numpy as np

from model.opponent_modeling_new import OpponentModeling
from model.obj_encoder_new import ObjectEncoder
from model.agent_encoder_new import AgentEncoder
from env.constants import AGENT_TYPE_NUMBER, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN

class HelperModelDep(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(HelperModelDep, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if os.path.exists("./object_name2index.json"):
            self.object_name2index = json.load(open("./object_name2index.json", "r"))
        else:
            raise Exception("object_name2index.json not found")

        self.agent_num = 2

        # print("for debug", num_outputs)

        if num_outputs is None:
            num_outputs = np.product(obs_space.shape)

        self.model = nn.Sequential(
            nn.Linear(128 * 2 + num_outputs + 256, num_outputs),
            nn.ReLU(),
            nn.Linear(num_outputs, num_outputs),
            nn.ReLU(), 
            nn.Linear(num_outputs, num_outputs),
            nn.ReLU(),
            nn.Linear(num_outputs, num_outputs),
            nn.ReLU(),
            nn.Linear(num_outputs, num_outputs),
            nn.Sigmoid()
        )

        self.type_encoder = nn.Sequential(        
            nn.Embedding(ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.type_encoder.load_state_dict(torch.load("./model/agent_encoder_dep.pth"))
        for params in self.type_encoder.parameters():
            params.requires_grad = False

        self.object_encoder = ObjectEncoder(self.type_encoder)
        self.object_encoder.load_state_dict(torch.load("./model/object_encoder_dep.pth"))
        for params in self.object_encoder.parameters():
            params.requires_grad = False
        self.agent_encoder = AgentEncoder(self.type_encoder)
        self.agent_encoder.load_state_dict(torch.load("./model/agent_encoder_without_type_dep.pth"))
        for params in self.agent_encoder.parameters():
            params.requires_grad = False
        self.opponent_modeling = OpponentModeling()
        self.opponent_modeling.load_state_dict(torch.load("./model/oppenent_modeling_dep.pth"))
        for params in self.opponent_modeling.parameters():
            params.requires_grad = False

        self.value_model = nn.Sequential(
            nn.Linear(128 * 2 + num_outputs + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.input_matrix = torch.zeros((1, 5, 32, 15))
        self.batch_size = 1

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):

        if input_dict['obs'].shape[0] != self.batch_size:
            self.batch_size = input_dict['obs'].shape[0]
            self.input_matrix = torch.zeros((self.batch_size, 5, 32, 15))

        self.input_matrix[:, :4] = self.input_matrix[:, 1:].clone()
        self.input_matrix[:, 4] = input_dict['obs']

        obj_obs = self.input_matrix[:, :, :30, :]
        obs_flat = input_dict['obs_flat']

        if torch.cuda.is_available():
            obj_obs = obj_obs.cuda()
            objs_type_id = obj_obs[:, :, :, 0].to(torch.int).cuda()
            objs_parent_receptacle_id = obj_obs[:, :, :, 1].to(torch.int).cuda()
        else:
            objs_type_id = obj_obs[:, :, :, 0].to(torch.int)
            objs_parent_receptacle_id = obj_obs[:, :, :, 1].to(torch.int)
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
            agents_action = self.input_matrix[:, :, 31, 6:9], 
            held_objs_type = self.input_matrix[:, :, 31, 9], 
            agents_pos = self.input_matrix[:, :, 31, :3], 
            agents_rot = self.input_matrix[:, :, 31, 3:6], 
        )

        type_embedding, tar_1_embedding, tar_2_embedding, subtask_embedding = self.opponent_modeling(
            obj_feature = objs_observation_embedding, 
            agents_embedding = main_agent_embedding
        )

        objs_observation_embedding = objs_observation_embedding[:, 0].squeeze()
        main_agent_embedding = main_agent_embedding[:, 0].squeeze()
        subtask_embedding = subtask_embedding.squeeze()
        type_embedding = type_embedding.squeeze()
        tar_1_embedding = tar_1_embedding.squeeze()
        tar_2_embedding = tar_2_embedding.squeeze()

        # print("for debug", objs_observation_embedding.shape, main_agent_embedding.shape, type_embedding.shape, subtask_embedding.shape, helper_agent_embedding.shape)
        if self.batch_size != 1:
            observation_embedding = torch.cat([obs_flat, type_embedding, subtask_embedding, tar_1_embedding, tar_2_embedding], dim=1)
            self.value_input = observation_embedding
        else:
            type_embedding = type_embedding.unsqueeze(0)
            subtask_embedding = subtask_embedding.unsqueeze(0)
            tar_1_embedding = tar_1_embedding.unsqueeze(0)
            tar_2_embedding = tar_2_embedding.unsqueeze(0)
            # obs_flat = obs_flat.unsqueeze(0)
            observation_embedding = torch.cat([obs_flat, type_embedding, subtask_embedding, tar_1_embedding, tar_2_embedding], dim=1)
            self.value_input = observation_embedding

        output = self.model(observation_embedding)

        return output, state

    def value_function(self) -> TensorType:
        return self.value_model(self.value_input).squeeze(1)
    