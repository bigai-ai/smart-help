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

from model.opponent_modeling_v3 import OpponentModeling
from model.object_encoder_v3 import ObjectEncoder
from model.goal_encoder_v2 import GoalEncoder
from model.agent_encoder_v2 import AgentEncoder
from model.agent_encoder_without_type_v2 import AgentEncoderWithoutType
from env.constants import AGENT_TYPE_NUMBER, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN

class HelperModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(HelperModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if os.path.exists("./object_name2index.json"):
            self.object_name2index = json.load(open("./object_name2index.json", "r"))
        else:
            raise Exception("object_name2index.json not found")

        self.agent_num = obs_space.shape[1] - 9

        # print("for debug", num_outputs)

        if num_outputs is None:
            num_outputs = np.product(obs_space.shape)

        self.model = nn.Sequential(
            nn.Linear(128 * (2 + self.agent_num + 1), num_outputs),
            nn.ReLU(),
            nn.Linear(num_outputs, num_outputs),
            nn.Sigmoid()
        )

        self.type_encoder = nn.Sequential(        
            nn.Embedding(ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.type_encoder.load_state_dict(torch.load("/home/zhihao/A2SP/rllib_A2SP/model/type_encoder.pth"))
        for params in self.type_encoder.parameters():
            params.requires_grad = False

        self.object_encoder = ObjectEncoder(self.type_encoder)
        self.object_encoder.load_state_dict(torch.load("/home/zhihao/A2SP/rllib_A2SP/model/object_encoder.pth"))
        for params in self.object_encoder.parameters():
            params.requires_grad = False
        self.agent_encoder_without_type = AgentEncoderWithoutType(self.type_encoder)
        self.agent_encoder_without_type.load_state_dict(torch.load("/home/zhihao/A2SP/rllib_A2SP/model/agent_encoder_without_type.pth"))
        for params in self.agent_encoder_without_type.parameters():
            params.requires_grad = False
        self.opponent_modeling = OpponentModeling()
        self.opponent_modeling.load_state_dict(torch.load("/home/zhihao/A2SP/rllib_A2SP/model/oppenent_modeling.pth"))
        for params in self.opponent_modeling.parameters():
            params.requires_grad = False

        self.value_model = nn.Sequential(
            nn.Linear(128 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
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

        self.input_matrix = torch.zeros((1, 5, 33, 11))
        self.batch_size = 1

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):

        if input_dict['obs'].shape[0] != self.batch_size:
            self.batch_size = input_dict['obs'].shape[0]
            self.input_matrix = torch.zeros((self.batch_size, 5, 33, 11))

        self.input_matrix[:, :4] = self.input_matrix[:, 1:].clone()
        self.input_matrix[:, 4] = input_dict['obs']

        obj_obs = self.input_matrix[:, :, :30, :]

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
        src_key_padding_mask = obj_obs[:, :, -1]
        objs_observation_embedding = self.object_encoder(
            objs_type_id=objs_type_id,
            objs_parent_receptacle_id=objs_parent_receptacle_id,
            objs_properties=objs_properties,
            objs_height = objs_height,
            objs_weight = objs_weight, 
            src_key_padding_mask=src_key_padding_mask, 
        )
        objs_observation_embedding = objs_observation_embedding.squeeze(0)

        # helper_agent: 0
        # main_agent: 1
        # we can not directly know the type of main_agent
        main_agent_embedding = self.agent_encoder_without_type(
            agents_action = self.input_matrix[:, :, 32, 6:9], 
            held_objs_type = self.input_matrix[:, :, 32, 9]
        )

        type_embedding, subtask_embedding = self.opponent_modeling(
            obj_feature = objs_observation_embedding, 
            agents_embedding = main_agent_embedding
        )

        helper_agent_embedding = self.agent_encoder_without_type(
            agents_action = self.input_matrix[:, :, 31, 6:9], 
            held_objs_type = self.input_matrix[:, :, 31, 9]
        )

        objs_observation_embedding = objs_observation_embedding[:, 0].squeeze()
        main_agent_embedding = main_agent_embedding[:, 0].squeeze()
        helper_agent_embedding = helper_agent_embedding[:, 0].squeeze()
        subtask_embedding = subtask_embedding.squeeze()
        type_embedding = type_embedding.squeeze()

        # print("for debug", objs_observation_embedding.shape, main_agent_embedding.shape, type_embedding.shape, subtask_embedding.shape, helper_agent_embedding.shape)
        if self.batch_size != 1:
            observation_embedding = torch.cat([objs_observation_embedding, main_agent_embedding, type_embedding, subtask_embedding, helper_agent_embedding], dim=1)
            self.value_input = torch.cat([objs_observation_embedding, main_agent_embedding, type_embedding, subtask_embedding, helper_agent_embedding], dim=1)
        else:
            objs_observation_embedding = objs_observation_embedding.unsqueeze(0)
            main_agent_embedding = main_agent_embedding.unsqueeze(0)
            type_embedding = type_embedding.unsqueeze(0)
            subtask_embedding = subtask_embedding.unsqueeze(0)
            helper_agent_embedding = helper_agent_embedding.unsqueeze(0)
            observation_embedding = torch.cat([objs_observation_embedding, main_agent_embedding, type_embedding, subtask_embedding, helper_agent_embedding], dim=1)
            self.value_input = torch.cat([objs_observation_embedding, main_agent_embedding, type_embedding, subtask_embedding, helper_agent_embedding], dim=1)

        output = self.model(observation_embedding)

        return output, state

    def value_function(self) -> TensorType:
        return self.value_model(self.value_input).squeeze(1)
    
    # def estimate_subtask_and_type(self, input_dict: Dict[str, TensorType]):
    #     obj_obs = input_dict['obs'][:, :30, :]

    #     objs_type_id = obj_obs[:, :, 0].int()
    #     objs_parent_receptacle_id = obj_obs[:, :, 1].int()
    #     # batch_size, 30, obj_obs_len -> batch_size, 30, 3
    #     objs_properties = obj_obs[:, :, 2: 6 + self.agent_num]
    #     objs_height = obj_obs[:, :, 6 + self.agent_num]
    #     objs_weight = obj_obs[:, :, 7 + self.agent_num]
    #     src_key_padding_mask = obj_obs[:, :, -1]
    #     objs_observation_embedding = self.object_encoder(
    #         objs_type_id=objs_type_id,
    #         objs_parent_receptacle_id=objs_parent_receptacle_id,
    #         objs_properties=objs_properties,
    #         objs_height = objs_height,
    #         objs_weight = objs_weight, 
    #         src_key_padding_mask=src_key_padding_mask, 
    #     )
    #     objs_observation_embedding = objs_observation_embedding.squeeze(0)

    #     # helper_agent: 0
    #     # main_agent: 1
    #     # we can not directly know the type of main_agent
    #     main_agent_embedding = self.agent_encoder_without_type(
    #         agents_action = input_dict['obs'][:, 32, 6:9]
    #     )

    #     subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict = self.opponent_modeling.estimate_subtask_and_type(objs_observation_embedding, main_agent_embedding)
        
    #     return subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict
    

    # def get_estimate_loss(self, input_dict):
    #     subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict = self.estimate_subtask_and_type(input_dict=input_dict)

    #     subtask_predict = subtask_predict.detach().cpu()
    #     tar_index_1_predict = tar_index_1_predict.detach().to("cpu")
    #     tar_index_2_predict = tar_index_2_predict.detach().to("cpu")
    #     type_predict = type_predict.detach().to("cpu")

    #     batch_size = input_dict['obs'].shape[0]

    #     subtask_name = input_dict['obs'][:, 30, 0].int().to("cpu")
    #     tar_index_1 = input_dict['obs'][:, 30, 1].int().to("cpu")
    #     tar_index_2 = input_dict['obs'][:, 30, 2].int().to("cpu")
    #     # type_name = input_dict['obs'][:, 31, 0].int().to("cpu")
    #     # main_agent id: 1!!!
    #     main_type = input_dict['obs'][:, 32, 0:6].float().to("cpu")

    #     subtask_predict_index = torch.argmax(subtask_predict, dim=1)
    #     subtask_predict_success_num = torch.sum(subtask_predict_index == subtask_name)
    #     tar_index_1_predict_index = torch.argmax(tar_index_1_predict, dim=1)
    #     tar_index_1_predict_success_num = torch.sum(tar_index_1_predict_index == tar_index_1)
    #     tar_index_2_predict_index = torch.argmax(tar_index_2_predict, dim=1)
    #     tar_index_2_predict_success_num = torch.sum(tar_index_2_predict_index == tar_index_2)
    #     type_distance = torch.norm(type_predict - main_type)
    #     total_num = subtask_name.shape[0]

    #     goal_name_probablity = torch.zeros(batch_size, SUBTASK_NUM).to("cpu")
    #     goal_name_probablity[np.arange(len(subtask_name)), subtask_name] = 1
    #     tar_index_1_probability = torch.zeros(batch_size, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN).to("cpu")
    #     tar_index_1_probability[np.arange(len(tar_index_1)), tar_index_1] = 1
    #     tar_index_2_probability = torch.zeros(batch_size, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN).to("cpu")
    #     tar_index_2_probability[np.arange(len(tar_index_2)), tar_index_2] = 1

    #     total_loss = 0
    #     total_loss = total_loss + self.cross_entropy(subtask_predict, goal_name_probablity)
    #     total_loss = total_loss + self.cross_entropy(tar_index_1_predict, tar_index_1_probability)
    #     total_loss = total_loss + self.cross_entropy(tar_index_2_predict, tar_index_2_probability)
    #     total_loss = total_loss + torch.norm(type_predict - main_type)

    #     return total_loss, subtask_predict_success_num, tar_index_1_predict_success_num, tar_index_2_predict_success_num, type_distance, total_num

    # @staticmethod
    # def cross_entropy(tensor1, tensor2):
    #     tmp = tensor1 * torch.log(tensor2 + 1e-8)
    #     return -torch.sum(tmp)
    
