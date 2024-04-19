from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from env.goal import goal_list
from env.subtask import SUBTASK_NUM

from constants import (
    AGENT_TYPE_NUMBER,
    MAX_STEPS,
    OBJECT_BASE_PROPERTIES_NUMBER,
    DETAILED_GOALS_TYPE_NUM,
)

from env.constants import AGENT_TYPE_NUMBER, USED_ACTION_SPACE_LEN, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN

class OpponentModeling(nn.Module):
    def __init__(
        self,
        subtasks_type_num: int = SUBTASK_NUM,
        transformer_nhead: int = 2,
        transformer_dropout: float = 0.1,
        input_dim: int = 256,
        obj_types_num: int = ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN, 
    ) -> None:
        super().__init__()
        self.num_layers = 4
        self.hidden_size = 256

        self.time_sumarize_model = nn.Sequential(
            nn.Linear(5*256, 1028), 
            nn.ReLU(),
            nn.Linear(1028, 1028),
            nn.ReLU(),
            nn.Linear(1028, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.transformer_encoder_layer = nn.modules.TransformerEncoderLayer(
            d_model=128, 
            nhead=transformer_nhead, 
            dropout=transformer_dropout, 
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer_encoder_layer_goal = nn.modules.TransformerEncoderLayer(
            d_model=128, 
            nhead=transformer_nhead, 
            dropout=transformer_dropout, 
            dim_feedforward=256,
            batch_first=True
        )

        self.goal_encoder = nn.modules.TransformerEncoder(
            self.transformer_encoder_layer_goal,
            num_layers=4
        )

        self.type_encoder = nn.modules.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=4
        )

        self.goal_classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, subtasks_type_num), 
            nn.Softmax()
        )

        self.tar_index_1_predict = nn.Sequential(
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128), 
        )

        self.tar_index_2_predict = nn.Sequential(
            nn.Linear(384, 384), 
            nn.ReLU(), 
            nn.Linear(384, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128), 
        )

        self.tar_index_1_classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, obj_types_num),
            nn.Softmax()
        )

        self.tar_index_2_classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, obj_types_num),
            nn.Softmax()
        )

        self.type_MLP = nn.Sequential(
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 6), 
            nn.Sigmoid(), 
        )

        self.h = None
        self.c = None

        self.reset_lstm_state(64)
        
    def reset_lstm_state(self, batch_size):
        device = next(self.parameters()).device  # 获取模型的设备

        self.h = torch.zeros(self.num_layers, 128).to(torch.float32).to(device)
        self.c = torch.zeros(self.num_layers, 256).to(torch.float32).to(device)

    def forward(self, obj_feature, agents_embedding):

        total_feature = torch.cat((obj_feature, agents_embedding), dim=-1)

        # time_feature, [self.h, self.c] = self.lstm(total_feature, (self.h, self.c))
        batch_size = total_feature.shape[0]
        total_feature = total_feature.resize(batch_size, total_feature.shape[1]*256)
        # time_feature, [self.h, self.c] = self.lstm(total_feature)
        time_feature = self.time_sumarize_model(total_feature)
        # time_feature = time_feature[:, -1, :]
        # goal_feature = self.goal_encoder(time_feature)
        # type_feature = self.type_encoder(time_feature)
        goal_feature = self.goal_encoder(time_feature)
        type_feature = self.type_encoder(time_feature)
        
        tar_1_feature = self.tar_index_1_predict(torch.cat([time_feature, goal_feature], dim=-1))
        tar_2_feature = self.tar_index_2_predict(torch.cat([time_feature, goal_feature, tar_1_feature], dim=-1))

        return goal_feature, tar_1_feature, tar_2_feature, type_feature

    def estimate_subtask_and_type(self, obj_feature, agents_embedding):
        goal_feature, tar_1_feature, tar_2_feature, type_feature = self.forward(obj_feature, agents_embedding)
        goal_predict = self.goal_classifier(goal_feature)

        tar_index_1_predict = self.tar_index_1_classifier(tar_1_feature)
        tar_index_2_predict = self.tar_index_2_classifier(tar_2_feature)

        type_predict = self.type_MLP(type_feature)

        return goal_predict, tar_index_1_predict, tar_index_2_predict, type_predict