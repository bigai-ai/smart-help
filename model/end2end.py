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
from torch import tensor

from model.object_encoder_v3 import ObjectEncoder
from model.goal_encoder_v2 import GoalEncoder
from model.agent_encoder_v2 import AgentEncoder
from model.opponent_modeling_v3 import OpponentModeling
from model.agent_encoder_without_type_v2 import AgentEncoderWithoutType
from env.constants import AGENT_TYPE_NUMBER, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN

class End2End(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(End2End, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.agent_num = 2

        # print("for debug", num_outputs)

        if num_outputs is None:
            num_outputs = np.product(obs_space.shape)

        self.model = nn.Sequential(
            nn.Linear(np.product(obs_space.shape), num_outputs),
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

        self.value_model = nn.Sequential(
            nn.Linear(np.product(obs_space.shape), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):

        self.value_input = input_dict['obs_flat']
        
        output = self.model(self.value_input)

        return output, state

    def value_function(self) -> TensorType:

        return self.value_model(self.value_input).squeeze(1)
    
    
