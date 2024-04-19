from re import L
from typing import (
    Optional,
    Tuple,
    Sequence,
    Union,
    Dict,
    Any,
)
import gymnasium as gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import copy
from ray.rllib.utils.typing import ModelConfigDict

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.a2c import A2CConfig

# from model.normal import (
#     LinearActorHead, 
#     LinearCriticHead, 
#     Memory,
# )

from env.constants import (
    ObservationType,
)

# from allenact.base_abstractions.distributions import CategoricalDistr

from env.constants import (
    MAX_STEPS,
    OBJECT_BASE_PROPERTIES_NUMBER,
    AGENT_TYPE_NUMBER,
    GOALS_TYPE_NUM,
    SUBTASKS_TYPE_NUM,
    DETAILED_GOALS_TYPE_NUM,
    TRAIN_PROCESS,
    PROPERTIES_SELECTED,
    SYMBOLIC_OBJECT_EMBED_LEN, 
    MAX_OBJS_NUM_FROM_OBSERVATION,
    ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN,
    USED_ACTION_SPACE_LEN
)
# from A2SP.opponent_modeling import OpponentModeling

# TODO: How to import the above params?
AGENT_TYPE_NUMBER = 7
SUBTASKS_TYPE_NUM = 6
DETAILED_GOALS_TYPE_NUM = 57
ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN = 70
USED_ACTION_SPACE_LEN = 48

class MainAgentModel(TorchModelV2):
    '''
    An RNN actor-critic model for coordination tasks.
    '''
    def __init__(
        self, 
        model_config,
        action_space: gym.Space, 
        obs_space: gym.spaces.Dict,
        obj_properties_num: int = 7,
        use_src_key_padding_mask: bool = False,
        hidden_size: int = 512,
        rnn_layers_num: int = 1,
        rnn_type = 'GRU',
        use_goal_encoder: bool = True,
        goal_encoder_output_dim: int = 128,
        use_subtask_encoder: bool = True,
        subtask_encoder_output_dim: int = 128,
        num_outputs = 512, 
        agent_num = 2,
        name = 'main_agent_model',
        # model_config = A2CConfig, 

    ):
        super().__init__(action_space, obs_space, num_outputs, model_config, name)

        self.agent_num = agent_num

        assert(
            hidden_size % 8 == 0
        ),'output_dim or symbolic observation encoder should be divisible by 8'

        
        # Agent embed
        hidden_layer_divided_ratio = 2 
        intermediate_hidden_state = int(hidden_size / hidden_layer_divided_ratio)

        # observation encoder
        self.object_encoder = ObjectEncoder(
            obj_properties_num=obj_properties_num,
            output_dim=intermediate_hidden_state,
            use_src_key_padding_mask=use_src_key_padding_mask
        )

       
        # complicated agent encoder, use more information about agents
        self.agent_encoder = AgentEncoder(
            output_dim=intermediate_hidden_state
        )


        self._hidden_size = hidden_size
        
        # goal_encoder
        self.use_goal_encoder = use_goal_encoder
        if use_goal_encoder:
            self._hidden_size += goal_encoder_output_dim
            self.goal_encoder = GoalEncoder(
                output_dim=goal_encoder_output_dim
            )

        # subtask encoder
        self.use_subtask_encoder = use_subtask_encoder
        if use_subtask_encoder:
            self._hidden_size += subtask_encoder_output_dim
            self.subtask_encoder = SubtaskEncoder(
                output_dim=subtask_encoder_output_dim
            )

        assert(
            isinstance(action_space, gym.spaces.Tuple)
        ),"Error, action space gym.spaces.Discrete is deprecated!!!"
        

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    # TODO: 
    def forward(
        self, 
        observations, 
 
        prev_actions: torch.Tensor, 
        masks: torch.FloatTensor
    ) -> Tuple[torch.Tensor]:

        objs_observations_embedding = []

        obj_obs = observations["obj_obs"]

        # symbolic_object_observations = observations['in_symbolic_object']
        # symbolic_agent_observations = observations['in_symbolic_agent']
        # symbolic_current_goal_observations = observations['in_symbolic_goal']
        # symbolic_current_subtask_observations = observations['in_symbolic_subtask']
        # steps_taken_num = observations['in_symbolic_progress']

        # batch_shape = steps_taken_num.shape
        # print(batch_shape)
        # print(steps_taken_num)

        # objs_type_id = symbolic_object_observations['objs_type_id']
        # objs_pos = symbolic_object_observations['objs_pos']
        # objs_properties = symbolic_object_observations['objs_properties']
        # print(objs_properties.shape)
        assert isinstance(obj_obs, torch.Tensor)
        # batch_size, obj_obs_len, 30 -> batch_size, 1, 30
        objs_type_id = obj_obs[:, 0, :]
        # batch_size, obj_obs_len, 30 -> batch_size, 3, 30
        objs_pos = obj_obs[:, 1: 4, :]
        # 
        objs_properties = obj_obs[:, 4: 10 + self.agent_num, :]
        objs_parent_receptacle_id = obj_obs[:, 10 + self.agent_num, :]
        # objs_parent_receptacle_id = symbolic_object_observations['objs_parent_receptacle_id']
        # src_key_padding_mask = symbolic_object_observations['src_key_padding_mask']
        objs_observation_embedding = self.object_encoder(
            objs_type_id,
            objs_pos,
            objs_properties,
            objs_parent_receptacle_id,
            # src_key_padding_mask
        )
        objs_observations_embedding.append(objs_observation_embedding)
            
        # print(objs_observation_embedding.shape)
        bs_objs_observations_embedding = torch.stack(
            objs_observations_embedding, 2
        )
        # print(bs_objs_observations_embedding.shape)
        agents_spec = symbolic_agent_observations['agents_spec']
        agents_pose = symbolic_agent_observations['agents_pose']
        agents_action = symbolic_agent_observations['agents_action']
        agents_type = agents_spec.split([1, 1], dim=-1)[-1]

        
        helper_agents_pose = agents_pose[..., 1 : , :]
        helper_agents_action = agents_action[..., 1 : , :]
        helper_agents_type = agents_type[..., 1 : , :]
        # bs_helper_agents_embedding = self.agent_encoder(
        #     helper_agents_type, helper_agents_pose, helper_agents_action
        # ).view(*batch_shape, -1)
        bs_helper_agents_embedding = self.agent_encoder(
            agents_type,agents_pose,agents_action
        ).view(*batch_shape, -1)

        # print(bs_helper_agents_embedding.shape)

        x = torch.cat(
            (bs_objs_observations_embedding.view(*batch_shape, -1), bs_helper_agents_embedding),
            dim=-1
        )
        
        if self.use_goal_encoder:
            current_goals_type_id = symbolic_current_goal_observations[
                'goals_type_id'
            ]
            current_goal_objs_type_id = symbolic_current_goal_observations[
                'goal_objs_type_id'
            ]
            current_goal_objs_pos = symbolic_current_goal_observations[
                'goal_objs_pos'
            ]
            goal_embedding = self.goal_encoder(
                current_goals_type_id, current_goal_objs_type_id, current_goal_objs_pos
            ).squeeze(-2)
            x = torch.cat(
                [
                    x, goal_embedding
                ],
                dim=-1
            )
        
        if self.use_subtask_encoder:
            subtask_type_id = symbolic_current_subtask_observations[
                'subtask_type_id'
            ]
            target_obj_type_id = symbolic_current_subtask_observations[
                'target_obj_type_id'
            ]
            target_obj_pos = symbolic_current_subtask_observations[
                'target_obj_pos'
            ]
            receptacle_obj_type_id = symbolic_current_subtask_observations[
                'receptacle_obj_type_id'
            ]
            receptacle_obj_pos = symbolic_current_subtask_observations[
                'receptacle_obj_pos'
            ]
            subtask_embedding = self.subtask_encoder(
                subtask_type_id, target_obj_type_id, target_obj_pos,
                receptacle_obj_type_id, receptacle_obj_pos
            )
            x = torch.cat(
                [
                    x, subtask_embedding
                ],
                dim=-1
            )
        
        x = x.unsqueeze(-2)
        # print(x.shape)
        #x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)
        return x

class MultiLinearActorHead(nn.Module):
    def __init__(self, agents_num: int, inputs_num: int, outputs_num: int) -> None:
        super().__init__()
        self.agents_num = agents_num
        if agents_num == 1:
            self.actor_header_0 = nn.Linear(inputs_num, outputs_num)
        elif agents_num == 2:
            self.actor_header_0 = nn.Linear(inputs_num, outputs_num)
            self.actor_header_1 = nn.Linear(inputs_num, outputs_num)
        else:
            raise NotImplementedError
    def forward(self, x: torch.FloatTensor):
        if self.agents_num == 1:
            xs_cat_processed = self.actor_header_0(x)
        elif self.agents_num == 2:
            x_processed_0 = self.actor_header_0(x)
            x_processed_1 = self.actor_header_1(x)
            xs_cat_processed = torch.cat(
                [x_processed_0, x_processed_1], dim=-2
            )
        else:
            raise NotImplementedError
        return torch.distributions.categorical.Categorical(logits=xs_cat_processed)

class ObjectEncoder(nn.Module):
    def __init__(
        self, 
        obj_types_num: int = ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN, 
        obj_pos_coordinates_num: int = 3,
        obj_properties_num: int = 8,
        output_dim: int = 128,
        transformer_n_head: int = 8,
        transformer_dropout: float = 0.2,
        use_src_key_padding_mask: bool = False
    ) -> None:
        super().__init__()

        output_dim = int(output_dim)
        assert(
            output_dim % 2 == 0
        ),'output_dim or symbolic observation encoder should be divisible by 2'

        _hidden_size = int(output_dim / 2)
        
        # 20 is the observation length
        # batch_shape x observe_len -> batch_shape x observe_len x (output_dim / 2)
        self.type_encoder = nn.Sequential(        
            nn.Embedding(obj_types_num, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        # batch_shape x observe_len x 3 -> batch_shape x observe_len x (output_dim / 2)
        self.pos_encoder = nn.Sequential(         
            nn.Linear(obj_pos_coordinates_num, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        # batch_shape x observe_len x 8 -> batch_shape x  20 x (output_dim / 2)
        self.property_encoder = nn.Sequential(   
            nn.Linear(obj_properties_num, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        # batch_shape x observe_len x (2 * output_dim) -> batch_shape x observe_len x output_dim
        self.state_encoder = nn.Sequential(         
            nn.ReLU(),
            nn.Linear(in_features = 2*output_dim, out_features = output_dim)
        )

        self.obj_embedding = nn.Parameter(data=torch.randn(1, output_dim))

        self.use_src_key_padding_mask = use_src_key_padding_mask
        self.transformer_encoder_layer = nn.modules.TransformerEncoderLayer(
            d_model=output_dim, 
            nhead=transformer_n_head, 
            dropout=transformer_dropout, 
            dim_feedforward=output_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.modules.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=4
        )

    def forward(
        self, 
        objs_type_id, 
        objs_pos, 
        objs_properties, 
        objs_parent_receptacle_id, 
        src_key_padding_mask,
    ):
        objs_pos = objs_pos.to(torch.float)
        objs_properties = objs_properties.to(torch.float)


        # batch_shape x observe_len -> batch_shape x observe_len x (output_dim / 2)
        objs_type_embedding = self.type_encoder(objs_type_id)
        # batch_shape x observe_len x 3 -> batch_shape x observe_len x (output_dim / 2)
        objs_pos_embedding = self.pos_encoder(objs_pos)
        # batch_shape x observe_len x 8 -> batch_shape x observe_len x (output_dim / 2)
        objs_properties_embedding = self.property_encoder(objs_properties)
        # batch_shape x observe_len -> batch_shape x observe_len x (output_dim / 2)
        objs_parent_receptacle_id_embedding = self.type_encoder(objs_parent_receptacle_id)

        # batch_shape x observe_len x (2 * output_dim)
        _objs_state_embedding = torch.cat(   
            [
                objs_type_embedding,
                objs_pos_embedding,
                objs_properties_embedding,
                objs_parent_receptacle_id_embedding
            ],
            dim=-1
        )
        # batch_shape x observe_len x (2 * output_dim) -> batch_shape x observe_len x output_dim
        objs_state_embedding = self.state_encoder(_objs_state_embedding)
        assert(
            len(objs_state_embedding.shape) == 4
        ),'processed_observation_embedding has false dimension!!!'
        # batch_shape = 1 x batch_size
        batch_shape = objs_state_embedding.shape[0 : 2] 
        # if batch_shape[0] == 64:
        #     print('debug')
        # batch_shape x output_dim
        bs_obj_embedding = self.obj_embedding.repeat(*batch_shape, 1, 1)
        # 1 x batch_size x observe_len x output_dim -> 1 x batch_size x (observe_len+1) x output_dim
        objs_state_embedding = torch.cat(
            [
                bs_obj_embedding, 
                objs_state_embedding
            ],
            dim=-2
        )
        # embedding_shape = (observe_len+1) x output_dim
        embedding_shape = objs_state_embedding.shape[2 : ]  
        
        # 1 x batch_size x (observe_len+1) x output_dim -> batch_size x (observe_len+1) x output_dim
        objs_state_embedding_reshaped = objs_state_embedding.view(-1, *embedding_shape)
        
        # mask operation
        '''
        With respect to mask operation, the following variables are mentionable:
            - For BoolTensor src_key_padding_mask_bool: 
                - the positions with the value of True will be ignored,
                - while the position with the value of False will be unchanged.
            - For BoolTensor has_objs:
                - the positions with the value of True represent observed objects' number is not 0, 
                - while the positions with the value of False represent observed objects' number is 0. 
        
        The final src_key_padding_mask_bool = torch.eq(src_key_padding_mask_bool, has_objs).
        
        This is because when no object is observed, the values of original 
        src_key_padding_mask_bool are all True, leading to a 'nan' embedding. 
        Accordingly, the training process is interrupted because of the error. 
        
        In order to ensure that no errors are reported during the training process, 
        if no object is observed, the values of original src_key_padding_mask_bool 
        are all set to False.
        
        That is to say, if there are some observed objects, only these objects will
        be encoded, 'None' will not be encoded. If no object is observed, all the 'None'
        will be encoded to avoid error. 
        '''
        # mask_shape = observe_len
        mask_shape = src_key_padding_mask.shape[2 : ] 
        # 1 x batch_size x observe_len -> batch_size x observe_len
        src_key_padding_mask_bool = src_key_padding_mask.view(-1, *mask_shape).bool()
        # has_objs: 1 x batch_size x observe_len -> batch_size x 1
        has_objs = torch.sum(
            objs_type_id.view(-1, *objs_type_id.shape[2 : ]), dim=-1, keepdim=True
        ).bool()
        # batch_size x observe_len -> batch_size x observe_len
        src_key_padding_mask_bool = torch.eq(src_key_padding_mask_bool, has_objs)

        obj_embedding_mask = torch.zeros(
            src_key_padding_mask_bool.shape[0], 1, device=src_key_padding_mask_bool.device
        ).bool()
        src_key_padding_mask_bool = torch.cat(
            [
                obj_embedding_mask, 
                src_key_padding_mask_bool
            ],
            dim=-1
        )
        
        # batch_size x observe_len x output_dim -> 1 x batch_size x observe_len x output_dim
        if self.use_src_key_padding_mask == True:
            _obj_observation_embedding = self.transformer_encoder(
                objs_state_embedding_reshaped, src_key_padding_mask=src_key_padding_mask_bool
            ).view(*batch_shape, *embedding_shape)
        else:
            _obj_observation_embedding = self.transformer_encoder(
                objs_state_embedding_reshaped
            ).view(*batch_shape, *embedding_shape)
        
        # 1 x batch_size x observe_len x output_dim -> 1 x batch_size x output_dim
        obj_observation_embedding = _obj_observation_embedding[..., 0, :]

        return obj_observation_embedding

class AgentEncoder(nn.Module):
    def __init__(
        self,
        agents_type_num: int = AGENT_TYPE_NUMBER,
        agents_pose_num: int = 6,
        actions_space_len: int = USED_ACTION_SPACE_LEN,
        obj_types_num: int = ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN, 
        output_dim: int = 128,
    ) -> None:
        super().__init__()

        output_dim = int(output_dim)
        assert(
            output_dim % 4 == 0
        ),'output_dim or symbolic observation encoder should be divisible by 4'
        
        _hidden_size = int(output_dim / 4)

        self.agent_type_encoder = nn.Sequential(        
            nn.Embedding(agents_type_num, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.obj_type_encoder = nn.Sequential(        
            nn.Embedding(obj_types_num, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.action_space_encoder = nn.Sequential(        
            nn.Embedding(actions_space_len, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.agent_pose_encoder = nn.Sequential(
            nn.Linear(agents_pose_num, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.action_success_encoder = nn.Sequential(
            nn.Linear(1, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(int(3 * _hidden_size), _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size * 2)
        )

        self.agent_encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(int(4 * _hidden_size), output_dim)
        )


    def forward(self, agents_type, agents_pose, agents_action):
        (
            action_name_id, held_obj_type_id, action_success
        ) = agents_action.split([1, 1, 1], dim=-1)
        
        agents_pose = agents_pose.to(torch.float)
        held_obj_type_id = held_obj_type_id.to(torch.long)
        action_name_id = action_name_id.to(torch.long)
        action_success = action_success.to(torch.float)

        agents_type_embedding = self.agent_type_encoder(agents_type)
        agents_pose_embedding = self.agent_pose_encoder(agents_pose)
        held_obj_type_embedding = self.obj_type_encoder(held_obj_type_id)
        action_name_embedding = self.action_space_encoder(action_name_id)
        action_success_embedding = self.action_success_encoder(action_success)

        agents_type_embedding = agents_type_embedding.squeeze(-2)
        action_name_embedding = action_name_embedding.squeeze(-2)
        held_obj_type_embedding = held_obj_type_embedding.squeeze(-2)

        _last_action_embedding = torch.cat(
            [
                held_obj_type_embedding, 
                action_name_embedding, 
                action_success_embedding
            ],
            dim=-1
        )
        last_action_embedding = self.action_encoder(_last_action_embedding)
        _agents_embedding = torch.cat(
            [
                agents_type_embedding,
                agents_pose_embedding,
                last_action_embedding
            ],
            dim=-1
        )
        agents_embedding = self.agent_encoder(_agents_embedding)

        return agents_embedding

class GoalEncoder(nn.Module):
    def __init__(
        self,
        goals_type_num: int = DETAILED_GOALS_TYPE_NUM,
        objs_type_num: int = ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN,
        obj_pos_coordinates_num: int = 3, 
        output_dim: int = 128,
    ) -> None:
        super().__init__()   

        output_dim = int(output_dim)
        _hidden_size = output_dim

        self.goal_type_encoder = nn.Sequential(
            nn.Embedding(goals_type_num, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.obj_type_encoder = nn.Sequential(        
            nn.Embedding(objs_type_num, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.obj_pos_encoder = nn.Sequential(         
            nn.Linear(obj_pos_coordinates_num, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.combine_goal_encoder = nn.Sequential(
            nn.Linear(_hidden_size * 3, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

    def forward(
        self, 
        goals_type_id,
        goal_objs_type_id, 
        goal_objs_pos
    ):
        goals_type_id = goals_type_id.to(torch.long)
        goal_objs_type_id = goal_objs_type_id.to(torch.long)
        goal_objs_pos = goal_objs_pos.to(torch.float)
       
        # batch_shape x observe_len -> batch_shape x observe_len x _hidden_size
        goals_type_embedding = self.goal_type_encoder(goals_type_id)
        # batch_shape x observe_len -> batch_shape x observe_len x _hidden_size
        objs_type_embedding = self.obj_type_encoder(goal_objs_type_id)
        # batch_shape x observe_len x 3 -> batch_shape x observe_len x _hidden_size
        objs_pos_embedding = self.obj_pos_encoder(goal_objs_pos)
        
        objs_type_pos_embedding_ = torch.cat(
            [
                goals_type_embedding,
                objs_type_embedding,
                objs_pos_embedding,
            ],
            dim=-1
        )

        objs_type_pos_embedding = self.combine_goal_encoder(objs_type_pos_embedding_)

        return objs_type_pos_embedding


class SubtaskEncoder(nn.Module):
    def __init__(
        self,
        objs_type_num: int = ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN,
        subtasks_type_num: int = SUBTASKS_TYPE_NUM,
        obj_pos_coordinates_num: int = 3, 
        output_dim: int = 128,
    ) -> None:
        super().__init__()

        output_dim = int(output_dim)
        _hidden_size = int(output_dim / 2)

        self.subtask_type_encoder = nn.Sequential(        
            nn.Embedding(subtasks_type_num, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.obj_type_encoder = nn.Sequential(        
            nn.Embedding(objs_type_num, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.obj_pos_encoder = nn.Sequential(         
            nn.Linear(obj_pos_coordinates_num, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.combine_subtask_decoder = nn.Sequential(
            nn.Linear(_hidden_size * 5, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

    def forward(
        self,
        subtask_type_id, 
        target_obj_type_id,
        target_obj_pos,
        receptacle_obj_type_id,
        receptacle_obj_pos
    ):
        subtask_type_id = subtask_type_id.to(torch.long)
        target_obj_type_id = target_obj_type_id.to(torch.long)
        target_obj_pos = target_obj_pos.to(torch.float)
        receptacle_obj_type_id = receptacle_obj_type_id.to(torch.long)
        receptacle_obj_pos = receptacle_obj_pos.to(torch.float)

        subtask_type_embedding = self.subtask_type_encoder(subtask_type_id)
        target_obj_type_embedding = self.obj_type_encoder(target_obj_type_id)
        target_obj_pos_embedding = self.obj_pos_encoder(target_obj_pos)
        receptacle_obj_type_embedding = self.obj_type_encoder(receptacle_obj_type_id)
        receptacle_obj_pos_embedding = self.obj_pos_encoder(receptacle_obj_pos)

        subtask_embedding_ = torch.cat(
            [
                subtask_type_embedding,
                target_obj_type_embedding,
                target_obj_pos_embedding,
                receptacle_obj_type_embedding,
                receptacle_obj_pos_embedding
            ],
            dim=-1
        )
        
        subtask_embedding = self.combine_subtask_decoder(subtask_embedding_)
        
        return subtask_embedding


if __name__ == '__main__':
    num_agents = 2
    agent_num_embed_len = 3
    idx = (torch.ones(num_agents, num_agents) - torch.diag(torch.ones(num_agents))).bool()
    # print(idx)
    seed = 2 ** 30 - 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    agent_num_embeddings = nn.Parameter(
        torch.rand(num_agents, agent_num_embed_len)
    )
    print(agent_num_embeddings)



    observation_1 = torch.rand([1, 1, 3, 4])
    observation_2 = torch.rand([1, 1, 3, 4])
    feature_shape = torch.Size([3, 4])
    batch_shape = torch.Size([1, 1])
    print(observation_1.shape[:-2])
    print(feature_shape)
    observation_1_reshaped = observation_1.view(*batch_shape, -1)
    observation_2_reshaped = observation_2.view(*batch_shape, -1)
    print(observation_1_reshaped.shape)
    print(observation_2_reshaped.shape)
    
    observation_1_reshaped = observation_1_reshaped.unsqueeze(-2)
    observation_2_reshaped = observation_2_reshaped.unsqueeze(-2)
    print(observation_1_reshaped.shape)
    print(observation_2_reshaped.shape)

    print(agent_num_embeddings.shape)
    agent_num_embeddings_reshaped = agent_num_embeddings.view(*batch_shape, *agent_num_embeddings.shape)
    print(agent_num_embeddings_reshaped.shape)

    observations = torch.cat(
        [observation_1_reshaped, observation_2_reshaped], 
        dim=-2
    )

    x = torch.cat(
        [agent_num_embeddings_reshaped, observations],
        dim=-1
    )
    print(x.shape)

    a = [[1, 2, 3], [4, 5, 6]]
    print(torch.tensor(a))


    actor = LinearActorHead(512, 85)
    x = torch.rand([1, 1, 2, 512])
    x_a = actor(x)
    print(x_a)
    print(x_a.log_probs_tensor)
    print(x_a.mode())
    print(x_a.mode().shape)
    
    multi_actor = MultiLinearActorHead(2, 512, 85)

    x = torch.rand([1, 1, 512])

    x_processed = multi_actor(x)
    print(x_processed.probs_tensor.shape)
    print(x_processed.mode())
    print(x_processed.mode().shape)
