import torch
import torch.nn as nn

from env.constants import AGENT_TYPE_NUMBER, USED_ACTION_SPACE_LEN, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN


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
        
        _hidden_size = int(output_dim / 2)

        self.agent_type_encoder = nn.Sequential(        
            nn.Embedding(agents_type_num, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        self.obj_type_encoder = nn.Sequential(        
            nn.Embedding(obj_types_num, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.action_space_encoder = nn.Sequential(        
            nn.Embedding(actions_space_len, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.action_success_encoder = nn.Sequential(
            nn.Embedding(3, 8),
            nn.ReLU(),
            nn.Linear(8, 8)
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.agent_encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(160, output_dim)
        )


    def forward(self, agents_type, agents_action):
        (
            action_name_id, held_obj_type_id, action_success
        ) = agents_action.split([1, 1, 1], dim=-1)
        
        held_obj_type_id = held_obj_type_id.to(torch.long)
        action_name_id = action_name_id.to(torch.long)
        action_success = action_success.to(torch.long)

        # print("for debug", agents_type)
        agents_type_embedding = self.agent_type_encoder(agents_type)
        held_obj_type_embedding = self.obj_type_encoder(held_obj_type_id)
        action_name_embedding = self.action_space_encoder(action_name_id)
        action_success_embedding = self.action_success_encoder(action_success)

        agents_type_embedding = agents_type_embedding.squeeze(-2)
        action_name_embedding = action_name_embedding.squeeze(-2)
        held_obj_type_embedding = held_obj_type_embedding.squeeze(-2)

        if len(action_success_embedding.shape) == 3:
            action_success_embedding = action_success_embedding.squeeze(1)

        _last_action_embedding = torch.cat(
            [
                held_obj_type_embedding, 
                action_name_embedding, 
                action_success_embedding
            ],
            dim=-1
        )
        last_action_embedding = self.action_encoder(_last_action_embedding)
        # print("for debug", agents_type_embedding.shape, agents_pose_embedding.shape, last_action_embedding.shape)
        # print(agents_type, agents_pose, agents_action)
        if len(agents_type_embedding.shape) == 1:
            agents_type_embedding = agents_type_embedding.unsqueeze(0)
        # agents_type_embedding = agents_type_embedding.unsqueeze(0)
        _agents_embedding = torch.cat(
            [
                agents_type_embedding,
                last_action_embedding
            ],
            dim=-1
        )
        agents_embedding = self.agent_encoder(_agents_embedding)

        return agents_embedding
