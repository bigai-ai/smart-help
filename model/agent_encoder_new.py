import torch
import torch.nn as nn

from env.constants import AGENT_TYPE_NUMBER, USED_ACTION_SPACE_LEN, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN
from env.goal import GOAL_NUM, goal_list
from action import action_list


class AgentEncoder(nn.Module):
    def __init__(
        self,
        object_encoder, 
        actions_space_len: int = len(action_list),
        output_dim: int = 128,
    ) -> None:
        super().__init__()

        output_dim = int(output_dim)
        assert(
            output_dim % 4 == 0
        ),'output_dim or symbolic observation encoder should be divisible by 4'

        _hidden_size = int(output_dim / 4)

        self.obj_type_encoder = object_encoder

        self.action_space_encoder = nn.Sequential(        
            nn.Embedding(actions_space_len, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(64, _hidden_size * 4),
            nn.ReLU(),
            nn.Linear(_hidden_size * 4, _hidden_size * 4)
        )

        self.action_success_encoder = nn.Sequential(
            nn.Linear(1, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.agent_encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(224, output_dim)
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.rot_encoder = nn.Sequential(
            nn.Linear(3, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )


    def forward(self, agents_action, held_objs_type, agents_pos, agents_rot):
        (
            action_name_id, held_obj_type_id, action_success
        ) = agents_action.split([1, 1, 1], dim=-1)
        if torch.cuda.is_available():
            action_name_id = action_name_id.to(torch.long).cuda()
            held_objs_type = held_objs_type.to(torch.long).cuda()
            action_success = action_success.to(torch.float).cuda()
            agents_pos = agents_pos.to(torch.float).cuda()
            agents_rot = agents_rot.to(torch.float).cuda()
        else:
            action_name_id = action_name_id.to(torch.long)
            held_objs_type = held_objs_type.to(torch.long)
            action_success = action_success.to(torch.float)
            agents_pos = agents_pos.to(torch.float)
            agents_rot = agents_rot.to(torch.float)

        # print("for debug", agents_type)
        action_name_embedding = self.action_space_encoder(action_name_id)

        action_name_embedding = action_name_embedding.squeeze(-2)

        action_success_embedding = self.action_success_encoder(action_success)
        pos_embedding = self.pos_encoder(agents_pos)
        rot_embedding = self.rot_encoder(agents_rot)

        _last_action_embedding = torch.cat(
            [
                action_name_embedding, 
                action_success_embedding, 
            ],
            dim=-1
        )
        last_action_embedding = self.action_encoder(_last_action_embedding)
        # print("for debug", agents_type_embedding.shape, agents_pose_embedding.shape, last_action_embedding.shape)
        # print(agents_type, agents_pose, agents_action)
        held_objs_embedding = self.obj_type_encoder(held_objs_type)
        _agents_embedding = torch.cat(
            [
                last_action_embedding,
                held_objs_embedding, 
                pos_embedding, 
                rot_embedding, 
            ],
            dim=-1
        )
        agents_embedding = self.agent_encoder(_agents_embedding)

        return agents_embedding
