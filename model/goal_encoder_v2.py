import torch
import torch.nn as nn
from torch import tensor

from env.goal import GOAL_NUM
from env.constants import ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN

class GoalEncoder(nn.Module):
    def __init__(
        self,
        goals_type_num: int = GOAL_NUM,
        objs_type_num: int = ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN,
        obj_pos_coordinates_num: int = 3, 
        output_dim: int = 128,
    ) -> None:
        super().__init__()   

        output_dim = int(output_dim)
        _hidden_size = 32

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

        self.combine_goal_encoder = nn.Sequential(
            nn.Linear(_hidden_size * 3, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

    def forward(
        self, 
        goals_type_id,
        goal_objs_type_id_1, 
        goal_objs_type_id_2,
    ):
        goals_type_id = goals_type_id.to(torch.long)
        goal_objs_type_id_1 = goal_objs_type_id_1.to(torch.long)
        goal_objs_type_id_2 = goal_objs_type_id_2.to(torch.long)
       
        # batch_shape x observe_len -> batch_shape x observe_len x _hidden_size
        goals_type_embedding = self.goal_type_encoder(goals_type_id)
        # batch_shape x observe_len -> batch_shape x observe_len x _hidden_size
        objs_type_embedding_1 = self.obj_type_encoder(goal_objs_type_id_1)
        objs_type_embedding_2 = self.obj_type_encoder(goal_objs_type_id_2)
        
        objs_type_pos_embedding_ = torch.cat(
            [
                goals_type_embedding,
                objs_type_embedding_1,
                objs_type_embedding_2,
            ],
            dim=-1
        )

        objs_type_pos_embedding = self.combine_goal_encoder(objs_type_pos_embedding_)

        return objs_type_pos_embedding
