import ai2thor
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from typing import (
    Dict,
    Tuple,
    Any,
    Optional,
    Union,
    List,
    Sequence,
)
import networkx as nx
from torch.distributions.utils import lazy_property
import copy

import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.join(os.path.abspath('.'), '../../'))

from env.utils import (
    distance_3D, 
    find_closest_position, 
    compute_face_2_pos_rough_plus,
    obj_id_2_obj_type,
    find_parent_receptacle_plus
)

from env.constants import (
    AgentType,
    GRID_SIZE,
    USED_ACTION_NAME_2_ID,
    OPENABLE_RECEPTACLES_TYPE,
    STARTER_DATA_DIR
)

from model.navigator import NavigatorTHOR

AgentLocKeyType = Tuple[float, float, int, int]
AgentPoseKeyType = Tuple[float, float, int, int, bool]


class Expert:
    '''
    An agent which greedily attempts to complete a given household task.
    '''
    def __init__(
        self,
        goal_list: list, 
        controller: Controller,
        step_size: float = GRID_SIZE, 
        agent_id: int = 0,
        agent_num = 1, 
        env = None,
        boundary = None, 
    ) -> None:
        self.agent_id = agent_id
        self.agent_num = agent_num

        self.navigator = NavigatorTHOR(controller, step_size, agent_id=self.agent_id, agent_num=self.agent_num)

        self.expert_action_list: List[Optional[int]] = []
        self.goal_list = copy.deepcopy(goal_list)
        
        self.current_goal: list = None
        self.goal_pose = None
        self.current_path: List = []
        self.boundary = boundary
        # after Open_and_PickUp Action Sequence, Close Action need to be executed.

        # first restore the objectId to be closed, then execute PickUp Action, finally execute Close Action. 
        self.need_to_close: bool = False
        self.to_be_closed_obj_id: str = None

        self._last_held_object_name: Optional[str] = None
        self._last_to_interact_object_pose: Optional[Dict[str, Any]] = None

        self.navigator.on_reset()
        # self.update()

        self.env = env

    @property
    def reachable_positions(self) -> List:
        '''
        return current reachable positions 
        '''
        return self.navigator.reachable_positions

    @property
    def expert_action(self) -> int:
        '''
        Get the current greedy expert action.

        # Returns An integer specifying the expert action in the current
        state. This corresponds to the order of actions in
        `self.task.action_names()`. For this action to be available the
        `update` function must be called after every step.
        '''
       
        return self.expert_action_list[-1]

    def update(self, _agent_state, _scene_state, except_pos):
        '''
        Update the expert with the last action taken and whether or not that 
        action succeeded.

        _agent_state = {
            'agent_id': agent_id, 'agent_type': agent_type, 
            'x_pos': x_pos, 'y_pos': y_pos,'z_pos': z_pos,
            'rotation': rotation, 'horizon': horizon, 'standing': standing,
            'held_obj': held_obj, 'action_name': action_name, 
            'action_success': action_success
        }
        '''
        # print("for debug expert goal", self.goal_list)
        # except_pos = []
        
        if len(self.goal_list) != 0:
            current_goal = self.goal_list[0]
        else:
            self.expert_action_list.append(['Wait', 0])
            return
        
        if current_goal[0] == 'Wait':
            self.expert_action_list.append(['Wait', 0])
            return
        
        current_pose = (
                round(_agent_state['x_pos'], 2), round(_agent_state['y_pos'], 2), round(_agent_state['z_pos'], 2), 
                round(_agent_state['rotation'], 0) % 360, 
                round(_agent_state['horizon'], 0) % 360,
                _agent_state['standing']
            )

        if self.current_goal != current_goal:
            # print("for debug expert goal, GOAL CHANGE!!!")
            self.current_goal = current_goal

            self.current_path = self.find_path(self.current_goal, _scene_state, current_pose, except_pos, _agent_state)
            
        if self.current_path is None:
            self.expert_action_list.append(['Wait', 0])
            return

        # print("for debug expert", current_pose[ : 2], self.current_path[0])
        # print("for debug expert", self.current_path)

        # Action: [action_name, obj_id=None]
        # print("for debug Expert", current_pose, self.goal_pose)
        if current_pose == self.goal_pose:
            goal = self.current_goal[0]
            
            if current_goal[0] in {'PickUp', 'Open', 'Close', 'Slice', 'ToggleOn', 'ToggleOff'}:
                action_type = current_goal[0]
                obj_id = current_goal[1]
                action_name = [action_type, obj_id]
                
                if action_type in {'Open', 'Close'}:
                    self.to_be_closed_obj_id = obj_id
                    self.need_to_close = True
            elif current_goal[0] == 'Put':
                action_type = 'Put'
                obj_id = current_goal[1]
                action_name = [action_type, obj_id]
            elif current_goal[0] == 'PutIn':
                action_type = 'Open' if _scene_state[current_goal[1]]['is_open'] == False else 'Put'
                obj_id = current_goal[1]
                action_name = [action_type, obj_id]
            else:
                raise NotImplementedError
        elif current_pose[ : 3] == self.goal_pose[ : 3]:
            if current_pose[3] < self.goal_pose[3]:
                action_name = ['RotateRight', 0]
            elif current_pose[3] > self.goal_pose[3]:
                action_name = ['RotateLeft', 0]
            elif current_pose[4] < self.goal_pose[4]:
                action_name = ['LookDown', 0]
            elif current_pose[4] > self.goal_pose[4]:
                action_name = ['LookUp', 0]
            elif current_pose[5] != self.goal_pose[5] and self.goal_pose[5] == True:
                action_name = ['Stand', 0]
            elif current_pose[5] != self.goal_pose[5] and self.goal_pose[5] == False:
                action_name = ['Crouch', 0]
            else:
                raise NotImplementedError
        elif current_pose[ : 2] == self.current_path[0]:
            # print("for debug A*", self.current_path)
            action_name = self.path_to_action(current_pose, self.current_path[1])

            # if action_name == None: # graph is dynamic, so this is possible
            #     self.current_path = self.navigator.shortest_path(
            #         current_pose, self.goal_pose
            #     )
            #     action_name = self.navigator.shortest_pose_path_next_action(
            #         current_pose, self.current_path[1]
            #     )
            if action_name[0] == 'MoveAhead':
                self.current_path.pop(0)
        else:
            action_name = self.current_path
            # try:
            # self.current_path = self.find_path(self.current_goal, _scene_state, current_pose, except_pos)
            # if self.current_path is None:
            #     action_name = ['Wait', 0]
            # elif len(self.current_path) < 2:
            #     action_name = ['Wait', 0]
            # else:
            #     action_name = self.path_to_action(current_pose, self.current_path[1])
            #     if action_name[0] == 'MoveAhead':
            #         self.current_path.pop(0)
            # except:
            #     action_name = 'Wait'
        self.expert_action_list.append(action_name)
        # print("for debug expert", action_name)

    def path_to_action(self, current_pose, next_pos):
        # print("for debug A*", current_pose, next_pos)
        x_ofst, z_ofst = ((next_pos[0] - current_pose[0]), (next_pos[1] - current_pose[1]))
        if z_ofst > 0:
            rotation = 0
        elif x_ofst > 0:
            rotation = 90
        elif z_ofst < 0:
            rotation = 180
        elif x_ofst < 0:
            rotation = 270
        else:
            raise NotImplementedError
        
        if current_pose[2] < rotation:
            action_name = ['RotateRight', 0]
        elif current_pose[2] > rotation:
            action_name = ['RotateLeft', 0]
        else:
            action_name = ['MoveAhead', 0]

        return action_name

    def find_path(self, goal, _scene_state, current_pose, except_pos, _agent_state=None):
        if goal[0].startswith(
                (
                    'PickUp', 'Open', 'Close', 'Slice', 'Toggle'
                )
            ):
            # goal = 'PickUp=obj_id'
            target_obj_id = self.env.find_id(goal[1])
        elif goal[0].startswith('Put'):
            # goal = 'PutIn=obj_id=recep_id' or 'PutOn=obj_id=recep_id'
            target_obj_id = self.env.find_id(goal[1])
        else:
            print(goal)
            raise NotImplementedError
        
        obj_index = self.env.find_index(target_obj_id)

        # _goal_pose = compute_face_2_pos_rough_plus(
        #     target_obj_id, _scene_state[obj_index]['obj_pos'], 
        #     self.reachable_positions, except_pos
        # )

        if target_obj_id is False:
            return None

        print(self.boundary)
        _goal_pose = compute_face_2_pos_rough_plus(
            target_obj_id, self.env.metadata['objects'][obj_index]['position'], 
            self.reachable_positions, except_pos, 
            self.boundary
        )


        self.goal_pose = (
            round(_goal_pose['x'], 2), round(_goal_pose['y'], 2), round(_goal_pose['z'], 2), 
            round(_goal_pose['rotation'], 0) % 360, 
            round(_goal_pose['horizon'], 0) % 360,
            _goal_pose['standing']
        )

        return ["Teleport", self.goal_pose, goal[1]]

        return self.navigator.shortest_path(
            current_pose, self.goal_pose, except_pos
        )
    
        

    def next_goal(self):
        # print("for debug before next goal", self.goal_list)
        _next_goal = self.goal_list.pop(0)
        # print("for debug after next goal", self.goal_list)
        return _next_goal

