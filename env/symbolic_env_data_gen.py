from gymnasium import spaces
from ai2thor.controller import Controller
import random
from gymnasium import Env
from env.type import sample_type

from env.action_space import MyActionSpace
import numpy as np
from env.agent import Agent
import json
import os
import copy
from action import action_list
import action as actions
from utils import find_parent_receptacle_plus
from env.subtask import subtask_list
from env.goal import goal_list, GOAL_NUM
import math
from env.parse import *
import datetime
from env.task import sample_task
from env.utils import find_parent_receptacle_plus
from torchvision import transforms
from env.utils import *
from PIL import Image


class SymbolicEnvGen(Env):
    def __init__(self, config):
        Env.__init__(self)

        '''These variables are constant and stationary during runtime'''
        # basic settings
        self._agent_num = config['agents_num']
        self._agent_ids = list(range(self._agent_num))
        self._agent_type = config['agents_type']
        # main_agent: 1
        # helper: 0

        self.scene_name = config['controller_kwargs']["scene"]
        self.agents = []
        for _ in range(self._agent_num):
            self.agents.append(Agent())

        self.task = copy.deepcopy(config['task'])
        for i in range(1, len(self.task)):
            if isinstance(self.task[i], str):
                if not self.task[i].find('|') == -1:
                    self.task[i] = self.task[i][:self.task[i].index('|')]
        
        # share reward with helper
        self.goal_complish_reward = 0

        if config['controller_kwargs'] is None:
            # a parameter that can force elements out
            self.controller = Controller(
                agentCount=2,
                scene=self.scene_name,
                platform="CloudRendering",
                # image modalities
                renderDepthImage=True,
                renderInstanceSegmentation=True,
                visibilityDistance=6,
                quality='Very Low',
            )
        else:
            self.controller = Controller(
                **config['controller_kwargs']
            )

        self.update_dict()
        self.subtask_list = parse_task(self.task, self)
        self.goal_list = self.parse()
        self.goal_num = len(self.subtask_list)

        self.need_help = self.need_help_goal(self.subtask_list)
        
        self.build_maintain_corresponding_dict()
        self.build_boundary()

        # step to terminate
        self.step_count = 0

        self.observation_space = spaces.Box(low=-360, high=360, shape=(31 + self._agent_num, self.obs_width))
        
        self.expert_goal_list = []
        self.action_space = MyActionSpace(low=np.asarray((0, 0)),
                                          high=np.asarray((GOAL_NUM - 1, len(self.object_index2name) - 1)),
                                          dtype=int)
        
        self.last_report_time = datetime.datetime.now()
        self.fps_count = 0

        # for metric
        self.helper_finish_goal_num = 0
        self.helper_finish_required_action_num = 0
        self.main_finish_goal_num = 0
        self.finish_goal_num = 0
        self.helper_action_num = 0
        self.helper_finish_necc_goal_num = 0

    def build_boundary(self):
        corner_points = self.controller.last_event.events[0].metadata['sceneBounds']['cornerPoints']
        x_min = 0
        x_max = 0
        z_min = 0
        z_max = 0
        for point in corner_points:
            x_min = min(x_min, point[0])
            x_max = max(x_max, point[0])
            z_min = min(z_min, point[2])
            z_max = max(z_max, point[2])
        self.boundary = [x_min, x_max, z_min, z_max]

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            ])

    # single or multi-agent will have different way to call metadata
    def metadata(self, agent_id=0):
        if self._agent_num > 1:
            return self.controller.last_event.events[agent_id].metadata
        else:
            return self.controller.last_event.metadata

    def build_maintain_corresponding_dict(self):
        if os.path.exists("./object_index2name.json") and os.path.getsize("./object_index2name.json") > 0:
            # JSON 会将int key 转变为 str
            tmp_dict = json.load(open("./object_index2name.json", "r"))
            self.object_index2name = {}
            for key, value in tmp_dict.items():
                self.object_index2name[int(key)] = value
        else:
            self.object_index2name = {0: "None"}
        if os.path.exists("./object_name2index.json") and os.path.getsize("./object_name2index.json") > 0:
            self.object_name2index = json.load(open("./object_name2index.json", "r"))
        else:
            self.object_name2index = {"None": 0}
        for obj in self.metadata()['objects']:
            obj_id = obj['objectId']
            obj_name = obj_id[:obj_id.index('|')]
            if self.object_name2index.get(obj_name) is None:
                # new object
                new_index = len(self.object_index2name)
                self.object_index2name[new_index] = obj_name
                self.object_name2index[obj_name] = new_index
        json.dump(self.object_index2name, open("./object_index2name.json", "w"))
        json.dump(self.object_name2index, open("./object_name2index.json", "w"))

    def update_dict(self):
        # init id2index for fast operation
        # the objectId is strange
        self.object_id2index = {}
        self.object_index2id = {}
        self._object_ids = []
        if self.metadata() is not None:
            objects = self.metadata()['objects']
            for i, object in enumerate(objects):
                self.object_id2index[object['objectId']] = i
                self.object_index2id[i] = object['objectId']
                self._object_ids.append(object['objectId'])

    def reset(self, *, seed=None, options=None):
        i = random.randint(1, 30)
        index = i
        # exclude scenario
        while i in []:
            i = random.randint(1, 30)
        self.scene = "FloorPlan{}".format(index)

        if self._agent_num > 1:
            # self._agent_type[1] = [1, 1, 1, 1, 1, 1]
            self._agent_type[1] = sample_type()

        self.controller.reset(self.scene, renderInstanceSegmentation=True)

        self.step_count = 0
        for agent in self.agents:
            agent.reset()
        # update parsing
        self.task = sample_task()
        self.update_dict()
        self.subtask_list = parse_task(self.task, self)
        self.need_help = self.need_help_goal(self.subtask_list)
        self.goal_list = self.parse()
        self.goal_num = len(self.subtask_list)
        # update dict
        obs = self.get_obs(0)
        image = self.controller.last_event.cv2img
        # print("for debug generator", image.shape)
        obs = [obs, image]
        # update boundary
        self.build_boundary()

        self.helper_finish_goal_num = 0
        self.helper_finish_required_action_num = 0
        self.main_finish_goal_num = 0
        self.finish_goal_num = 0
        self.helper_action_num = 0
        self.helper_finish_necc_goal_num = 0

        return obs, {}
        # default main agent is the agent 0

    def find_index(self, obj_id):
        obj_index = None
        if isinstance(obj_id, int):
            obj_index = obj_id
        else:
            if obj_id.find('|') == -1:
                for i, obj in enumerate(self.controller.last_event.metadata['objects']):
                    if obj['objectId'][:obj['objectId'].find('|')] == obj_id:
                        obj_index = i
                pass
            else:
                obj_index = self.object_id2index[obj_id]
        return obj_index

    def find_id(self, obj_index):
        obj_id = False
        if isinstance(obj_index, int):
            obj_id = self.object_index2id[obj_index]
        elif isinstance(obj_index, np.ndarray):
            obj_id = self.object_index2id[int(obj_index)]
        else:
            if obj_index.find('|') == -1:
                for i, obj in enumerate(self.controller.last_event.metadata['objects']):
                    if obj['objectId'][:obj['objectId'].find('|')] == obj_index:
                        obj_id = obj['objectId']
                pass
            else:
                obj_id = obj_index
        return obj_id

    # more fast
    def _get_reward_and_done(self, agent_id=None, action=None, goal=None):
        # print(self.goal_list)
        reward = -0.01
        # wait will not cause
        if action is not None and action['action'] == 0:
            reward = 0

        # This kind of removal will hinder bug
        to_remove = []
        # subtask can be achieved with any sequence
        for subtask in self.subtask_list:
            if self.check_sub_task(subtask, agent_id=agent_id, action=action):
                self.finish_goal_num += 1
                if agent_id == 0:
                    self.helper_finish_goal_num += 1
                    if subtask[0] in ["In", "On"]:
                        self.helper_finish_required_action_num += 2
                    else:
                        self.helper_finish_required_action_num += 1
                    if self.need_help_goal([subtask]):
                        self.helper_finish_necc_goal_num += 1
                        reward += 20
                    else:
                        reward -= 10
                if agent_id == 1:
                    self.main_finish_goal_num += 1
                    reward += 20
                    self.goal_complish_reward += 20
                if agent_id == 0:
                    print("Subtask", subtask, "Finish! by ", agent_id, "AgentType:", self._agent_type[1])
                to_remove.append(subtask)
            else:
                break

        remove_len = len(to_remove)
        for i in range(remove_len):
            self.subtask_list.remove(to_remove[remove_len - i - 1])
            # print("for debug env", "remove", f"Agent{agent_id}", to_remove[remove_len - i - 1])

        is_finish = self.check_task()
        return reward, is_finish

    def parse(self):
        sub_task_list = self.subtask_list
        goal_list = []
        for sub_task in sub_task_list:
            goal_list += parse_sub_task(sub_task)
        return goal_list

    def check_goal(self, goal, agent_id=0, action=None):
        # from the perspective of main agent
        metadata = self.metadata(1)
        if goal[0] == 'PickUp':
            obj_index = self.find_index(goal[1])
            if obj_index is None:
                return False
            # use isPickedUp flag
            if metadata['objects'][obj_index]['isPickedUp']:
                return True
        elif goal[0] == 'Put':
            obj_index_1 = self.find_index(goal[1])
            obj_index_2 = self.find_index(goal[2])
            if obj_index_1 is None or obj_index_2 is None:
                return False
            # use parentReceptacles
            tar_id = metadata['objects'][obj_index_1]['objectId']
            receptacleObjectIds = metadata['objects'][obj_index_2]['receptacleObjectIds']
            if receptacleObjectIds is not None and len(receptacleObjectIds) > 0:
                if (self.agents[agent_id].pick_up_obj_id is None) and (action is not None) and (action['action'] == action_list.index("Put")):
                    if action["tar_index"] == self.object_name2index[goal[2]]:
                        for id in receptacleObjectIds:
                            if id == tar_id:
                                return True
        elif goal[0] == 'ToggleOn':
            obj_index = self.find_index(goal[1])
            if obj_index is None:
                return False
            if metadata['objects'][obj_index]['isToggled'] is True:
                return True
        elif goal[0] == 'ToggleOff':
            obj_index = self.find_index(goal[1])
            if obj_index is None:
                return False
            if metadata['objects'][obj_index]['isToggled'] is False:
                return True
        elif goal[0] == 'Open':
            obj_index = self.find_index(goal[1])
            if obj_index is None:
                return False
            if metadata['objects'][obj_index]['isOpen']:
                return True
        elif goal[0] == 'Close':
            obj_index = self.find_index(goal[1])
            if obj_index is None:
                return False
            if metadata['objects'][obj_index]['isOpen'] is False:
                return True
        return False

    def check_task(self):
        return len(self.subtask_list) == 0

    # Check Sub_Task with state
    def check_sub_task(self, subtask, agent_id=1, action=None):
        if self._agent_num > 1:
            metadata = self.controller.last_event.events[1].metadata
        else:
            metadata = self.controller.last_event.metadata
        if subtask[0] == 'Get':
            obj_index = self.find_index(subtask[1])
            # use isPickedUp flag
            if metadata['objects'][obj_index]['isPickedUp']:
                return True
        elif subtask[0] == 'On' or subtask[0] == 'In':
            obj_index_1 = self.find_index(subtask[1])
            obj_index_2 = self.find_index(subtask[2])
            # use parentReceptacles
            tar_id = metadata['objects'][obj_index_1]['objectId']
            receptacleObjectIds = metadata['objects'][obj_index_2]['receptacleObjectIds']
            if receptacleObjectIds is not None and len(receptacleObjectIds) > 0:
                if (self.agents[agent_id].pick_up_obj_id is None) and (action is not None) and (action['action'] == action_list.index("Put")):
                    if action["tar_index"] == self.object_name2index[subtask[2]]:
                        for id in receptacleObjectIds:
                            if id == tar_id:
                                # print("for debug check subtask", "Agent", agent_id, tar_id, receptacleObjectIds, action)
                                return True
        elif subtask[0] == 'Open':
            obj_index = self.find_index(subtask[1])
            if metadata['objects'][obj_index]['isOpen']:
                return True
        elif subtask[0] == 'Close':
            obj_index = self.find_index(subtask[1])
            if metadata['objects'][obj_index]['isOpen'] is False:
                return True
        elif subtask[0] == 'ToggleOn':
            obj_index = self.find_index(subtask[1])
            if metadata['objects'][obj_index]['isToggled'] is True:
                return True
        elif subtask[0] == 'ToggleOff':
            obj_index = self.find_index(subtask[1])
            if metadata['objects'][obj_index]['isToggled'] is False:
                return True
        elif subtask[0] == 'Cook':
            obj_index = self.find_index(subtask[1])
            if metadata['objects'][obj_index]['isCooked']:
                return True
        return False

    # return False if the action is invalid
    def execuate_action(self, action, agent_id=0):
        # print("for debug execuate_action", agent_id, action)
        if action is None:
            return False
        # 0 means failure
        if not self.is_action_valid(action, agent_id=agent_id):
            self.agents[agent_id].action_history.append([action["action"], action["tar_index"], 0])
            return False
        action_name = action_list[int(action["action"])]
        action_func = getattr(actions, action_name)
        if action_name == "Teleport":
            event = action_func(self, action["teleport_pose"], agent_id)
            self.agents[agent_id].action_history.append([action["action"], action["tar_index"], 1])
        else:
            obj_id = self.find_id(self.object_index2name[int(action['tar_index'])])
            event = action_func(self, obj_id, agent_id)
        # print("for debug execuate_action", "Action Finish")
        if event is False:
            self.agents[agent_id].action_history.append([action["action"], action["tar_index"], 0])
        else:
            self.agents[agent_id].action_history.append([action["action"], action["tar_index"], 1])
            # print("for debug execuate_action", event.metadata['errorMessage'])
        return event

    # The ability of agent
    def is_action_valid(self, action, agent_id=0):
        # check whether the action is valid before parse
        # type: [height, weight, open, close, toggle_on, toggle_off]
        action_index = action["action"]
        action_name = actions.action_list[int(action_index)]
        if action_name == "Teleport":
            return True
        action_tar = int(action["tar_index"])
        if action_name in ['PickUp', 'Put', 'Open', 'Close', 'ToggleOn', 'ToggleOff', 'Slice']:
            # if action_name == "PickUp" and self._agent_type[agent_id].value == AgentType.AGENT_WITH_PICKUP_ISSUES.value:
            #     return False
            if action_name == "Open" and self._agent_type[agent_id][2] < 0.5:
                return False
            elif action_name == "Close" and self._agent_type[agent_id][3] < 0.5:
                return False
            elif action_name == "ToggleOn" and self._agent_type[agent_id][4] < 0.5:
                return False
            elif action_name == "ToggleOff" and self._agent_type[agent_id][5] < 0.5:
                return False
            elif action_name == "PickUp":
                obj_id = self.find_id(self.object_index2name[action_tar])
                if obj_id is False:
                    return False
                obj_index = self.find_index(obj_id=obj_id)
                if obj_index is None:
                    return False
                # height check
                if self.metadata(agent_id=agent_id)['objects'][obj_index]['position']['y'] > (self._agent_type[agent_id][0] + 0.5):
                    return False
                # mass check
                if self.metadata(agent_id)['objects'][obj_index]['mass'] > self._agent_type[agent_id][1]:
                    return False
            obj_id = self.find_id(self.object_index2name[action_tar])
            if obj_id is False:
                return False
            if action_name in ['Open', 'Close']:
                if self.find_index(obj_id) is None:
                    return False
                if not self.metadata(1)['objects'][self.find_index(obj_id)]['openable']:
                    return False
            if action_name in ['ToggleOn', 'ToggleOff'] and self.metadata(1)['objects'][self.object_id2index[obj_id]]['toggleable']:
                if self.find_index(obj_id) is None:
                    return False
                if not self.metadata(1)['objects'][self.find_index(obj_id)]['toggleable']:
                    return False
        return True

    def goal_to_str(self, goal):
        for i in range(len(goal)):
            goal[i] = int(goal[i])
        return str(goal[0]) + '_' + str(goal[1])

    @property
    def obs_width(self):
        return 17

    def reachable_positions(self, agent_id=0):
        return self.controller.step(
            action='GetReachablePositions',
            agentId=agent_id
        ).metadata['actionReturn']

    # Here like full observation
    def get_agent_obs(self, agent_id):
        agent_obs = np.zeros(shape=(self._agent_num, self.obs_width))
        for i, agent in enumerate(self.agents):
            # 
            agent_dict = self.metadata(agent_id)['agent']
            agent_obs[i, 0] = agent_dict['position']['x']
            agent_obs[i, 1] = agent_dict['position']['y']
            agent_obs[i, 2] = agent_dict['position']['z']
            agent_obs[i, 3] = agent_dict['rotation']['x']
            agent_obs[i, 4] = agent_dict['rotation']['y']
            agent_obs[i, 5] = agent_dict['rotation']['z']
            if len(self.agents[i].action_history) > 0:
                agent_obs[i, 6:9] = np.asarray([int(agent.action_history[-1][0]), int(agent.action_history[-1][1]), int(agent.action_history[-1][2]) + 1], dtype=np.float32) # for embedding
            else:
                agent_obs[i, 6:9] = np.asarray([0, 0, 0], dtype=np.float32)
            if agent.pick_up_obj_id == None:
                agent_obs[i, 9] = 0
            else:
                agent_obs[i, 9] = self.object_name2index[agent.pick_up_obj_id[:agent.pick_up_obj_id.index('|')]]

            # 计算相对角度

            relative_angle = get_angle_between_agents(self.metadata(0)['agent'], self.metadata(1)['agent'])

            # 判断agent2是否在agent1的视野内（假设视野为120度）
            if abs(relative_angle) <= 60:  # 120度视野的一半
                agent_obs[i, 10] = 1
            else:
                agent_obs[i, 10] = 0

            agent_obs[i, 11:17] = self._agent_type[i]
        return agent_obs

    def get_symbolic_observations(self, agent_id: int):
        observation_state = np.zeros((30, self.obs_width))
        # a list of object_dict
        objects = self.metadata(agent_id)['objects']
        adding_index = 0
        for obj in objects:
            obj_id = obj['objectId']
            obj_name = obj['objectId'][:obj['objectId'].index('|')]
            obj_index = self.object_name2index[obj_name]
            if obj['axisAlignedBoundingBox']['cornerPoints'] == None:
                is_existent = False
                continue
            else:
                is_existent = True
            if not is_existent:
                continue
            obj_properties = [obj['isPickedUp'], obj['isOpen'], obj['isCooked'], obj['isToggled']]

            parent_receptacle = find_parent_receptacle_plus(
                obj_id, obj['parentReceptacles'],
                objects, self.object_id2index
            )

            obj_height = obj['position']['y']
            obj_weight = obj['mass']

            obj_pos = obj['position']
            obj_distance = obj['distance']

            is_visibles = []
            for i in range(self._agent_num):
                is_visibles.append(
                    self.metadata(i)['objects'][self.find_index(obj_id)]['visible'])

            if self._agent_num == 1:
                if is_visibles is False:
                    continue
            else:
                if is_visibles[agent_id] is False:
                    continue
            tmp = [obj_index]

            if parent_receptacle is None:
                tmp.append(0)
            else:
                tmp.append(self.object_name2index[parent_receptacle[:parent_receptacle.index("|")]])

            for property in obj_properties:
                tmp.append(property)

            for is_visible in is_visibles:
                tmp.append(int(is_visible))

            tmp.append(obj_height)
            tmp.append(obj_weight)      

            tmp.append(obj_pos['x'])
            tmp.append(obj_pos['y'])
            tmp.append(obj_pos['z'])

            tmp.append(obj_distance)

            tmp.append(0)
            tmp.append(0)

            tmp.append(1)

            observation_state[adding_index, :] = np.asarray(tmp, dtype=np.float32)
            adding_index += 1
            if adding_index >= 30:
                break
        return observation_state

    def get_subtask_obs(self, agent_id=0):
            subtask_obs = np.zeros(shape=(1, self.obs_width))
            if len(self.subtask_list) == 0:
                subtask_obs[0, :9] = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            else:
                subtask = self.subtask_list[0]
                subtask_index = subtask_list.index(subtask[0])
                if subtask[0] in ['In', "On"]:
                    subtask_obj1_index = self.object_name2index[subtask[1]]
                    subtask_obj2_index = self.object_name2index[subtask[2]]
                    subtask_obs[0, :3] = np.asarray([subtask_index, subtask_obj1_index, subtask_obj2_index], dtype=np.float32)
                else:
                    subtask_obj1_index = self.object_name2index[subtask[1]]
                    subtask_obj2_index = 0
                    subtask_obs[0, :3] = np.asarray([subtask_index, subtask_obj1_index, subtask_obj2_index], dtype=np.float32)
            return subtask_obs.astype(np.float32)

    def get_obs(self, agent_id):
        obj_obs = self.get_symbolic_observations(agent_id=agent_id)
        goal_obs = self.get_subtask_obs()
        obs = np.concatenate((obj_obs, goal_obs), axis=0)
        agent_obs = self.get_agent_obs(agent_id=agent_id)
        obs = np.concatenate((obs, agent_obs), axis=0)
        return obs

    def step(self, goal):
        cur_datetime = datetime.datetime.now()
        if (cur_datetime - self.last_report_time).total_seconds() < 10:
            self.fps_count += 1
        else:
            print("-----------report fps over 10s-----------", self.fps_count)
            self.fps_count = 0
            self.last_report_time = cur_datetime
        if self._agent_num > 1:

            # main_agent id: 1
            if len(self.goal_list) > 0:
                while self.check_goal(self.goal_list[0], agent_id=1):
                    self.goal_list.pop(0)
                    if len(self.goal_list) == 0:
                        break

            if len(self.goal_list) > 0:
                main_goal = self.goal_list[0]
            else:
                main_goal = ["Wait", "None"]

            if main_goal[0] == "Put":
                main_goal = copy.deepcopy(main_goal)
                main_goal.pop(1)
            # print(main_goal)
            self.execuate_goal(main_goal, agent_id=1)

            # update goal
            self._get_reward_and_done(agent_id=1)

        # if len(self.goal_list) != 0 and self.goal_list[0][0] == "ToggleOn" and self._agent_type[1].value == AgentType.AGENT_WITH_TOGGLE_ISSUES.value:
        #     goal = {"goal": 3, "tar_index": 24}
        # elif len(self.goal_list) != 0 and self.goal_list[0][0] == "Open" and self._agent_type[1].value == AgentType.AGENT_WITH_OPEN_ISSUES.value:
        #     goal = {"goal": 5, "tar_index": 24}
        # elif len(self.goal_list) != 0 and self.goal_list[0][0] == "Close" and self._agent_type[1].value == AgentType.AGENT_WITH_CLOSE_ISSUE.value:
        #     goal = {"goal": 6, "tar_index": 24}

        rew_accumulate = self.execuate_goal(self.goal_dict_to_list(goal), agent_id=0)

        rew, terminated = self._get_reward_and_done(agent_id=0)
        truncated = False
        info = {}

        rew = rew_accumulate
        rew -= 0.02

        obs = self.get_obs(0)
        image = self.controller.last_event.cv2img
        image = Image.fromarray(image)
        img_tensor = self.preprocess(image)
        img_np = np.asarray(img_tensor)        
        img_np = img_np.transpose((2, 0, 1))
        obs = [obs, img_np]

        self.step_count = self.step_count + 1
        if self.step_count >= 200:
            terminated = True
            rew -= 20
        return obs, rew, terminated, truncated, info

    def execuate_goal(self, goal, agent_id=0):
        # if agent_id == 1:
        #     print("for debug env", agent_id, goal)
        if goal[0] == "Wait":
            object_pos = self.metadata(1)['agent']['position']
            teleport_pos = self.metadata(agent_id=agent_id)['agent']['position']
            yaw_appropriate = round(
                        (np.arctan2(-teleport_pos['x'] + object_pos['x'], -teleport_pos['z'] + object_pos['z']) / np.pi * 180) % 360,
                        1)
            yaw_modified = 90.0 * round(yaw_appropriate / 90.0)

            pitch = round(
                (np.arctan2(object_pos['y'] - teleport_pos['y'], np.sqrt((object_pos['x'] - teleport_pos['x'])**2 + (object_pos['z'] - teleport_pos['z'])**2)) / np.pi * 180) % 360,
                1
            )
            pitch_modified = 90.0 * round(pitch / 90.0)

            standing = True
            # crouch
            if (round(object_pos['y'], 1) < 0.7):
                standing = False
                horizon = 30
            # look down by 30 degrees
            elif (round(object_pos['y'], 1) < 1.2):
                horizon = 30
            # look up by 30 degrees
            elif (round(object_pos['y'], 1) > 1.8):
                horizon = -30
            else:
                horizon = 0

            if teleport_pos is not None:
                teleport_pos = (teleport_pos['x'], teleport_pos['y'], teleport_pos['z'], yaw_modified, standing, horizon, pitch_modified)

            self.execuate_action(
                {"action": action_list.index("Teleport"), "tar_index": 0, "teleport_pose": teleport_pos},
                agent_id=agent_id)
            
            self.update_dict()

            return 0
        else:
            self.helper_action_num += 1
            object_name = goal[1]
            if object_name is None:
                return -0.3
            object_index = self.object_name2index[object_name]
            if self.find_index(object_name) is None:
                return -0.3
            object_pos = self.metadata(agent_id=agent_id)['objects'][self.find_index(object_name)]['position']
            agent_pos = self.metadata(agent_id=agent_id)['agent']['position']

            reachable_pos_list = self.controller.step(action="GetReachablePositions", agentId=agent_id).metadata[
                "actionReturn"]

            mini_distance = 1000
            action_name = None
            teleport_pos = None

            if reachable_pos_list is None or len(reachable_pos_list) == 0:
                action_name = 'Wait'
            else:
                action_name = "Teleport"
                for pos_dict in reachable_pos_list:
                    distance = math.sqrt((pos_dict['x'] - object_pos['x']) ** 2 + (pos_dict['z'] - object_pos['z']) ** 2)
                    if distance < mini_distance:
                        teleport_pos = pos_dict
                        mini_distance = distance

                if teleport_pos is None:
                    reward_accumulate = 0
                # elif teleport_pos == agent_pos:
                #     action_name = "Wait"
                else:
                    yaw_appropriate = round(
                        (np.arctan2(-teleport_pos['x'] + object_pos['x'], -teleport_pos['z'] + object_pos['z']) / np.pi * 180) % 360,
                        1)
                    yaw_modified = 90.0 * round(yaw_appropriate / 90.0)

                    pitch = round(
                        (np.arctan2(object_pos['y'] - teleport_pos['y'], np.sqrt((object_pos['x'] - teleport_pos['x'])**2 + (object_pos['z'] - teleport_pos['z'])**2)) / np.pi * 180) % 360,
                        1
                    )
                    pitch_modified = 90.0 * round(pitch / 90.0)

                    standing = True
                    # crouch
                    if (round(object_pos['y'], 1) < 0.7):
                        standing = False
                        horizon = 30
                    # look down by 30 degrees
                    elif (round(object_pos['y'], 1) < 1.2):
                        horizon = 30
                    # look up by 30 degrees
                    elif (round(object_pos['y'], 1) > 1.8):
                        horizon = -30
                    else:
                        horizon = 0

                    if teleport_pos is not None:
                        teleport_pos = (teleport_pos['x'], teleport_pos['y'], teleport_pos['z'], yaw_modified, standing, horizon, pitch_modified)

            if action_name == "Wait" or teleport_pos is None:
                pass
                reward_accumulate = 0
            else:
                self.execuate_action(
                    {"action": action_list.index(action_name), "tar_index": object_index, "teleport_pose": teleport_pos},
                    agent_id=agent_id)
                reward_accumulate = -0.01

            self.update_dict()

            action_input = {"action": action_list.index(goal[0]), "tar_index": object_index}
            event = self.execuate_action(action_input, agent_id=agent_id)
            self.update_dict()
            if event is False:
                reward_accumulate -= 0.1
            # reward from goal
            _, reward = self._get_reward_and_done(agent_id, action=action_input)
            reward_accumulate += reward

            # print("for debug env", goal, agent_id, action_name, self._agent_type[1])
            if agent_id == 1:
                # print(event)
                # print(self.metadata(agent_id)['errorMessage'])
                # print("Subtask list", self.subtask_list)
                # print(self.goal_list)
                if self.metadata(agent_id)['errorMessage'] == "No valid positions to place object found":
                    # print("Error Message Correct")
                    pick_up_obj_id = self.agents[agent_id].pick_up_obj_id
                    # print(self.subtask_list[0][0] in ["In", "On"], self.subtask_list[0][1], pick_up_obj_id[:pick_up_obj_id.index('|')])
                    if self.subtask_list[0][0] in ["In", "On"] and self.subtask_list[0][1] == pick_up_obj_id[:pick_up_obj_id.index('|')]:
                        self.execuate_action({"action": action_list.index("Drop"), "tar_index": 0}, agent_id=agent_id)
                        self.update_dict()
                        self.subtask_list.pop(0)
                        self.goal_list.pop(0)
                        self.goal_num -= 1
                        print("Drop!!!")

            # check_goal
            if len(self.goal_list) > 1:
                while self.check_goal(self.goal_list[0], agent_id=agent_id, action=action_input):
                    self.goal_list.pop(0)
                    if len(self.goal_list) == 0:
                        break
            return  reward_accumulate

    def goal_dict_to_list(self, goal_dict):
        goal_name = goal_list[int(goal_dict['goal'])]
        obj_name = self.object_index2name[int(goal_dict['tar_index'])]
        return [goal_name, obj_name]
    
    def necessary_capability(self, subtask_list, agent_id=0):
        type = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for subtask in subtask_list:
            if subtask[0] in ["In", "On", "Get"]:
                obj_name = subtask[1]
                obj_index = self.find_index(obj_name)
                if obj_index is None:
                    # nearly impossible
                    type[0] = 1
                    type[1] = 1
                else:
                    type[0] = max(type[0], self.metadata(agent_id)['objects'][obj_index]['position']['y'] - 0.5)
                    type[1] = max(type[1], self.metadata(agent_id)['objects'][obj_index]['mass'])
            elif subtask[0] == "Open": 
                type[2] = 0.5
            elif subtask[0] == "Close":
                type[3] = 0.5
            elif subtask[0] == "ToggleOn":
                type[4] = 0.5
            elif subtask[0] == "ToggleOff":
                type[5] = 0.5
        return type

    def need_help_goal(self, goal_list, agent_id=0):
        required_type = self.necessary_capability(goal_list, agent_id)
        for i in range(6):
            # the main agent can not finish the task
            # print("need help?", "type", i,  required_type[i], self._agent_type[1][i])
            if required_type[i] > self._agent_type[1][i]:
                return True