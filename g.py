from env.single_env_symbolic import Single_Env_Symbolic
from model.expert import Expert
from constants import AgentType
from action import action_list
import time

from gymnasium import spaces

print("Coffee".find("|"))

raise Exception

config = {
    'agents_num': 1,
    'agents_type': 'rllib',
    'main_agent_id': 0,
    'mode': 'train',
    'controller_kwargs': {
        "agentCount": 1,
        "scene": 'FloorPlan2',
        "local_executable_path": "/home/zhihao/A2SP/thor-Linux64-local/thor-Linux64-local",
        "renderDepthImage": True,
        "renderInstanceSegmentation": True,
        "visibilityDistance": 6, 
    },
    # 'task': ['move_to', 'Apple|-00.93|+01.15|+00.95']
    # todo id change?
    # 'task': ['get', 'Tomato|+00.17|+00.97|-00.28'],
    'task': ['PutOn', 'Tomato|+00.17|+00.97|-00.28', 'Sink|+00.00|+00.89|-01.44'],
    'agents_type': {0: AgentType.AGENT_WITH_CARELESS_MIND}
}

env = Single_Env_Symbolic(config=config)

expert = Expert(env.goal_list, env.controller, 0.25)

print("Bug?")

# print('--------------', len(env.get_symbolic_object_observations(0)))

print(env.controller.step(action="GetReachablePositions").metadata['actionReturn'])

# print(env.scene_state)

test_space = spaces.Dict({
    "obj_obs": spaces.Box(low=-30, high=70, shape=(11 + 2, 30)), 
    # goal_index, goal_obj1_index, goal_obj2_index, goal_obj1_pos, goal_obj2_pos
    "goal": spaces.Box(low=0, high=70, shape=(1, 6)),
})

print(test_space['obj_obs'].shape)

raise Exception

# expert.update(env.get_agent_state(0), env.scene_state, None)

while True:

    expert.update(env.get_agent_state(0), env.scene_state, None)

    print(expert.expert_action)

    # print('--------------', len(env.get_symbolic_object_observations(0)))

    print('--------------', env.get_symbolic_object_observations(0))

    print(expert.current_goal)

    action_index = action_list.index(expert.expert_action[0])

    if isinstance(expert.expert_action[1], int):
        action_input = {"action": action_index, "tar_index": expert.expert_action[1]}
    else:
        action_input = {"action": action_index, "tar_index": env.object_name2index[expert.expert_action[1]]}

    _, _, done, _, _ = env.step(action=action_input)

    if done:
        break

    time.sleep(0.5)





# print("///////////////////", env.get_main_observations())