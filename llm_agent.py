import argparse
import os
import time

import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

from ray.rllib.algorithms.algorithm import Algorithm
import openai
openai.api_key = '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'
import re
    
import cv2
import json
from env.symbolic_env import SymbolicEnv
from constants import AgentType
from model.helper_v3 import HelperModel
from env.type import sample_type
from env.task import sample_task

ray.init()

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=50, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=0.1, help="Reward at which we stop training."
)
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


controller_kwargs = {

}

config = {
    'agents_num': 2,
    # 'agents_type': {0: AgentType.AGENT_WITH_FULL_CAPABILITIES, 1: AgentType.AGENT_WITH_TOGGLE_ISSUES},
    'agents_type': {0: [1, 1, 1, 1, 1, 1], 1: sample_type()},
    'main_agent_id': 1,
    'mode': 'test',
    'controller_kwargs': {
        "agentCount": 2,
        "scene": 'FloorPlan2',
        "local_executable_path": "/home/zhihao/Downloads/thor-Linux64-local/thor-Linux64-local",
        # "local_executable_path": "/home/zhihao/A2SP/thor-Linux64-local/thor-Linux64-local",
        "renderDepthImage": True,
        "renderInstanceSegmentation": True,
        "visibilityDistance": 30, 
        "quality": "Very Low", 
        # "platform": "CloudRendering", 
    },
    'task': sample_task(),
}

with open('object_index2name.json', 'r') as fp:
    obj_idx2name = json.load(fp)
with open('object_name2index.json', 'r') as fp:
    obj_name2idx = json.load(fp)
from env.goal import goal_list
all_actions = 'all the actions are as follow: '
for idx, goal in enumerate(goal_list):
    all_actions += f'{idx}, {goal} '
action_dict = {action: idx for idx, action in enumerate(goal_list)}




def get_prompt_objs(obs_objs):
    prompt_objs = 'The objects and their properties are as follow: '
    for idx, obj in enumerate(obs_objs):
        if obj[-1] == 0:
            continue
        obj_name = obj_idx2name[str(int(obj[0]))]
        if obj[1] == 0:
            recep_name = None
        else:
            recep_name = obj_idx2name[str(int(obj[1]))]
        is_pickedup = True if obj[2] == 1 else False
        is_open = True if obj[3] == 1 else False
        is_cooked = True if obj[4] == 1 else False
        is_toggled = True if obj[5] == 1 else False
        visible_1 = True if obj[6] == 1 else False
        visible_2 = True if obj[7] == 1 else False
        height = obj[8]
        weight = obj[9]
        
        prompt_single_obj = f'{idx}, {obj_name}, '
        if recep_name == None: prompt_single_obj += 'has no parent receptacle, '
        else: prompt_single_obj += f'is in/on the {recep_name}, '
        if is_pickedup: prompt_single_obj += 'it can be picked up, '
        else: prompt_single_obj += 'it cannot be picked up, '
        if is_open: prompt_single_obj += 'it can be opened, '
        else: prompt_single_obj += 'it cannot be opened, '
        if is_cooked: prompt_single_obj += 'it can be cooked, '
        else: prompt_single_obj += 'it cannot be cooked, '
        if is_toggled: prompt_single_obj += 'it can be toggled on or off, '
        else: prompt_single_obj += 'it cnnnot be toggled on or off, '
        
        prompt_single_obj += 'its heights is {:.2f} and its weight is {:.2f}. '.format(height, weight)
        prompt_objs += prompt_single_obj
        
    return prompt_objs


def get_prompt_agents(obs_agts):
    # [height, weight, open, close, toggle_on, toggle_off]
    llm_agt, m_agt = obs_agts[0], obs_agts[1]
    
    prompt_agts = 'Except you, there is another agent in the environment, that is called your partner. You have the full capability, but your partner might have some defects. So you need to help him and cooperate a task. ' 
    if m_agt[0] < 1: agt_type = 'Your partner is short to reach high objects. '
    elif m_agt[1] < 1: agt_type = 'Your partner lacks strength and this may prevent him from picking objects up. '
    elif m_agt[2] < 1: agt_type = 'Your partner may fail when he tries to open an object. '
    elif m_agt[3] < 1: agt_type = 'Your partner may fail when he tries to close an object. '
    elif m_agt[4] < 1: agt_type = 'Your partner may fail when he tries to toggle an object on. '
    elif m_agt[5] < 1: agt_type = 'Your partner may fail when he tries to toglle an object off. '
    else: agt_type = 'Your partner has full capability, too.'

    def get_action_info(obs_agt):
        if obs_agt[6] == 0 and obs_agt[7] == 0 and obs_agt[8] == 0:
            return None
        else:
            action = all_actions[int(obs_agt[6])]
            target_object = obj_idx2name[str(int(obs_agt[7]))]
            success = obj_idx2name[str(int(obs_agt[8]))]
            return action, target_object, success
    
    m_agt_action_info = get_action_info(m_agt)
    if m_agt_action_info == None:
        m_agt_action = 'Your partner has not taken any action. '
    else:
        action, target_object, success = m_agt_action_info
        m_agt_action = f'Your partner has taken {action} at last step with object {target_object}. '
        if success:
            m_agt_action += 'His last action is taken successfully. '
        else:
            m_agt_action += 'But his last action is failed. '    
    llm_agt_action_info = get_action_info(llm_agt)
    if llm_agt_action_info == None:
        llm_agt_action = 'You have not taken any action. '
    else:
        action, target_object, success = llm_agt_action_info
        llm_agt_action = f'You have taken {action} at last step with object {target_object}. '
        if success:
            llm_agt_action += 'Your last action is taken successfully. '
        else:
            llm_agt_action += 'But your last action is failed. '    

    prompt_agts += agt_type + m_agt_action + llm_agt_action
    return prompt_agts


def get_prompt_requirement():
    prompt_requirement = 'Now, you and your partner are cooperating to complete a task. ' 
    prompt_requirement += 'You do not know how to complete this task, but your partner do. '
    prompt_requirement += 'You need to observe the action of your partner and the current environment to infer the goals of your partner. '
    prompt_requirement += 'Based on your observation and inference, you can decide your next action and target object. '
    prompt_requirement += '\n'
    prompt_requirement += 'The properties of the observed objects need to be considered. You can only select a action from the aforementioned actions. '
    prompt_requirement += '\n'
    prompt_requirement += 'You must output your decision following the format of "action-object", e.g., "PutIn-Fridge" or "PickUp-Apple". '
    prompt_requirement += 'If you have not decided an action, you can output "Wait-None". '
    prompt_requirement += 'You can only output single "action-object" pair at once. The output format must followed strictly, and do not output other infomation. '
    return prompt_requirement


def get_action(observation):
    prompt_objs = get_prompt_objs(observation[0: 30])
    prompt_agts = get_prompt_agents(observation[-2: ])
    prompt_rqm = get_prompt_requirement()
    prompt = prompt_objs + '\n' + all_actions + '\n' + prompt_agts + '\n' + prompt_rqm
    
    response = openai.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response.choices[0].text)
    pattern = r'\b(\w+)-(\w+)\b'
    matches = re.findall(pattern, response.choices[0].text)

    if matches:
        extracted_string = matches[0]
        # print(extracted_string)
        return extracted_string
    else:
        # print("No match found.")
        return None


def test_pipeline(args):
    ModelCatalog.register_custom_model(
        "helper", HelperModel
    )
    
    test_env = SymbolicEnv(config=config)

    
    num_episodes = 10
    for episode in range(num_episodes):
        observation, _ = test_env.reset()
        print("==========A NEW ENV=========")
        print("Agent type", test_env._agent_type[1])
        print("required type", test_env.necessary_capability(test_env.subtask_list))
        print('subatsks: ', test_env.subtask_list)
        done = False
        episode_reward = 0

        while not done:
            action_info = get_action(observation)
            try:
                action = dict(goal=action_dict[action_info[0]], tar_index=obj_name2idx[action_info[1]])
            except:
                action = dict(goal=0, tar_index=0)
            print(action)
            observation, reward, done, _, _ = test_env.step(action)
            episode_reward += reward
            # time.sleep(1)
        print("Evaluation")
        print("SR :", int(not (test_env.step_count == 200)))
        if test_env.goal_num == 0:
            print("GSR:", -1)
        else:
            print("GSR:", test_env.finish_goal_num / test_env.goal_num)
        if test_env.finish_goal_num == 0:
            print("CR :", -1)
        else:
            print("CR :", test_env.helper_finish_goal_num / test_env.finish_goal_num)
        if test_env.helper_action_num == 0:
            print("HE :", -1)
        else:
            print("HE :", test_env.helper_finish_required_action_num / test_env.helper_action_num)
        if test_env.helper_finish_goal_num == 0:
            print("HN :", -1)
        else:
            print("HN :", test_env.helper_finish_necc_goal_num / test_env.helper_finish_goal_num)
        print(f"Episode {episode+1}: Total Reward = {episode_reward}")

    # dict = {
    #     'position_list': position_list,
    #     'target_pos': target_pos
    # }

    # # 指定要保存的JSON文件路径
    # json_file_path = "./file.json"

    # # 使用json.dump()将list保存为JSON文件
    # with open(json_file_path, "w") as json_file:
    #     json.dump(dict, json_file)



def test_llm():
    import openai
    openai.api_key = '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'

    response = openai.completions.create(
        model="text-davinci-003",
        prompt='I want to write a abstract about multi-agent system, please write a script for me',
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    print(response.choices[0].text)



if __name__ == "__main__":
    args = parser.parse_args()

    test_pipeline(args)

    # test_llm()

            