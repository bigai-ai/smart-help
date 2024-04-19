# 需要确定测试集组成
# 推荐组成：所有的task，所有的env，所有的type联合，测三遍

# task_list = ["MakeBreakfast", "MakeCoffee", "ArrangeRoom"]
# task_list = ["MakeCoffee"]
task_list = ["MakeBreakfast", "MakeCoffee", "ArrangeRoom"]

import argparse
import os
import time
import copy
import json
import numpy as np

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

import cv2
import json
from env.symbolic_env import SymbolicEnv
from constants import AgentType
from model.helper_v3 import HelperModel
from env.type import sample_type
from env.task import sample_task
from model.end2end import End2End
from model.helper_without_super import HelperModelWithoutSupervise
from action import action_list
import openai
openai.api_key = 'openai-api-key'
import re
import pandas as pd
# import tiktoken





tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

controller_kwargs = {

}

config = {
    'agents_num': 2,
    # 'agents_type': {0: AgentType.AGENT_WITH_FULL_CAPABILITIES, 1: AgentType.AGENT_WITH_TOGGLE_ISSUES},
    'agents_type': {0: [2, 2, 1, 1, 1, 1], 1: sample_type()},
    'main_agent_id': 1,
    'mode': 'train',
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
        pos = obj[10:13]
        dis = obj[13]
        
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
        
        prompt_single_obj += 'its heights is {:.2f} and its weight is {:.2f}, '.format(height, weight)
        prompt_single_obj += 'its position is x:{:.2f}, y:{:.2f}, z:{:.2f}, '.format(pos[0], pos[1], pos[2])
        prompt_single_obj += 'its distance from main agent is {:.2f}. '.format(dis)
        prompt_objs += prompt_single_obj
        
    return prompt_objs


def get_prompt_agents(obs_agts, main_agent_action, llm_agent_action):
    # [height, weight, open, close, toggle_on, toggle_off]
    llm_agt, m_agt = obs_agts[0], obs_agts[1]
    
    prompt_agts = '' 

    def get_action_info(obs_agt):
        if obs_agt[6] == 0 and obs_agt[7] == 0 and obs_agt[8] == 0:
            return None
        else:
            print(int(obs_agt[6]))
            action = action_list[int(obs_agt[6])]
            success = True if int(obs_agt[8]) == 2 else False
            return action, success

    m_agt_action_info = get_action_info(m_agt)
    if m_agt_action_info == None:
        m_agt_action = 'Your partner has not taken any action. '
    else:
        action, success = m_agt_action_info
        m_agt_action = f'Your partner has taken {action} at last step. '
        if success:
            m_agt_action += 'His last action is taken successfully. '
        else:
            m_agt_action += 'But his last action is failed. '  
        main_agent_action.append([action, success])  
        m_agt_action += f"His previous actions are: {main_agent_action}. For a single action, it is [action, success]. " 
    m_agt_action += f"His position is x:{m_agt[0]}, y:{m_agt[1]}, z:{m_agt[2]}. "
    m_agt_action += f"His rotation is x:{m_agt[3]}, y:{m_agt[4]}, z:{m_agt[5]}. "
    m_agt_action += f"Now he is holding {obj_idx2name[str(int(m_agt[9]))]}. "
    llm_agt_action_info = get_action_info(llm_agt)
    if llm_agt_action_info == None:
        llm_agt_action = 'You have not taken any action. '
    else:
        action, success = llm_agt_action_info
        llm_agt_action = f'You have taken {action} at last step. '
        if success:
            llm_agt_action += 'Your last action is taken successfully. '
        else:
            llm_agt_action += 'But your last action is failed. '    
        llm_agent_action.append([action, success])
    llm_agt_action += f"His position is x:{llm_agt[0]}, y:{llm_agt[1]}, z:{llm_agt[2]}. "
    llm_agt_action += f"His rotation is x:{llm_agt[3]}, y:{llm_agt[4]}, z:{llm_agt[5]}. "
    llm_agt_action += f"Now he is holding {obj_idx2name[str(int(llm_agt[9]))]}. "

    prompt_agts += m_agt_action + llm_agt_action
    return prompt_agts


def get_prompt_requirement():
    prompt_requirement = """To help your partner, you need to infer which task and goal that he is doing based on your observation of objects and agents,
    and infer his capability at the same time. 
    Based on your inferred goal and capability, you can choose whether and how to help your partner. 
    Only when you find your partner can not finish the task on his own, the helping is appropriate. """


    # prompt_requirement = 'Now, you and your partner are cooperating to complete a task. ' 
    # prompt_requirement += 'You do not know how to complete this task, but your partner do. '
    # prompt_requirement += 'You need to observe the action of your partner and the current environment to infer the goals of your partner. '
    # prompt_requirement += 'Based on your observation and inference, you can decide your next action and target object. '
    # prompt_requirement += '\n'
    # prompt_requirement += 'The properties of the observed objects need to be considered. You can only select a action from the aforementioned actions. '
    # prompt_requirement += '\n'
    prompt_requirement += 'You must output your decision following the format of "action-object", e.g., "Put-Fridge" or "ToggleOn-Faucet". '
    prompt_requirement += 'If you have not decided an action, you can output "Wait-None". '
    prompt_requirement += 'You can only output single "action-object" pair at once. '
    prompt_requirement += "You must choose a concrete action and object for outputing. "
    prompt_requirement += "For example, ('action', 'object') is invalid, please output the action like ('PickUp', 'Cup') instead. "
    # prompt_requirement += 'You can only output single "action-object" pair at once. The output format must followed strictly, and do not output other infomation. '
    prompt_requirement += "Let's think step by step and output your action. "
    
    # prompt_requirement += 'You can only output single "action-object" pair at once. The output format must followed strictly, and do not output other infomation. '
    return prompt_requirement

def get_prompt_example():
    prompt_example = 'Here is an example: '
    prompt_example += 'You observe There are Faucet, Apple, Fridge, etc. in your environment. '
    prompt_example += 'You observe your partner is picking up something, and succeeds. '
    prompt_example += 'Then you must infer what he has picked up, and infer his capability. '
    prompt_example += 'Example ends. '
    return prompt_example

def get_action(observation, last_observation, main_agent_action, llm_agent_action):
    begin_text = """There are two agents in the environment: you and your partner. Your goal is to help your partner. 
    To do this, you need to observe the current environment to infer the task and goal of your partner. 
    There are three possible tasks: MakeBreakfast, MakeCoffee, and ArrangeRoom. 
    There are seven possible goals: Wait, PickUp something, Put something on somewhere, ToggleOn something, ToggleOff something, Open something, Close something
    To complete the task, your partner will separate the task into several goals and complete them successively. 
    However, your partner's capability may be lacking. 
    He will fail in some actions. 
    There are six kinds of capabilities: mass, height, open, close, toggle_on, toggle_off, and your partner will fail in none or several kinds of capabilities. 
    Mass and height determine the maximum weight and height that the helper can pick up, and the last four capabilities will affect the success of the corresponding action. 
    Then you will receive the objects' informaction which are in your sight, and the state of your partner. 
    """
    prompt_objs = get_prompt_objs(observation[0: 30])
    prompt_agts = get_prompt_agents(observation[-2: ], main_agent_action, llm_agent_action)
    prompt_rqm = get_prompt_requirement()
    prompt_eg = get_prompt_example()

    if last_observation is None:
        prompt = begin_text + '\n' + prompt_objs + '\n' + all_actions + '\n' + prompt_agts + '\n' + prompt_eg + '\n' + prompt_rqm 
    else:
        # if np.array_equal(last_observation[0], observation[0]):
        #     prompt = prompt_agts
        # else:
        #     prompt = prompt_agts + prompt_objs
        # prompt = prompt_agts + '\n' + prompt_objs + '\n' + prompt_rqm
        prompt = begin_text + '\n' + prompt_objs + '\n' + all_actions + '\n' + prompt_agts + '\n' + prompt_eg + '\n' + prompt_rqm 
    
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
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


    # print(prompt)
    # print(response)
    # print(matches)

    if matches:
        while True:
            extracted_string = matches[-1]
            if extracted_string[0] == 'action' or extracted_string[1] == 'object': 
                matches.remove(extracted_string)
                if len(matches) == 0:
                    print("Action, ", ['Wait', 'None'])
                    return ['Wait', 'None']
                continue
            if 'put' in extracted_string[0].lower():
                print("Action, ", extracted_string)
                return ('Put', extracted_string[1])
            # print(extracted_string)
            print("Action, ", extracted_string)
            return extracted_string
    else:
        # print("No match found.")
        print("Action, ", ['Wait', 'None'])
        return ['Wait', 'None']



if __name__ == "__main__":
    total_dict = {}
    task_dict = {}

    os.makedirs('new_test_llm_result', exist_ok=True)

    position_list = []

    # 3. 定义测试环境
    test_env = SymbolicEnv(config=config)

    # exit(0)
    # 4. 运行测试
    num_episodes = 30
    SR_list = []
    GSR_list = []
    CR_list = []
    HE_list = []
    HN_list = []
    reward_list = []
    eposide_len_list = []
    SPL_list = []
    helping_num = 0
    need_help_num = 0
    for task in task_list:
        task_dict[task] = {}
        for type_index in range(7):
            if task == "MakeBreakfast" and type_index in [6]:
                continue
            elif task == "MakeCoffee" and type_index in [2, 3, 6]:
                continue
            elif task == "ArrangeRoom" and type_index in [4, 5, 6]:
                continue
            for idx in range(3):
                action_history = []
                task_dict[task][type_index] = {}
                tmp_SR_list = []
                tmp_GSR_list = []
                tmp_CR_list = []
                tmp_HE_list = []
                tmp_HN_list = []
                
                tmp_helping_num = 0
                tmp_need_help_num = 0
                # type_index == 6: full capability
                for env_index in range(20, 30):
                    if os.path.exists(f'new_llm/{task}_{type_index}_{env_index}_{idx}.json'):
                        # print(f'test_llm_result/{task}_{type_index}_{env_index}.json')
                        continue
                    print("Testing start!!!")
                    main_agent_action = []
                    llm_agent_action = []
                    observation, _ = test_env.reset(env_index=env_index + 1, task=task, type=type_index)
                    if test_env.need_help:
                        tmp_need_help_num += 1  
                    # print("Agent type", test_env._agent_type[1])
                    # print("required type", test_env.necessary_capability(test_env.subtask_list))
                    # lstm_state = algo.get_policy().get_initial_state()
                    done = False
                    episode_reward = 0
                    last_observation = None
                    while not done:
                        action_info = get_action(observation, last_observation, main_agent_action, llm_agent_action)
                        last_observation = observation
                        try:
                            action = dict(goal=action_dict[action_info[0]], tar_index=obj_name2idx[action_info[1]])
                        except:
                            action = dict(goal=0, tar_index=0)
                        observation, reward, done, _, _ = test_env.step(action)
                        episode_reward += reward
                        # time.sleep(1)
                    print("Evaluation")
                    SR = int(not (test_env.step_count == 30))
                    print("SR :", SR)
                    if test_env.goal_num == 0:
                        GSR = 0
                    else:
                        GSR = test_env.finish_goal_num / test_env.goal_num
                    print("GSR:", GSR)
                    if test_env.finish_goal_num == 0:
                        CR = 0
                    else:
                        CR = test_env.helper_finish_goal_num / test_env.finish_goal_num
                    print("CR :", CR)
                    if test_env.helper_action_num == 0:
                        HE = -1
                    else:
                        HE = test_env.helper_finish_required_action_num / test_env.helper_action_num
                    print("HE :", HE)
                    if test_env.helper_finish_goal_num == 0:
                        HN = -1
                        print("HN :", -1)
                    else:
                        HN = test_env.helper_finish_necc_goal_num / test_env.helper_finish_goal_num
                    print("HN :", HN)
                    if HN == -1:
                        pass
                    else:
                        tmp_helping_num += 1
                        tmp_HE_list.append(HE)
                        tmp_HN_list.append(HN)
                    tmp_SR_list.append(SR)
                    tmp_GSR_list.append(GSR)
                    tmp_CR_list.append(CR)
                    reward_list.append(episode_reward)
                    eposide_len_list.append(test_env.step_count)
                    SPL_list.append(SR * (test_env.goal_num / max(test_env.goal_num, test_env.step_count)))
        
                    print(f"Total Reward = {episode_reward}")
                    with open(f'dep_llm/{task}_{type_index}_{env_index}_{idx}.json', 'w') as fp:
                        json.dump(
                            dict(
                                main_action=main_agent_action, 
                                llm_action=llm_agent_action, 
                                SR=SR, GSR=GSR, CR=CR, HE=HE, HN=HN,
                                goal_num = test_env.goal_num,
                                SPL = SR * (test_env.goal_num / max(test_env.goal_num, test_env.step_count)),
                                need_help = test_env.need_help,
                                total_reward = episode_reward,
                            ), fp, indent=4
                        )
            task_dict[task][type_index]['SR'] = copy.deepcopy(tmp_SR_list)
            task_dict[task][type_index]['GSR'] = copy.deepcopy(tmp_GSR_list)
            task_dict[task][type_index]['CR'] = copy.deepcopy(tmp_CR_list)
            task_dict[task][type_index]['HE'] = copy.deepcopy(tmp_HE_list)
            task_dict[task][type_index]['HN'] = copy.deepcopy(tmp_HN_list)
            task_dict[task][type_index]['helping_num'] = copy.deepcopy(tmp_helping_num)
            task_dict[task][type_index]['need_help_num'] = copy.deepcopy(tmp_need_help_num)
            # print("=======END=======")
            # print(f"task: {task}, type: {type_index}")
            # print("average_SR : ", sum(tmp_SR_list) / len(tmp_SR_list))
            # print("average_GSR: ", sum(tmp_GSR_list) / len(tmp_GSR_list))
            # print("average_CR : ", sum(tmp_CR_list) / len(tmp_CR_list))
            # print("average_HE : ", sum(tmp_HE_list) / max(len(tmp_HE_list), 1))
            # print("average_HN : ", sum(tmp_HN_list) / max(len(tmp_HN_list), 1))
            # print("helping_num : ", tmp_helping_num)
            # print("need_help_num:", tmp_need_help_num)
            SR_list = SR_list + tmp_SR_list
            GSR_list = GSR_list + tmp_GSR_list
            CR_list = CR_list + tmp_CR_list
            HE_list = HE_list + tmp_HE_list
            HN_list = HN_list + tmp_HN_list
            helping_num += tmp_helping_num
            need_help_num += tmp_need_help_num
    print("=======END=======")
    print("average_SR : ", sum(SR_list) / max(1, len(SR_list)))
    print("average_GSR: ", sum(GSR_list) / max(1, len(GSR_list)))
    print("average_CR : ", sum(CR_list) / max(1, len(CR_list)))
    print("average_HE : ", sum(HE_list) / max(len(HE_list), 1))
    print("average_HN : ", sum(HN_list) / max(len(HN_list), 1))
    print("helping_num : ", helping_num)
    print("need_help_num:", need_help_num)
    total_dict['SR'] = SR_list
    total_dict['GSR'] = GSR_list
    total_dict['CR'] = CR_list
    total_dict['HE'] = HE_list
    total_dict['HN'] = HN_list
    total_dict['helping_num'] = helping_num
    total_dict['need_help_num'] = need_help_num

    with open("total.json", "w") as json_file:
        json.dump(total_dict, json_file)

    with open("task.json", "w") as json_file:
        json.dump(task_dict, json_file)
