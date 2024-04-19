# 需要确定测试集组成
# 推荐组成：所有的task，所有的env，所有的type联合，测三遍

task_list = ["MakeBreakfast", "MakeCoffee", "ArrangeRoom"]
import os
import time
import copy
import json
import csv

#from ray.rllib.utils.framework import try_import_tf, try_import_torch

import cv2
import json
import numpy as np
from env.symbolic_env import SymbolicEnv
from env.type import sample_type
from env.task import sample_task
from env.subtask import subtask_list
from env.goal import goal_list
from model.classifier import Classifier_OppenentModeling_v2
from mcts_helper import run_mcts

import concurrent.futures

import pandas as pd

# tf1, tf, tfv = try_import_tf()
# torch, nn = try_import_torch()

import torch
import torch.nn as nn

TEST_SUBTASK_PREDICTION = True

controller_kwargs = {

}

config = {
    'agents_num': 2,
    'agents_type': {0: [2, 2, 1, 1, 1, 1], 1: sample_type()},
    'main_agent_id': 1,
    'mode': 'train',
    'controller_kwargs': {
        "agentCount": 2,
        "scene": 'FloorPlan2',
        "local_executable_path": "/home/lfan/ai2thor/ai2thor/unity/builds/thor-Linux64-local/thor-Linux64-local",
        "renderDepthImage": True,
        "renderInstanceSegmentation": True,
        "visibilityDistance": 30, 
        "quality": "Very Low", 
    },
    'task': sample_task(),
}



if __name__ == "__main__":
    total_dict = {}
    task_dict = {}
    position_list = []
    # 3. 定义测试环境
    test_env = SymbolicEnv(config=config)
    classifier = Classifier_OppenentModeling_v2()
    if torch.cuda.is_available():
        classifier = classifier.cuda()
        
    tmp_env = SymbolicEnv(config=config)

    OBJ_NUM = len(test_env.object_index2name)

    # exit(0)
    # 4. 运行测试
    num_episodes = 30
    SR_list = []
    GSR_list = []
    CR_list = []
    HE_list = []
    HN_list = []
    episode_len_list = []
    SPL_list = []
    helping_num = 0
    need_help_num = 0
    list_results = [['env', 'type', 'task', 'ground_truth', 'predicted_subtask']]
    for task in task_list:
        task_dict[task] = {}
        for type_index in range(0, 1):
            if task == "MakeBreakfast" and type_index in [6]:
                continue
            elif task == "MakeCoffee" and type_index in [2, 3, 6]:
                continue
            elif task == "ArrangeRoom" and type_index in [4, 5, 6]:
                continue
            task_dict[task][type_index] = {}
            tmp_SR_list = []
            tmp_GSR_list = []
            tmp_CR_list = []
            tmp_HE_list = []
            tmp_HN_list = []
            tmp_episode_len_list = []
            tmp_SPL_list = []
            tmp_helping_num = 0
            tmp_need_help_num = 0
            # type_index == 6: full capability
            
            for env_index in range(0, 10):
                data = np.zeros((5, 32, 11))
                observation, _ = test_env.reset(env_index=env_index + 21, task=task, type=type_index)
                
                if test_env.need_help:
                    tmp_need_help_num += 1  
                done = False
                episode_reward = 0
                action_seq = []
                
                while not done:
                    data[:4] = data[1:]
                    data[4] = observation
                    subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict = classifier(data)
                    subtask_predict = subtask_list[torch.argmax(subtask_predict).item()]
                    tar_index_1_predict = test_env.object_index2name[torch.argmax(tar_index_1_predict).item()]
                    tar_index_2_predict = test_env.object_index2name[torch.argmax(tar_index_2_predict).item()]

                    # MCTS here
                    # Set mode
                    select_goal = 'predicted_goal'   # true_goal, random_goal, predicted_goal
                    if TEST_SUBTASK_PREDICTION:
                        select_goal = 'predicted_goal'
                    if select_goal == 'true_goal':
                        predicted_subtask = test_env.subtask_list[0]
                    elif select_goal == 'predicted_goal':
                        predicted_subtask = [subtask_predict, tar_index_1_predict]
                        if tar_index_2_predict is not None:
                            predicted_subtask.append(tar_index_2_predict)
                    else:
                        predicted_subtask = None
                    
                    list_results.append([env_index, type_index, test_env.task, test_env.subtask_list[0], predicted_subtask])

                    # Test ground truth
                    #predicted_subtask = test_env.subtask_list[0]
                    
                    print(f'Testing env: {env_index}, type: {type_index}, task: {test_env.task}, predicted_subtask: {predicted_subtask}')
                    #print(f'Testing env: {env_index}, task: {test_env.task}: {test_env.subtask_list}, predicted_subtask: {predicted_subtask}')
                    #print(predicted_subtask)
                    
                    tmp_env.reset(env_index=env_index + 21, task=task, type=type_index)

                    # Params here
                    if TEST_SUBTASK_PREDICTION:
                        action = run_mcts(tmp_env, action_seq, predicted_subtask, type_predict, num_sim=1, sample_prob=0.0)
                    else:
                        action = run_mcts(tmp_env, action_seq, predicted_subtask, type_predict, num_sim=1, sample_prob=0.0)

                    action_seq.append(action)
                    print(f"Taking action {action['goal']}-{goal_list[action['goal']]} \
                    with obj {action['tar_index']}-{test_env.object_index2name[action['tar_index']]}")

                    observation, reward, done, _, _ = test_env.step(action)
                    episode_reward += reward
                    if test_env.check_task():
                        print('Task complete!')
                        done = True
                    if test_env.step_count >= 30:
                        done = True
                # time.sleep(1)
                #print(test_env.step_count)
                print("Evaluation")
                episode_len = test_env.step_count
                SR = int(not (test_env.step_count >= 30))
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
                #print(f"Total Reward = {episode_reward}")
                SPL = SR * (test_env.goal_num / max(test_env.goal_num, test_env.step_count))
                tmp_episode_len_list.append(episode_len)
                tmp_SPL_list.append(SPL)
            task_dict[task][type_index]['SR'] = copy.deepcopy(tmp_SR_list)
            task_dict[task][type_index]['GSR'] = copy.deepcopy(tmp_GSR_list)
            task_dict[task][type_index]['CR'] = copy.deepcopy(tmp_CR_list)
            task_dict[task][type_index]['HE'] = copy.deepcopy(tmp_HE_list)
            task_dict[task][type_index]['HN'] = copy.deepcopy(tmp_HN_list)
            task_dict[task][type_index]['helping_num'] = copy.deepcopy(tmp_helping_num)
            task_dict[task][type_index]['need_help_num'] = copy.deepcopy(tmp_need_help_num)
            task_dict[task][type_index]['episode_len'] = copy.deepcopy(tmp_episode_len_list)
            task_dict[task][type_index]['SPL'] = copy.deepcopy(tmp_SPL_list)
            print("=======END=======")
            print(f"task: {task}, type: {type_index}")
            print("average_SR : ", sum(tmp_SR_list) / len(tmp_SR_list))
            print("average_GSR: ", sum(tmp_GSR_list) / len(tmp_GSR_list))
            print("average_CR : ", sum(tmp_CR_list) / len(tmp_CR_list))
            print("average_HE : ", sum(tmp_HE_list) / max(len(tmp_HE_list), 1))
            print("average_HN : ", sum(tmp_HN_list) / max(len(tmp_HN_list), 1))
            print("average_episode_len: ", sum(tmp_episode_len_list) / len(tmp_episode_len_list))
            print("average_SPL: ", sum(tmp_SPL_list) / len(tmp_SPL_list))
            print("helping_num : ", tmp_helping_num)
            print("need_help_num:", tmp_need_help_num)
            SR_list = SR_list + tmp_SR_list
            GSR_list = GSR_list + tmp_GSR_list
            CR_list = CR_list + tmp_CR_list
            HE_list = HE_list + tmp_HE_list
            HN_list = HN_list + tmp_HN_list
            episode_len_list = episode_len_list + tmp_episode_len_list
            SPL_list = SPL_list + tmp_SPL_list
            helping_num += tmp_helping_num
            need_help_num += tmp_need_help_num
    print("=======END=======")
    print("average_SR : ", sum(SR_list) / len(SR_list))
    print("average_GSR: ", sum(GSR_list) / len(GSR_list))
    print("average_CR : ", sum(CR_list) / len(CR_list))
    print("average_HE : ", sum(HE_list) / max(len(HE_list), 1))
    print("average_HN : ", sum(HN_list) / max(len(HN_list), 1))
    print("average_eposide_len: ", sum(episode_len_list) / len(episode_len_list))
    print("average_SPL: ", sum(SPL_list) / len(SPL_list))
    print("helping_num : ", helping_num)
    print("need_help_num:", need_help_num)
    total_dict['SR'] = SR_list
    total_dict['GSR'] = GSR_list
    total_dict['CR'] = CR_list
    total_dict['HE'] = HE_list
    total_dict['HN'] = HN_list
    total_dict['helping_num'] = helping_num
    total_dict['need_help_num'] = need_help_num
    total_dict['episode_len'] = episode_len_list
    total_dict['SPL'] = SPL_list

    if TEST_SUBTASK_PREDICTION:
        with open("subtask_prediction_results.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for row in list_results:
                writer.writerow(row)

    with open("total.json", "w") as json_file:
        json.dump(total_dict, json_file)

    with open("task.json", "w") as json_file:
        json.dump(task_dict, json_file)

