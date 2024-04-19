# 需要确定测试集组成
# 推荐组成：所有的task，所有的env，所有的type联合，测三遍

task_list = ["MakeBreakfast", "MakeCoffee", "ArrangeRoom"]
import copy
import json

import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.algorithms.algorithm import Algorithm

import json
from env.symbolic_env import SymbolicEnv
from model.helper_v3 import HelperModel
from env.type import sample_type
from env.task import sample_task
from model.end2end import End2End
from model.helper_without_super import HelperModelWithoutSupervise
from model.helper_dep_flow import HelperModelDepFlow
from model.helper_v4 import HelperModelV4
from env.goal import GOAL_NUM
from model.helper_dep import HelperModelDep

from env.symbolic_env import SymbolicEnv
from model.end2end import End2End
import pandas as pd
import random

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

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
        "local_executable_path": "/home/zhihao/Downloads/thor-Linux64-local/thor-Linux64-local",
        "renderDepthImage": False,
        "renderInstanceSegmentation": False,
        "visibilityDistance": 30, 
        "quality": "Very Low", 
    },
    'task': sample_task(),
}

if __name__ == "__main__":
    total_dict = {}
    task_dict = {}

    ModelCatalog.register_custom_model(
        "helper", HelperModel
    )

    ModelCatalog.register_custom_model(
        "e2e", End2End
    )

    ModelCatalog.register_custom_model(
        "helperwithoutsuper", HelperModelWithoutSupervise
    )

    ModelCatalog.register_custom_model(
        "helperv4", HelperModelV4
    )

    ModelCatalog.register_custom_model(
        "helperdepflow", HelperModelDepFlow
    )

    ModelCatalog.register_custom_model(
        "helperdep", HelperModelDep
    )


    ray.init()

    position_list = []

    # algo = (
    #     PPOConfig()
    #     .rollouts(num_rollout_workers=4)
    #     .resources(num_gpus=1)
    #     .environment(MultiEnv, env_config=config)
    #     .build()
    # )

    # 3. 定义测试环境

    
    print("======================Start ======================")
    test_env = SymbolicEnv(config=config)
    print("======================Env ======================")

    # algo = Algorithm.from_checkpoint('/home/zhihao/Downloads/checkpoint_finial/ours/409')
    # algo = Algorithm.from_checkpoint('/home/zhihao/下载/checkpoint_finial/e2e/339')
    # algo = Algorithm.from_checkpoint('/home/zhihao/Downloads/ours_e2e_wo_pun/349')
    # algo = Algorithm.from_checkpoint('/home/zhihao/Downloads/ours_dep/929')
    # algo = Algorithm.from_checkpoint('/home/zhihao/Downloads/ours_wo_pun/839')
    # algo = Algorithm.from_checkpoint('/home/zhihao/Downloads/ours_wo_pun_finetune/229')
    algo = Algorithm.from_checkpoint('/home/zhihao/Downloads/ours_wo_pun_low_lr/229')
    algo = Algorithm.from_checkpoint('/home/zhihao/Downloads/ours_dep/839')
    print("======================THE CHECK POINT ======================")

    # exit(0)
    # 4. 运行测试
    # only the left 10 scenarios
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
            task_dict[task][type_index] = {}
            tmp_SR_list = []
            tmp_GSR_list = []
            tmp_CR_list = []
            tmp_HE_list = []
            tmp_HN_list = []
            tmp_reward_list = []
            tmp_eposide_len_list = []
            tmp_SPL_list = []
            tmp_helping_num = 0
            tmp_need_help_num = 0
            # type_index == 6: full capability
            for env_index in range(10):
                for _ in range(3):
                    observation, _ = test_env.reset(env_index=env_index + 21, task=task, type=type_index)
                    if test_env.need_help:
                        tmp_need_help_num += 1  
                    lstm_state = algo.get_policy().get_initial_state()
                    done = False
                    episode_reward = 0
                    while not done:
                        action, lstm_state, _ = algo.compute_single_action(observation, state=lstm_state)
                        # action_name = random.randint(0, GOAL_NUM - 1)
                        # tar_index = random.randint(0, len(test_env.object_name2index)-1)
                        # action = {
                        #     "goal": action_name, 
                        #     "tar_index": tar_index
                        # }
                        observation, reward, done, _, _ = test_env.step(action)
                        episode_reward += reward
                        # time.sleep(1)
                    print("Evaluation")
                    eposide_len = test_env.step_count
                    SR = int(not (test_env.step_count == 30))
                    print("SR :", SR)
                    if test_env.goal_num == 0:
                        GSR = 0
                    else:
                        GSR = test_env.finish_goal_num / test_env.goal_num
                    print("GSR:", GSR)
                    if test_env.helper_finish_goal_num == 0:
                        if SR == 1:
                            HN = -1
                        else:
                            # 为了help_num的计算，取一个特殊的值，实际上应为0
                            HN = -2
                    else:
                        HN = test_env.helper_finish_necc_goal_num / test_env.helper_finish_goal_num
                        # for finding scenario
                        if HN < 0.4:
                            print(env_index, task, type_index)
                            exit()
                    print("HN :", HN)
                    if HN == -1:
                        pass
                    else:
                        if HN != -2:
                            tmp_helping_num += 1
                        # 将特殊值转为实际上的0
                        if HN == -2:
                            HN = 0
                        tmp_HN_list.append(HN)
                    tmp_SR_list.append(SR)
                    tmp_GSR_list.append(GSR)
                    print(f"Total Reward = {episode_reward}")
                    SPL = SR * (test_env.goal_num / max(test_env.goal_num, test_env.step_count))
                    tmp_reward_list.append(episode_reward)
                    tmp_eposide_len_list.append(eposide_len)
                    tmp_SPL_list.append(SPL)
            task_dict[task][type_index]['SR'] = copy.deepcopy(tmp_SR_list)
            task_dict[task][type_index]['GSR'] = copy.deepcopy(tmp_GSR_list)
            task_dict[task][type_index]['CR'] = copy.deepcopy(tmp_CR_list)
            task_dict[task][type_index]['HE'] = copy.deepcopy(tmp_HE_list)
            task_dict[task][type_index]['HN'] = copy.deepcopy(tmp_HN_list)
            task_dict[task][type_index]['helping_num'] = copy.deepcopy(tmp_helping_num)
            task_dict[task][type_index]['need_help_num'] = copy.deepcopy(tmp_need_help_num)
            task_dict[task][type_index]['reward'] = copy.deepcopy(tmp_reward_list)
            task_dict[task][type_index]['eposide_len'] = copy.deepcopy(tmp_eposide_len_list)
            task_dict[task][type_index]['SPL'] = copy.deepcopy(tmp_SPL_list)
            print("=======END=======")
            print(f"task: {task}, type: {type_index}")
            print("average_SR : ", sum(tmp_SR_list) / len(tmp_SR_list))
            print("average_GSR: ", sum(tmp_GSR_list) / len(tmp_GSR_list))
            # print("average_CR : ", sum(tmp_CR_list) / len(tmp_CR_list))
            # print("average_HE : ", sum(tmp_HE_list) / max(len(tmp_HE_list), 1))
            print("average_HN : ", sum(tmp_HN_list) / max(len(tmp_HN_list), 1))
            print("average_reward: ", sum(tmp_reward_list) / len(tmp_reward_list))
            print("average_eposide_len: ", sum(tmp_eposide_len_list) / len(tmp_eposide_len_list))
            print("average_SPL: ", sum(tmp_SPL_list) / len(tmp_SPL_list))
            print("helping_num : ", tmp_helping_num)
            print("need_help_num:", tmp_need_help_num)
            SR_list = SR_list + tmp_SR_list
            GSR_list = GSR_list + tmp_GSR_list
            CR_list = CR_list + tmp_CR_list
            HE_list = HE_list + tmp_HE_list
            HN_list = HN_list + tmp_HN_list
            reward_list = reward_list + tmp_reward_list
            eposide_len_list = eposide_len_list + tmp_eposide_len_list
            SPL_list = SPL_list + tmp_SPL_list
            helping_num += tmp_helping_num
            need_help_num += tmp_need_help_num
    print("=======END=======")
    print("average_SR : ", sum(SR_list) / len(SR_list))
    print("average_GSR: ", sum(GSR_list) / len(GSR_list))
    print("average_HN : ", sum(HN_list) / max(len(HN_list), 1))
    print("average_reward: ", sum(reward_list) / len(reward_list))
    print("average_eposide_len: ", sum(eposide_len_list) / len(eposide_len_list))
    print("average_SPL: ", sum(SPL_list) / len(SPL_list))
    print("helping_num : ", helping_num)
    print("need_help_num:", need_help_num)
    total_dict['SR'] = SR_list
    total_dict['GSR'] = GSR_list
    total_dict['HN'] = HN_list
    total_dict['helping_num'] = helping_num
    total_dict['need_help_num'] = need_help_num
    total_dict['reward'] = reward_list
    total_dict['eposide_len'] = eposide_len_list
    total_dict['SPL'] = SPL_list

    with open("total.json", "w") as json_file:
        json.dump(total_dict, json_file)

    with open("task.json", "w") as json_file:
        json.dump(task_dict, json_file)
