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

import cv2
import json
from env.symbolic_env import SymbolicEnv
from constants import AgentType
from model.helper_v3 import HelperModel
from env.type import sample_type
from env.task import sample_task
from model.end2end import End2End

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


config = {
    'agents_num': 2,
    # 'agents_type': {0: AgentType.AGENT_WITH_FULL_CAPABILITIES, 1: AgentType.AGENT_WITH_TOGGLE_ISSUES},
    'agents_type': {0: [1, 1, 1, 1, 1, 1], 1: sample_type()},
    'main_agent_id': 1,
    'mode': 'train',
    'controller_kwargs': {
        "agentCount": 2,
        "scene": 'FloorPlan2',
        "local_executable_path": "/home/zhihao/A2SP/thor-Linux64-local_1/thor-Linux64-local/thor-Linux64-local",
        # "local_executable_path": "/home/zhihao/A2SP/thor-Linux64-local/thor-Linux64-local",
        "renderDepthImage": True,
        "renderInstanceSegmentation": True,
        "visibilityDistance": 30, 
        "quality": "Very Low", 
        # "platform": "CloudRendering", 
    },
    'task': sample_task(),
}

if __name__ == "__main__":
    args = parser.parse_args()
    tmp_position_helper = []
    tmp_position_main = []

    # ModelCatalog.register_custom_model(
    #         "testmodel", MyCustomModel
    #     )

    ModelCatalog.register_custom_model(
        "helper", HelperModel
    )

    ModelCatalog.register_custom_model(
        "e2e", End2End
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

    algo = Algorithm.from_checkpoint('/home/zhihao/ray_results/pretrain_freeze/checkpoint_000400')

    # for i in range(100):
    #     result = algo.train()
    #     print(pretty_print(result))

    #     if i % 5 == 0:
    #         checkpoint_dir = algo.save()
    #         print(f"Checkpoint saved in directory {checkpoint_dir}")

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
    helping_num = 0
    need_help_num = 0
    for episode in range(num_episodes):
        observation, _ = test_env.reset()
        if test_env.need_help:
            need_help_num += 1
        print("==========A NEW ENV=========")
        print("Agent type", test_env._agent_type[1])
        print("required type", test_env.necessary_capability(test_env.subtask_list))
        lstm_state = algo.get_policy().get_initial_state()
        done = False
        episode_reward = 0
        while not done:
            action, lstm_state, _ = algo.compute_single_action(observation, state=lstm_state)
            observation, reward, done, _, _ = test_env.step(action)
            episode_reward += reward
            # time.sleep(1)
        print("Evaluation")
        SR = int(not (test_env.step_count == 200))
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
            helping_num += 1
            HE_list.append(HE)
            HN_list.append(HN)
        SR_list.append(SR)
        GSR_list.append(GSR)
        CR_list.append(CR)
        print(f"Episode {episode+1}: Total Reward = {episode_reward}")
    print("=======END=======")
    print("average_SR : ", sum(SR_list) / len(SR_list))
    print("average_GSR: ", sum(GSR_list) / len(GSR_list))
    print("average_CR : ", sum(CR_list) / len(CR_list))
    print("average_HE : ", sum(HE_list) / max(len(HE_list), 1))
    print("average_HN : ", sum(HN_list) / max(len(HN_list), 1))
    print("helping_num : ", helping_num)
    print("need_help_num:", need_help_num)

    # dict = {
    #     'position_list': position_list,
    #     'target_pos': target_pos
    # }

    # # 指定要保存的JSON文件路径
    # json_file_path = "./file.json"

    # # 使用json.dump()将list保存为JSON文件
    # with open(json_file_path, "w") as json_file:
    #     json.dump(dict, json_file)



            