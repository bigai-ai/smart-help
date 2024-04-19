import h5py
import numpy as np
from tqdm import tqdm
import random
import os

import sys
sys.path.append("/home/zhihao/文档/GitHub/rllib_A2SP")

from env.symbolic_env_data_gen import SymbolicEnvGen
from env.type import sample_type
from env.task import sample_task
from env.goal import GOAL_NUM

early_end = 30

def collect_data(env: SymbolicEnvGen):
    # 这个函数应当返回一个33x11的矩阵和一个标签
    action_name = random.randint(0, GOAL_NUM - 1)
    tar_index = random.randint(0, len(env.object_name2index)-1)
    action = {
        "goal": action_name, 
        "tar_index": tar_index
    }
    obs, _,  done, _, _ = env.step(action)
    return obs, done

config = {
    'agents_num': 2,
    # 'agents_type': {0: AgentType.AGENT_WITH_FULL_CAPABILITIES, 1: AgentType.AGENT_WITH_TOGGLE_ISSUES},
    'agents_type': {0: [2, 2, 1, 1, 1, 1], 1: [1, 1, 1, 1, 1, 1]},
    'main_agent_id': 1,
    'mode': 'train',
    'controller_kwargs': {
        "agentCount": 2,
        "scene": 'FloorPlan2',
        "local_executable_path": "/home/zhihao/Downloads/thor-Linux64-local/thor-Linux64-local",
        # "local_executable_path": "/home/zhihao/A2SP/thor-Linux64-local/thor-Linux64-local",
        "renderDepthImage": False,
        "renderInstanceSegmentation": False,
        "visibilityDistance": 30, 
        "quality": "Very Low", 
        # "platform": "CloudRendering", 
    },
    'task': sample_task(),
}

env = SymbolicEnvGen(config=config)
data_length = 4000

with tqdm(total=data_length, desc="Processing items") as pbar:
    for k in range(data_length):

        while os.path.exists(f'./dataset_new/trajectories_image_{k}.h5'):
            k = k + 100

        if k > 4000:
            break

        # 创建一个h5文件并写入数据
        with h5py.File(f'./dataset_new/trajectories_image_{k}.h5', 'w') as f:
            # 创建数据集
            data_set = f.create_dataset('data', (0, early_end, 33, 17), maxshape=(32, early_end, 33, 17))
            
            # # 开始采集数据
            # for i in range(0, 32):  # 收集10000条数据
            sequence = np.zeros((early_end, 33, 17))  # 初始化一个新的序列
            image_sequence = np.zeros((early_end, 3, 224, 224))
            done = False
            for j in range(early_end):  # 每个序列有200个步骤
                # if not done:
                data, done = collect_data(env)
                sequence[j] = data[0]
                image_sequence[j] = data[1] 
            pbar.update(1)
            env.reset()
            # 将数据添加到数据集中
            data_set.resize((1, early_end, 33, 17))
            data_set[0] = sequence

            # 创建图像数据集
            image_set = f.create_dataset('image', data=image_sequence, compression='gzip')