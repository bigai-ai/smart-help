import argparse
import os

import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


from env.symbolic_env import SymbolicEnv
from env.symbolic_env_wo_pun import SymbolicEnvWoPun

from model.helper_v3 import HelperModel
from model.custom_callback import CustomCallbacks
from env.type import sample_type
from env.task import sample_task
from model.end2end import End2End
from model.helper_without_super import HelperModelWithoutSupervise
# from model.helper_without_opp import HelperModelWithoutOpp
from model.helper_flow import HelperModelFlow
from model.helper_v4 import HelperModelV4
from model.helper_new import HelperModelNew
from model.helper_dep import HelperModelDep
import shutil

import json

from constants import AgentType
import wandb
# from tqdm import tqdm

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--model", type=str, default="simple", help="The model to use in A2C, for testing only."
)
parser.add_argument(
    "--agent_num", type=int, default=2, help="The number of agents."
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


if __name__ == "__main__":
    os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda/lib64"
    print("Start the program", torch.cuda.is_available(), )
    save_path_dict = {}

    args = parser.parse_args()

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
            "renderDepthImage": False,
            "renderInstanceSegmentation": False,
            "visibilityDistance": 30, 
            "quality": "Very Low", 
        },
        'task': sample_task(),
    }

    ray.init()

    position_list = []

    algo = None

    if args.run == 'PPO':

        ModelCatalog.register_custom_model(
            "helper", HelperModel
        )

        ModelCatalog.register_custom_model(
            "helperv4", HelperModelV4
        )

        ModelCatalog.register_custom_model(
            "e2e", End2End
        )

        ModelCatalog.register_custom_model(
            "helperwithoutsuper", HelperModelWithoutSupervise
        )

        ModelCatalog.register_custom_model(
            "helperflow", HelperModelFlow
        )

        ModelCatalog.register_custom_model(
            "helpernew", HelperModelNew
        )

        ModelCatalog.register_custom_model(
            "helperdep", HelperModelDep
        )

        config_PPO = PPOConfig()
        config_PPO["preprocessor_pref"] = None

        algo = (
            config_PPO
            .rollouts(num_rollout_workers=6)
            # .rl_module( _enable_rl_module_api=False)
            .resources(num_gpus=1)
            .experimental(_enable_new_api_stack=False)
            # .framework("torch")
            .training(model={
                "use_lstm": True,
                "lstm_cell_size": 512,
                "custom_model": "helperdep", 
                "vf_share_layers": True,}, 
                train_batch_size=2000)
            .environment(SymbolicEnvWoPun, env_config=config)
            .callbacks(CustomCallbacks)
            .build()
        )

        # algo = Algorithm.from_checkpoint('/home/zhihao/Downloads/ours/1834')
            # # my_restored_policy = Policy.from_checkpoint('/home/zhihao/ray_results/helper_failed/checkpoint_000955')

        print("algo build finished!!!")

    for i in range(2000):
        # inner_progress_bar = tqdm(total=4000, desc=f"Iteration {i+1}")
        print("Training Start")
        result = algo.train()
        print(pretty_print(result))
        log_dict = {}
        for k, v in result.items():
            if v is not None:
                log_dict[str(k)] = v
        # wandb.log(log_dict)

        if i % 5 == 4:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir.checkpoint.path}")
            save_path_dict[i] = checkpoint_dir.checkpoint.path
            with open("save_path.json", "w") as json_file:
                json.dump(save_path_dict, json_file)
            shutil.copytree(checkpoint_dir.checkpoint.path, f"/home/zhihao/Downloads/ours_wo_pun/{i}/")
