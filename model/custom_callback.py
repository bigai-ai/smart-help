from typing import Dict, Optional, Union
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy.sample_batch import SampleBatch
from model.custom_policy import CustomPolicy
from env.symbolic_env import SymbolicEnv


class CustomCallbacks(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.SR_list = []
        self.GSR_list = []
        self.CR_list = []
        self.HE_list = []
        self.HN_list = []

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        if isinstance(policy, CustomPolicy):
            result["callBackOk"] = True
            result["SR_list"] = self.SR_list
            result["GSR_list"] = self.GSR_list
            result["CR_list"] = self.CR_list
            result["HE_list"] = self.HE_list
            result["HN_list"] = self.HN_list

            policy.reset_record()

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index,
                               **kwargs)->None:
        base_env = base_env.get_sub_environments()[0]
        self.SR_list.append(int(not (base_env.step_count == 200)))
        if base_env.goal_num == 0:
            self.GSR_list.append(-1)
        else:
            self.GSR_list.append(base_env.finish_goal_num / base_env.goal_num)
        if base_env.finish_goal_num == 0:
            self.CR_list.append(-1)
        else:
            self.CR_list.append(base_env.helper_finish_goal_num / base_env.finish_goal_num)
        if base_env.helper_action_num == 0:
            self.HE_list.append(-1)
        else:
            self.HE_list.append(base_env.helper_finish_required_action_num / base_env.helper_action_num)
        if base_env.helper_finish_goal_num == 0:
            self.HN_list.append(-1)
        else:
            self.HN_list.append(base_env.helper_finish_necc_goal_num / base_env.helper_finish_goal_num)
        pass
