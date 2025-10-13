from ray.rllib.algorithms.ppo import PPOTorchPolicy
import logging
from typing import Dict, List, Type, Union

import ray
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import TensorType

import torch
import torch.nn as nn
import wandb
import copy

logger = logging.getLogger(__name__)


# 试图将opponent modelling 中预测的agent type和goal加入到ppo的训练
class CustomPolicy(PPOTorchPolicy):
    def __init__(self, *args, **kwargs):
        self.pre_loss_record = 0
        self.estimate_loss_record = 0
        self.total_loss_record = 0
        self.add_num = 1
        self.subtask_num = 0
        self.tar_index_1_num = 0
        self.tar_index_2_num = 0
        self.type_distance = 0
        self.total_num = 0
        super().__init__(*args, **kwargs)

    def reset_record(self):
        self.pre_loss_record = 0
        self.estimate_loss_record = 0
        self.total_loss_record = 0
        self.add_num = 1
        self.subtask_num = 0
        self.tar_index_1_num = 0
        self.tar_index_2_num = 0
        self.type_distance = 0
        self.total_num = 0

    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Here we try to compute the loss between estimated agent type and goal with ground truth
        # we assume that the model is our custom helper model
        # print("for debug pre loss", total_loss)
        pre_loss = total_loss
        pre_loss = pre_loss.detach().cpu()
        self.pre_loss_record = self.pre_loss_record * (self.add_num - 1) / self.add_num + pre_loss / self.add_num
        estimate_loss, subtask_num, tar_index_1_num, tar_index_2_num, type_distance, total_num = self.model.get_estimate_loss(train_batch)
        self.subtask_num += subtask_num
        self.tar_index_1_num += tar_index_1_num
        self.tar_index_2_num += tar_index_2_num
        self.type_distance += type_distance
        self.total_num += total_num
        total_loss = total_loss + estimate_loss * 0.0001
        total_loss_for_compute = total_loss
        
        estimate_loss = estimate_loss.detach().cpu()
        total_loss_for_compute = total_loss_for_compute.detach().cpu()
        
        self.estimate_loss_record = self.estimate_loss_record * (self.add_num - 1) / self.add_num + estimate_loss / self.add_num
        self.total_loss_record = self.total_loss_record * (self.add_num - 1) / self.add_num + total_loss_for_compute / self.add_num
        self.add_num += 1
        # 移动平均，最后每一次只拿出一个loss
        # self.pre_loss_list.append(pre_loss)
        # self.estimate_loss_list.append(estimate_loss)
        # self.total_loss_list.append(total_loss)

        # print("for debug after loss", total_loss)
        # raise Exception

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["estimate_loss"] = estimate_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss
    