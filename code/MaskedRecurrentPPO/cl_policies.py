from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.common.maskable.distributions import MaskableDistribution, make_masked_proba_distribution
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy



class MaskableRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        super(MaskableRecurrentActorCriticPolicy, self).__init__(*args, **kwargs)
        self.action_dist = make_masked_proba_distribution(self.action_space)
        
    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        
        actions, values, log_prob, new_lstm_states = super().forward(obs, lstm_states, episode_starts, deterministic)
        
        distribution = self.action_dist(actions)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob, new_lstm_states

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        action_masks: Optional[np.ndarray] = None
    ) -> Tuple[MaskableDistribution, Tuple[th.Tensor, ...]]:
        
        distribution, new_lstm_states = super().get_distribution(obs, lstm_states, episode_starts)
        
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        return distribution, new_lstm_states
    
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        return self.action_dist.proba_distribution(action_logits=mean_actions)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        
        self.set_training_mode(False)
        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions, _, _, _ = self.forward(observation, state, episode_start, deterministic, action_masks)
            actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions.squeeze(axis=0)
        return actions, None
