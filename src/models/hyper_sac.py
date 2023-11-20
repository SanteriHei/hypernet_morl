""" Define Hyper-SAC that utilizes hypernetworks for both actor and the critic
"""
from __future__ import annotations

import pathlib

import torch
from torch import nn

from ..utils import common, nets
from . import hypernet


class HyperSAC:

    def __init__(
            self, cfg, policy_cfg, hypernet_cfg
    ):

        self._embedding = hypernet.Embedding()

        # Define heads for Critics
        self._critics_heads = [hypernet.HeadV2() for _ in range(2)]
        self._target_critic_heads = [hypernet.HeadV2() for _ in range(2)]

        # TODO: Add additional "head" for the mean and variance of the
        # Gaussian policy
        self._policy_head = hypernet.HeadV2()

    @property
    def config(self):
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        raise NotImplementedError()

    def save(self, dir_path: str | pathlib.Path):
        raise NotImplementedError()

    def take_action(
            self, obs: torch.Tensor, prefs: torch.Tensor
    ):
        z = self._embedding(obs, prefs)
        weights, biases, scales = self._policy_head(z)
        return nets.target_network(
                obs, weights=weights, biases=biases, scales=scales
        )


    def update(self, replay_sample: common.ReplaySample):
        replay_sample = replay_sample.as_tensors(self._device)
        
        # Calculate the target Q-values
        with torch.no_grad():
            next_z = self._embedding(
                    replay_sample.next_obs, replay_sample.prefs
            )

            poly

        
            
        raise NotImplementedError()

    def _sample_action(self, obs: torch.Tensor, prefs: torch.Tensor):
        z = self._embedding(obs, prefs)
        weights, biases, scales = self._policy_head(z)
        return nets.target_network(
                obs, weights=weights, biases=biases, scales=scales
        )
