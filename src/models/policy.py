""" Define the policy network"""
from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import torch
from torch import nn

from .utils import nets


class GaussianPolicy(nn.Module):

    _LOG_SIG_MAX: int = 2
    _LOG_SIG_MIN: int = -20
    _EPS: float = 1e-6

    def __init__(
            self,
            obs_dim: int,
            reward_dim: int,
            output_dim: int,
            network_architecture: Tuple[int, ...],
            action_space: gym.spaces.Box,
            activation_fn: torch.nn.Module | str = "relu"
    ):
        """Create a Gaussian policy. Utilizes reparmeterization trick 
        to predict the mean and std for the Gaussian distribution.

        Parameters
        ----------
        obs_dim: int
           The dimension of the observations.
        reward_dim: int
            the dimension of the rewards.
        output_dim : int
            The dimension of the mean and std
        network_architecture : Tuple[int, ...]
            The architechture of the network
        action_space : gym.spaces.Box
            The action space of the used environment
        activation_fn : torch.nn.Module | str
            The activation function to use. Default "relu"
        """
        super().__init__()

        self._latent_pi = nets.create_mlp(
            input_dim=reward_dim + obs_dim,
            output_dim=network_architecture[-1],
            network_arch=network_architecture[:-1],
            activation_fn=activation_fn
        )

        self._mean_layer = nn.Linear(network_architecture[-1], output_dim)
        self._log_std_layer = nn.Linear(network_architecture[-1], output_dim)

        # for scaling the actions
        self.register_buffer(
            "action_scale",
            torch.tensor((action_space.high - action_space.low) / 2.0)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space.high - action_space.low) / 2.0)
        )

    def forward(
            self, obs: torch.Tensor, prefs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of the policy network

        Parameters
        ----------
        obs : torch.Tensor
            The observation from the environment.
        prefs : torch.Tensor
            The current preferences over the objectives.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The mean and logarithm of the standard deviation for the 
            Gaussian distribution
        """
        h = self._network(torch.cat((obs, prefs), dim=-1))

        mean = self._mean_layer(h)
        log_std = self._log_std_layer(h)
        # Apply clamping as in the original paper
        log_std = torch.clamp(
                log_std, min=GaussianPolicy._LOG_SIG_MIN,
                max=GaussianPolicy._LOG_SIG_MAX
        )
        return mean, log_std

    def take_action(
            self, obs: torch.Tensor, prefs: torch.Tensor
    ) -> torch.Tensor:
        """Takes an action using the current policy.

        Parameters
        ----------
        obs : torch.Tensor
            The observation from the environment.
        prefs : torch.Tensor
            The current preferences over the objectives

        Returns
        -------
        torch.Tensor
            The chosen action.
        """
        mean, __ = self.forward(obs, prefs)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def sample_action(
            self, obs: torch.Tensor, prefs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples an action from the current policy using the reparmeterization
        trick. NOTE: this action is not deterministic.

        Parameters
        ----------
        obs : torch.Tensor
            The observation from the environment
        prefs : torch.Tensor
            The current preferences over the objectives

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Returns the taken action, the mean and the (log) std for the 
            used Gaussian distribution.
        """
        mean, log_std = self.forward(obs, prefs)
        std = log_std.exp()

        normal_distr = torch.distributions.Normal(mean, std)
        x_t = normal_distr.rsample()  # reparmeterization trick

        # Convert the sample to the (-1, 1) scale
        y_t = torch.tanh(x_t)
        action = y_t * self._action_scale + self._action_bias

        log_prob = normal_distr.log_prob(x_t)

        # TODO: is this correct?
        log_prob -= torch.log(
                self._action_scale * (1 - y_t.pow(2)) + GaussianPolicy._EPS
        )
        log_prob = log_prob.sum(axis=1, keepdim=True)
        log_prob = log_prob.clamp(-1e3, 1e3)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean
