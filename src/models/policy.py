""" Define the policy network"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .. import structured_configs
from ..utils import configs, nets


class GaussianPolicy(nn.Module):

    _LOG_SIG_MAX: int = 2
    _LOG_SIG_MIN: int = -20
    _EPS: float = 1e-6

    def __init__(
            self, cfg: structured_configs.PolicyConfig,
    ):
        """Create a Gaussian policy. Utilizes reparmeterization trick 
        to predict the mean and std for the Gaussian distribution.

        Parameters
        ----------
        cfg: structured_configs.PolicyConfig
            The configuration for the GaussianPolicy
        """
        super().__init__()

        self._cfg = configs.as_structured_config(cfg)

        self._latent_pi = nets.create_mlp(
            input_dim=cfg.reward_dim + cfg.obs_dim,
            output_dim=cfg.network_arch[-1],
            network_arch=cfg.network_arch[:-1],
            activation_fn=cfg.activation_fn
        )

        self._mean_layer = nn.Linear(cfg.network_arch[-1], cfg.output_dim)
        self._log_std_layer = nn.Linear(cfg.network_arch[-1], cfg.output_dim)

        # for scaling the actions
        action_space_low = torch.as_tensor(cfg.action_space_low)
        action_space_high = torch.as_tensor(cfg.action_space_high)
        self.register_buffer(
            "_action_scale",
            (action_space_high - action_space_low) / 2.0
        )
        self.register_buffer(
            "_action_bias",
            (action_space_high - action_space_low) / 2.0
        )

    @property
    def config(self) -> structured_configs.PolicyConfig:
        """Returns the used configuration for the policy.

        Returns
        -------
        structured_configs.PolicyConfig
            The config.
        """
        return self._cfg

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
        h = self._latent_pi(torch.cat((obs, prefs), dim=-1))

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
        return torch.tanh(mean) * self._action_scale + self._action_bias

    def sample_action(
            self, obs: torch.Tensor, prefs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples an action from the current policy using the
        reparameterization trick. NOTE: this action is not deterministic.

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

        mean = torch.tanh(mean) * self._action_scale + self._action_bias

        return action, log_prob, mean
