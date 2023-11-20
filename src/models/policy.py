""" Define the policy network"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .. import structured_configs
from ..utils import configs, nets, log


class GaussianPolicy(nn.Module):

    LOG_SIG_MAX: int = 2
    LOG_SIG_MIN: int = -20
    LOG_PROB_RANGE: float = 1e3
    EPS: float = 1e-6

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
        self._logger = log.get_logger("models.gaussian_policy")
        self._cfg = configs.as_structured_config(cfg)

        # Do not apply activation function after the last layer
        n_layers = len(cfg.network_arch)
        apply_activation = tuple(
                i != n_layers - 1 for i in range(n_layers)
        )


        self._latent_pi = nets.create_mlp(
            input_dim=cfg.reward_dim + cfg.obs_dim,
            output_dim=cfg.network_arch[-1],
            network_arch=cfg.network_arch[:-1],
            activation_fn=cfg.activation_fn,
            apply_activation=apply_activation
        )
        
        self._logger.debug(
                "Mean and log-std layers | "
                f"{cfg.network_arch[-1]} -> {cfg.output_dim}"
        )
        self._mean_layer = nn.Linear(cfg.network_arch[-1], cfg.output_dim)
        self._log_std_layer = nn.Linear(cfg.network_arch[-1], cfg.output_dim)

        # for scaling the actions
        action_space_low = torch.as_tensor(
                cfg.action_space_low, dtype=torch.float32
        )
        action_space_high = torch.as_tensor(
                cfg.action_space_high, dtype=torch.float32
        )
        self.register_buffer(
            "_action_scale", (action_space_high - action_space_low) / 2.0
        )
        self.register_buffer(
            "_action_bias", (action_space_high - action_space_low) / 2.0
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
            log_std,
            min=GaussianPolicy.LOG_SIG_MIN,
            max=GaussianPolicy.LOG_SIG_MAX
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
        self._logger.debug(f"reparametrization trick x_t shape {x_t.shape}")

        # Convert the sample to the (-1, 1) scale
        y_t = torch.tanh(x_t)
        action = y_t * self._action_scale + self._action_bias

        log_prob = normal_distr.log_prob(x_t)
        self._logger.debug(f"log_prob shape {log_prob.shape}")

        # Enforce the action bounds
        # Compute the log prob as the normal distr sample which is processed by 
        # tanh
        log_prob -= torch.log(
            self._action_scale * (1 - y_t.pow(2)) + GaussianPolicy.EPS
        )
        log_prob = log_prob.sum(axis=1, keepdim=True)
        log_prob = log_prob.clamp(
                -GaussianPolicy.LOG_PROB_RANGE, GaussianPolicy.LOG_PROB_RANGE
        )

        mean = torch.tanh(mean) * self._action_scale + self._action_bias

        return action, log_prob, mean
