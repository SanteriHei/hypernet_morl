""" Define the policy network"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn

from .. import structured_configs
from ..utils import configs, log, nets
from . import hypernet as hn


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
        n_layers = len(cfg.layer_features)
        apply_activation = tuple(
            i != n_layers - 1 for i in range(n_layers)
        )

        self._logger.debug("Creating a Latent policy...")
        self._latent_pi = nets.create_mlp(
            input_dim=cfg.reward_dim + cfg.obs_dim,
            layer_features=cfg.layer_features,
            activation_fn=cfg.activation_fn,
            apply_activation=apply_activation
        )

        self._logger.debug(
            "Mean and log-std layers | "
            f"{cfg.layer_features[-1]} -> {cfg.output_dim}"
        )
        self._mean_layer = nn.Linear(cfg.layer_features[-1], cfg.output_dim)
        self._log_std_layer = nn.Linear(cfg.layer_features[-1], cfg.output_dim)

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
            "_action_bias", (action_space_high + action_space_low) / 2.0
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

    @torch.no_grad
    def eval_action(
            self, obs: torch.Tensor, prefs: torch.Tensor
    ) -> torch.Tensor:
        """Take an "evaluation" action using the current policy. NOTE: no 
        gradients are tracked when taking evaluation actions.

        Parameters
        ----------
        obs : torch.Tensor
            The current observation
        prefs : torch.Tensor
            The current preferences over the objectives.

        Returns
        -------
        torch.Tensor
            The selected action.
        """
        mean, _ = self.forward(obs, prefs)
        return torch.tanh(mean) * self._action_scale + self._action_bias

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
        mean, _ = self.forward(obs, prefs)
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


class HyperPolicy(nn.Module):
    LOG_SIG_MAX: int = 2
    LOG_SIG_MIN: int = -20
    LOG_PROB_RANGE: float = 1e3
    EPS: float = 1e-6

    def __init__(self, cfg: structured_configs.PolicyConfig):
        """Creates a "hyper" policy that utilizes a hypernet for generating
        parameters for a Gaussian policy.


        Parameters
        ----------
        cfg : structured_configs.PolicyConfig
            The configuration for the policy network.
        """
        super().__init__()
        self._logger = log.get_logger("models.gaussian_policy")
        self._cfg = cfg

        self._embedding = hn.Embedding(
            embedding_layers=cfg.hypernet_cfg.embedding_layers,
        )
        self._policy_head = hn.HeadNet(
                hidden_dim=cfg.hypernet_cfg.head_hidden_dim,
                layer_features=cfg.layer_features,
                target_output_dim=cfg.output_dim,
                target_input_dim=cfg.obs_dim + cfg.reward_dim,
                n_outputs=2,
                init_method=cfg.hypernet_cfg.head_init_method,
                init_stds=cfg.hypernet_cfg.head_init_stds,

        )

        # Create the mask for the activation functions
        self._logger.debug(f"Using layers {cfg.layer_features}")
        self._activation_mask = np.arange(len(cfg.layer_features)) < len(cfg.layer_features) - 1
        self._logger.debug((f"Network architecture {cfg.layer_features} | "
                            f"activation mask {self._activation_mask}"))

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
            "_action_bias", (action_space_high + action_space_low) / 2.0
        )

    @property
    def config(self) -> structured_configs.PolicyConfig:
        """Get the configuration of the policy.

        Returns
        -------
        structured_configs.PolicyConfig
            The used configuration.
        """
        return self._cfg
    
    @torch.no_grad
    def eval_action(
            self, obs: torch.Tensor, prefs: torch.Tensor
    ) -> torch.Tensor:
        """Take an "evaluation" action with the current policy. NOTE: No
        gradient information is tracked when these actions are taken.

        Parameters
        ----------
        obs : torch.Tensor
            The current observation.
        prefs : torch.Tensor
            The current preferences over the objectives.

        Returns
        -------
        torch.Tensor
            The selected action.
        """
        mean, _ = self.forward(obs, prefs)
        return torch.tanh(mean) * self._action_scale + self._action_bias

    def forward(
            self, obs: torch.Tensor, prefs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the forward pass to the policy.

        Parameters
        ----------
        obs : torch.Tensor
            The current observation.
        prefs : torch.Tensor
            The current preferences over the objectives.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Returns the mean and log standard deviation for a Gaussian 
            distribution
        """
        z = self._embedding(obs, prefs)
        weights, biases, scales = self._policy_head(z)
        

        policy_input = torch.cat((obs, prefs), dim=-1)
        # If the data does not contain an batch dimension, add it into the input
        if policy_input.ndim == 1:
            policy_input.unsqueeze_(-2)


        h = nets.target_network(
            policy_input,
            weights=weights[:-2],
            biases=biases[:-2],
            scales=scales[:-2],
            apply_activation=self._activation_mask,
            activation_fn=self._cfg.activation_fn
        )
        
        mean = nets.target_network(
            h, weights=[weights[-1]], biases=[biases[-1]], scales=[scales[-1]],
            apply_activation=[False]
        )
        log_std = nets.target_network(
            h, weights=[weights[-2]], biases=[biases[-2]], scales=[scales[-2]],
            apply_activation=[False]
        )
        log_std = torch.clamp(
            log_std, min=HyperPolicy.LOG_SIG_MIN, max=HyperPolicy.LOG_SIG_MAX
        )
        
        # Remove the batch and singleton dimension if they are not needed
        mean.squeeze_([0, 2])
        log_std.squeeze_([0, 2])
        
        return mean, log_std

    def sample_action(
            self, obs: torch.Tensor, prefs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action using the current policy.

        Parameters
        ----------
        obs : torch.Tensor
            The current observation.
        prefs : torch.Tensor
            The current preferences over the objectives.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The action, the log probability of the sample and the mean of the
            Gaussian distribution.
        """
        mean, log_std = self.forward(obs, prefs)
        std = log_std.exp()
        
        # self._logger.debug(f"Taking actions from N({mean.max()}, {std.max()})")
        normal_distr = torch.distributions.Normal(mean, std)
        x_t = normal_distr.rsample()  # Reparametrization trick

        # Scale the sample to (-1, 1)
        y_t = torch.tanh(x_t)
        action = y_t * self._action_scale + self._action_bias

        log_prob = normal_distr.log_prob(x_t)

        # Enforce the action bounds
        # Compute the log-prob as the normal distribution sample which
        # is processed by tanh
        log_prob -= torch.log(
            self._action_scale * (1 - y_t.pow(2)) * HyperPolicy.EPS
        )
        log_prob = log_prob.sum(axis=1, keepdims=True)
        log_prob = log_prob.clamp(
            -HyperPolicy.LOG_PROB_RANGE, HyperPolicy.LOG_PROB_RANGE
        )

        mean = torch.tanh(mean) * self._action_scale + self._action_bias
        return action, log_prob, mean

    def take_action(
            self, obs: torch.Tensor, prefs: torch.Tensor
    ) -> torch.Tensor:
        """Take an (deterministic) action using the current policy 

        Parameters
        ----------
        obs : torch.Tensor
            The current observation.
        prefs : torch.Tensor
            The current preferences over the objectives.

        Returns
        -------
        torch.Tensor
            The selected action.
        """
        mean, _ = self.forward(obs, prefs)
        return torch.tanh(mean) * self._action_scale + self._action_bias
