""" Define different critic models for MORL """

import numpy as np
import torch
from torch import nn

from .. import structured_configs
from ..utils import log, nets
from . import hypernet as hn


class HyperCritic(nn.Module):

    def __init__(
            self, cfg: structured_configs.CriticConfig
    ):
        """Create a multi-objective critic network that utilizes a hypernetwork
        for generating it parameters.

        Parameters
        ----------
        cfg : structured_configs.HypernetConfig
            The configuration for the critic.
        """
        super().__init__()
        self._logger = log.get_logger("models.hyper_critic")
        self._cfg = cfg

        self._logger.debug("Creating embedding...")

        self._embeddeding = hn.Embedding(
            embedding_layers=cfg.hypernet_cfg.embedding_layers
        )

        target_input_dim = self._get_target_input_dim()
        self._logger.debug("Creating the headnet...")
        self._critic_head = hn.HeadNet(
            hidden_dim=cfg.hypernet_cfg.head_hidden_dim,
            target_input_dim=target_input_dim,
            target_output_dim=cfg.reward_dim,
            layer_features=cfg.layer_dims,
            n_outputs=1,
            init_method=cfg.hypernet_cfg.head_init_method,
            init_stds=cfg.hypernet_cfg.head_init_stds
        )

        # create mask, such that the activation function is not applied at the
        # latest layer
        self._activation_mask = np.arange(
                len(cfg.layer_dims)
        ) < len(cfg.layer_dims) - 1

    @property
    def config(self) -> structured_configs.CriticConfig:
        ''' Get the configuration of the critic '''
        return self._cfg

    def forward(
            self, obs: torch.Tensor, action: torch.Tensor, prefs: torch.Tensor
    ) -> torch.Tensor:
        """Apply forward pass for the critic.

        Parameters
        ----------
        obs : torch.Tensor
            The current observation.
        action : torch.Tensor
            The action taken by the actor.
        prefs : torch.Tensor
            The current preferences over the objectives.

        Returns
        -------
        torch.Tensor
            The estimated Q(s, a, w) value with shape (batch_dim, reward_dim)
        """
        z = self._embeddeding(obs, prefs)
        weights, biases, scales = self._critic_head(z)

        self._logger.debug("Generated params")
        for w, b, s in zip(weights, biases, scales):
            self._logger.debug(f"W {w.shape} | B: {b.shape} | S {s.shape}")

        target_net_input = self._get_target_input(obs, action, prefs)
        self._logger.debug(f"Critic input shape {target_net_input.shape}")
        out = nets.target_network(
            target_net_input, weights=weights, biases=biases,
            scales=scales, apply_activation=self._activation_mask,
            activation_fn=self._cfg.activation_fn
        )
        # Remove the singleton dimension
        return out.squeeze(2)

    def _get_target_input_dim(self) -> int:
        """Get the input dimension for the target network, depending the 
        input configuration the user defined.

        Returns
        -------
        int
            The input dimension of the target network.
        """
        if (
                not self._cfg.use_obs and
                not self._cfg.use_action and
                not self._cfg.use_prefs
        ):
            raise ValueError(("Atleast one of 'use_obs', 'use_action' and "
                              "'use_prefs' must be True"))

        input_dim = None

        if self._cfg.use_action and self._cfg.use_prefs and self._cfg.use_obs:
            input_dim = self._cfg.obs_dim + self._cfg.action_dim + self._cfg.reward_dim
        elif self._cfg.use_action and self._cfg.use_prefs:
            input_dim = self._cfg.action_dim + self._cfg.reward_dim
        elif self._cfg.use_action and self._cfg.use_obs:
            input_dim = self._cfg.obs_dim + self._cfg.action_dim
        elif self._cfg.use_prefs and self._cfg.use_obs:
            input_dim = self._cfg.reward_dim + self._cfg.obs_dim
        elif self._cfg.use_obs:
            input_dim = self._cfg.obs_dim
        elif self._cfg.use_action:
            input_dim = self._cfg.action_dim
        elif self._cfg.use_prefs:
            input_dim = self._cfg.reward_dim

        assert input_dim is not None, \
            (f"Unknown combination of inputs: Obs {self._cfg.use_obs} | "
             f"Action {self._cfg.use_action} | Prefs: {self._cfg.use_prefs}")

        return input_dim

    def _get_target_input(
            self, obs: torch.Tensor, action: torch.Tensor, prefs: torch.Tensor
    ) -> torch.Tensor:
        """Create the input for the target network based on the use
        configuration.

        Parameters
        ----------
        obs : torch.Tensor
            The current observation.
        action : torch.Tensor
            The current action.
        prefs : torch.Tensor
            The current preferences over the objectives.

        Returns
        -------
        torch.Tensor
            The composed input for the target network.
        """
        if (
                not self._cfg.use_obs and
                not self._cfg.use_action and
                not self._cfg.use_prefs
        ):
            raise ValueError(("Atleast one of 'use_obs', 'use_action' and "
                              "'use_prefs' must be True"))

        out = None

        if self._cfg.use_action and self._cfg.use_prefs and self._cfg.use_obs:
            out = torch.cat((obs, action, prefs), dim=-1)
        elif self._cfg.use_action and self._cfg.use_prefs:
            out = torch.cat((obs, prefs), dim=-1)
        elif self._cfg.use_action and self._cfg.use_obs:
            out = torch.cat((obs, action), dim=-1)
        elif self._cfg.use_prefs and self._cfg.use_obs:
            out = torch.cat((obs, action), dim=-1)
        elif self._cfg.use_obs:
            out = obs
        elif self._cfg.use_action:
            out = action
        elif self._cfg.use_prefs:
            out = prefs

        assert out is not None, \
            (f"Unknown critic input config: obs: {self._cofg.use_obs} | "
             f"Action {self._cfg.use_action} |  Prefs {self._cfg.use_prefs}")
        return out
