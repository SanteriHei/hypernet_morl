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
        if cfg.hypernet_type == "mlp":
            self._embedding = nets.create_mlp(
                    input_dim=cfg.hypernet_cfg.input_dim,
                    layer_features=cfg.hypernet_cfg.layer_features,
                    activation_fn=cfg.hypernet_cfg.activation_fn,
                    apply_activation=cfg.hypernet_cfg.apply_activation,
                    dropout_rate=cfg.hypernet_cfg.dropout_rates
            )
        else:
            self._embedding = hn.Embedding(
                embedding_layers=cfg.hypernet_cfg.embedding_layers
            )

        self._logger.debug(self._embedding)
        
        target_input_dim = nets.get_network_input_dim(
                [
                    ("action" in self._cfg.target_net_inputs, self._cfg.action_dim),
                    ("prefs" in self._cfg.target_net_inputs, self._cfg.reward_dim),
                    ("obs" in self._cfg.target_net_inputs, self._cfg.obs_dim)
                ]
        )
        if target_input_dim == 0:
            raise ValueError(("Atleast one of 'use_obs', 'use_action' and "
                              "'use_prefs' must be True"))

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
        n_layers = len(cfg.layer_dims)
        self._activation_mask = np.arange(n_layers + 1) < n_layers
        self._logger.debug(f"Activation mask {self._activation_mask}")

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
        z = self._embedding(torch.cat((obs, prefs), dim=-1))
        self._logger.debug(f"Z shape {z.shape}")
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


    def get_dynamic_net_params(
            self, obs: torch.Tensor, prefs: torch.Tensor
    ):
        z = self._embedding(torch.cat((obs, prefs), dim=-1))
        weights, biases, scales = self._critic_head(z)
        return [
                {"weight": w, "bias": b, "scale": s}
                for w, b, s in zip(weights, biases, scales)
        ]


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

        out = []
        for target_input in self._cfg.target_net_inputs:
            self._logger.debug(f"Adding {target_input} to target input")
            match target_input:
                case "obs":
                    out.append(obs)
                case "action":
                    out.append(action)
                case "prefs":
                    out.append(prefs)
                case _:
                    raise ValueError(f"Unknown target input {target_input!r}!")
        if len(out) == 1:
            return out[0]
        return torch.cat(out, dim=-1)
