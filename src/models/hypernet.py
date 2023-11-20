""" Define the hypernetwork model for the MSA-hyper """
from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .. import structured_configs
from ..utils import common, configs, nets


class HeadV2(nn.Module):
    def __init__(
            self, *, hidden_dim: int, target_input_dim: int,
            target_output_dim: int,
            network_arch: Tuple[int, ...]
    ):
        super().__init__()

        self._weight_layers = nn.ModuleList()
        self._bias_layers = nn.ModuleList()
        self._scale_layers = nn.ModuleList()

        network_arch = (hidden_dim, *network_arch)

        for in_dim, out_dim in common.iter_pairwise(network_arch):
            self._weight_layers.append(nn.Linear(in_dim, out_dim))
            self._bias_layers.append(nn.Linear(in_dim, out_dim))
            self._scale_layers.append(nn.Liner(in_dim, out_dim))

        # Lastly add the layer that actually ouputs the weights
        self._weight_layers.append(
            nn.Linear(out_dim, target_input_dim*target_output_dim)
        )

        self._bias_layers.append(nn.Linear(out_dim, target_output_dim))
        self._scale_layers.append(nn.Liner(out_dim, target_output_dim))

    def forward(self, x: torch.Tensor):
        iter = zip(self._weight_layers, self._bias_layers, self._scale_layers)
        weights = []
        biases = []
        scales = []
        for weight_l, bias_l, scale_l in iter:
            weights.append(weight_l(x))
            biases.append(bias_l(x))
            scales.append(scale_l(x))
        return weights, biases, scales


class Head(nn.Module):
    def __init__(
            self, *,
            target_input_dim: int,
            target_output_dim: int,
            hidden_dim: int,
            init_std: float = 0.05
    ):
        """
        The "Head" that is used to generate the weights for a single linear
        layer

        Parameters
        ----------
        input_dim : int
            The input dimension for the target network (i.e action dimension)
        output_dim : int
            The output dimension of the target network.
        hidden_dim: int
            The size of the hidden dimension used in the layers.
        """
        super().__init__()

        self._target_input_dim = target_input_dim
        self._target_output_dim = target_output_dim

        self._weight_layer = nn.Linear(
            hidden_dim, target_input_dim * target_output_dim
        )
        self._bias_layer = nn.Linear(hidden_dim, target_output_dim)
        self._scale_layer = nn.Linear(hidden_dim, target_output_dim)

        self._init_weights(init_std)

    def get_bias_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get the shapes of biases in each used layer.

        Returns
        -------
        Dict[str, Tuple[int, ...]]
            The shapes of the biases for the weight, bias and scale layers.
        """
        return {
            "weight": self._weight_layer.bias.shape,
            "bias": self._bias_layer.bias.shape,
            "scale": self._scale_layer.bias.shape
        }

    def get_weight_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get the shapes of weights in each layer.

        Returns
        -------
        Dict[str, Tuple[int, ...]]
            The shapes of the weights for the weight, bias and scale layers.
        """
        return {
            "weight": self._weight_layer.weight.shape,
            "bias": self._bias_layer.weight.shape,
            "scale": self._bias_layer.weight.shape
        }

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the weights, bias and a scale for a Linear neural network
        layer.

        Parameters
        ----------
        x : torch.tensor
            The "meta input"

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The weights, bias and scale for the linear layer.

        """
        weights = self._weight_layer(x).view(
            -1, self._target_output_dim, self._target_input_dim
        )

        # Add singleton dimesion to both bias and scale
        bias = self._bias_layer(x).view(-1, self._target_output_dim, 1)
        scale = 1.0 + self._scale_layer(x).view(-1, self._target_output_dim, 1)
        return weights, bias, scale

    @torch.no_grad()
    def _init_weights(self, std: float):
        nn.init.uniform_(self._weight_layer.weight, -std, std)
        nn.init.uniform_(self._scale_layer.weight, -std, std)
        nn.init.uniform_(self._bias_layer.weight, -std, std)

        # Initialize biases to zeros
        nn.init.zeros_(self._weight_layer.bias)
        nn.init.zeros_(self._scale_layer.bias)
        nn.init.zeros_(self._bias_layer.bias)


class ResBlock(nn.Module):
    def __init__(
            self, *,
            input_dim: int,
            output_dim: int,
            network_arch: Tuple[int, ...],
            activation_fn: str | Callable = "relu"
    ):
        """Create a residual block that consists of a MLP.

        Parameters
        ----------
        input_dim : int
            The input dimension of the residual block.
        output_dim : int
            The ouput dimension of the residual block.
        network_arch : Tuple[int, ...]
            The architecture of the MLP as a tuple of neuron counts for each
            layer.
        activation_fn : str | Callable
            The activation function to use. Default "relu"
        """
        super().__init__()

        assert (n_layers := len(network_arch)) >= 1, \
           f"Expected atleast 2 layer resblock, got {n_layers} layers instead"

        # Do not apply activation function to the last layer.
        apply_activation = tuple(
            i != len(network_arch) for i in range(len(network_arch) + 1)
        )

        self._network = nets.create_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            network_arch=network_arch,
            activation_fn=activation_fn,
            apply_activation=apply_activation
        )
        self.apply(nets.init_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forwad pass of the Residual block

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        Returns
        -------
        torch.Tensor
            The output of the residual block.
        """
        x = F.relu(x)
        return x + self._network(x)


class Embedding(nn.Module):

    def __init__(
        self, *,
        reward_dim: int,
        obs_dim: int,
        resblock_arch: Tuple[structured_configs.ResblockConfig, ...],
    ):
        """Generate an "embedding" layer that transfers the current observation
        and preferences over the objectives into latent variable.

        Parameters
        ----------
        reward_dim : int
           The reward dimension 
        obs_dim : int
            The observation dimesion
        z_dim : int
            The latent dimension
        resblock_arch : Tuple[ResblockConfig, ...]
            The configuration for each linear + n residual blocks combos
        """
        super().__init__()

        warnings.warn(
            ("Parameters 'reward_dim' and 'obs_dim' are deprecated, and "
             " they will be replaced by the input dim of the first resblock"
             ), DeprecationWarning)

        self._hypernet = self._construct_network(
            resblock_arch
        )
        self._init_layers()

    def forward(
            self, obs: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Generate the hiddegymnasiumn presentation of the state weights 
        that is used to generated the weights for a linear layer.

        Parameters
        ----------
        state : torch.Tensor
            The current state of the environment.
        weights : torch.tensor
            The current preferences over the objectives

        Returns
        -------
        torch.Tensor
           The hidden presentation of the input state. Has shape
           (batch_size, ) (which was specified in the constructor)
        """

        # Condition the network on the preferences
        x = self._hypernet(torch.cat((obs, weights), dim=-1))
        return x
    
    @torch.no_grad()
    def _init_layers(self):
        for module in self._hypernet.modules():
            if isinstance(module, nn.Linear):
                # Bit hacky to use a private function for this!
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(2 * fan_in)
                nn.init.uniform_(module.weight, -bound, bound)

    def _construct_network(
            self,
            residual_network_arch: Tuple[structured_configs.ResblockConfig, ...]
    ) -> nn.Module:
        """Construct the embeddeding network that consists of multiple
        residual blocks.


        Parameters
        ----------
        residual_network_arch : Tuple[structured_configs.ResblockConfig, ...]
            The architecture for the network.

        Returns
        -------
        nn.Module
            The compiled sequential torch module containing the desired layers.
        """
        blocks = []
        for i, block_cfg in enumerate(residual_network_arch):
            assert len(block_cfg.network_arch) > 0, \
                "Resblock needs to have atleast one layer"
            blocks.append(
                nn.Linear(
                    block_cfg.input_dim,
                    block_cfg.network_arch[0]
                )
            )
            for ii in range(block_cfg.n_resblocks):
                output_dim = (block_cfg.output_dim
                              if ii == block_cfg.n_resblocks - 1 else
                              block_cfg.network_arch[-1])
                blocks.append(ResBlock(
                    input_dim=block_cfg.network_arch[0],
                    output_dim=output_dim,
                    network_arch=block_cfg.network_arch,
                    activation_fn=block_cfg.activation_fn
                ))
        return nn.Sequential(*blocks)


class HyperNet(nn.Module):
    def __init__(
        self,
        cfg: structured_configs.HypernetConfig
    ):
        """Hypernetwork for a Q-network, that is used to approximate
        the preference conditioned state-action values Q(s, a, w)

        Parameters
        ----------
        config : HypernetConfig
            The configuration for the Q-Hypernetwork
        """
        super().__init__()

        self._cfg = configs.as_structured_config(cfg)
        self._embeddeding = Embedding(
            reward_dim=cfg.reward_dim,
            obs_dim=cfg.obs_dim,
            resblock_arch=cfg.resblock_arch
        )

        self._heads = nn.ModuleList([
            Head(target_input_dim=cfg.action_dim,
                 hidden_dim=cfg.head_hidden_dim,
                 target_output_dim=cfg.layer_dims[0]
                 )
        ])

        for init_std, (in_dim, out_dim) in zip(
                cfg.head_init_stds, common.iter_pairwise(cfg.layer_dims)
        ):
            self._heads.append(
                Head(
                    target_input_dim=in_dim, target_output_dim=out_dim,
                    hidden_dim=cfg.head_hidden_dim, init_std=init_std
                )
            )
        if callable(cfg.activation_fn):
            self._activation_fn = cfg.activation_fn
        else:
            self._activation_fn = nets.get_activation_fn(cfg.activation_fn)

    @property
    def config(self) -> structured_configs.HypernetConfig:
        """Returns the configuration used for the hypernet.

        Returns
        -------
        structured_configs.HypernetConfig
            The used config.
        """
        return self._cfg

    def forward(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            prefs: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply a forward pass for the hypernetwork

        Parameters
        ----------
        obs : torch.Tensor
            The observation from the environment.
        action: torch.Tensor
            The action taken by the agent.
        prefs: torch.Tensor:
            The preferences over the objectives. Should have shape
            (reward_dim, )


        Returns
        -------
        torch.Tensor
            The Q-value of the action given the state.
            Has shape (batch_size, reward_dim)
        """
        z = self._embeddeding(obs, prefs)
        weights, biases, scales = zip(*[head(z) for head in self._heads])

        # Do not apply activation on the last layer
        # (i.e. create mask where last item is false)
        apply_activation = np.arange(len(weights)) < len(weights) - 1
        out = nets.target_network(
            action,  weights=weights, biases=biases, scales=scales,
            apply_activation=apply_activation, activation_fn=self._activation_fn
        )
        # Remove the singleton dimension.
        return out.squeeze(2)
