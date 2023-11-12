""" Define the hypernetwork model for the MSA-hyper """
from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn

from .. import structured_configs
from ..utils import common, configs, nets


def _apply_hyper_init(
        modules: Iterable["Head"],
        weight_shape: Tuple[int, ...],
        bias_shape: Tuple[int, ...],
        method: str = "bias",
        init_type: str = "xavier_uniform",
):
    """Apply hyper-init initialization to the heads of the hypernetwork. 
    Hypernetworks in Meta-Reinforcement learning, Beck et al, CORL 2022
    http://arxiv.org/abs/2210.11348 for more details.

    Parameters
    ----------
    modules : Iterable["Head"]
        The modules to initialize.
    weight_shape : Tuple[int, ...]
        The shapes of the weights.
    bias_shape : Tuple[int, ...]
        The shape of bias.
    method : str
        The method of the initializing. Can be either "bias" or "weights"
    init_type : str
        The used initialization function.
    """
    match init_type:
        case "xavier_uniform":
            init_fn = nn.init.xavier_uniform_
        case "xavier_normal":
            init_fn = nn.init.xavier_normal_
        case "orthogonal":
            init_fn = nn.init.orthogonal_
        case "kaiming_uniform":
            init_fn = nn.init.kaiming_uniform_
        case "kaiming_normal":
            init_fn = nn.init.kaiming_normal_
        case _:
            print(f"Unknown choice {init_type!r}, using ones instead")
            init_fn = nn.init.ones_

    # Generate the common initialization weights for the modules
    bias_weights = torch.zeros(bias_shape)
    weights = torch.zeros(weight_shape)
    if method == "bias":
        init_fn(bias_weights)
    else:
        init_fn(weights)

    for head in modules:
        head.init_weights(weights, bias_weights)


class Head(nn.Module):
    def __init__(
            self, *,
            target_input_dim: int,
            target_output_dim: int,
            hidden_dim: int,
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

        self.apply(nets.init_layers)

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

    def init_weights(self, weights: torch.Tensor, biases: torch.Tensor):
        """Initialize the weights for the network from the given weights.

        Parameters
        ----------
        weights : torch.Tensor
            The weights for the linear layer.
        bias_weights: torch.Tensor:
            The weights for the bias of the layers.
        """
        with torch.no_grad():
            self._weight_layer.weight.data.copy_(weights["weight"])
            self._weight_layer.bias.data.copy_(biases["weight"])

            self._bias_layer.weight.data.copy_(weights["bias"])
            self._bias_layer.bias.data.copy_(biases["bias"])

            self._scale_layer.weight.data.copy_(weights["scale"])
            self._scale_layer.bias.data.copy_(biases["scale"])

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

        n_layers = len(network_arch)
        assert n_layers >= 2, \
            f"Expected atleast 2 layer resblock, got {n_layers} layers instead"

        self._network = nets.create_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            network_arch=network_arch,
            activation_fn=activation_fn,
            apply_activation_to_last=False
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
        self.apply(nets.init_layers)

    def forward(
            self, obs: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Generate the hiddegymnasiumn presentation of the state weights 
        that is used to generated the weights for a linear layer.

        Parameters
        ----------Hypernetworks in Meta-Reinforcement Learning 
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

        for in_dim, out_dim in common.iter_pairwise(cfg.layer_dims):
            self._heads.append(
                Head(
                    target_input_dim=in_dim, target_output_dim=out_dim,
                    hidden_dim=cfg.head_hidden_dim
                )
            )

        self._init_heads()

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
        out = nets.apply_dynamic_pass(
            action,  weights=weights, biases=biases, scales=scales,
            apply_activation=apply_activation, activation_fn=self._activation_fn
        )
        # Remove the singleton dimension.
        return out.squeeze(2)

    @torch.no_grad()
    def _init_heads(self):
        """Initialize the heads bias only init"""

        # TODO: Figure out the hyper-init initialization
        assert len(self._heads) > 0, "No heads to initialize!"

        bias_shapes = self._heads[0].get_bias_shapes()
        weight_shapes = self._heads[0].get_weight_shapes()

        for head in self._heads:
            bias_shapes = head.get_bias_shapes()
            weight_shapes = head.get_weight_shapes()

            bias_inits = {}
            for layer, shape in bias_shapes.items():
                std = 1/math.sqrt(shape[0])
                init_weights = torch.zeros(shape)
                init_weights.uniform_(-std, std)
                bias_inits[layer] = init_weights
            weight_inits = {
                layer: torch.zeros(shape)
                for layer, shape in weight_shapes.items()
            }

            head.init_weights(weight_inits, bias_inits)
