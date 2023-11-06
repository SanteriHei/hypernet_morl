""" Define the hypernetwork model for the MSA-hyper """
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import torch
from torch import nn

from .utils import common, nets


@dataclass
class ResblockConfig:
    """
    A configuration for a Residual block

    Attributes
    ----------
    n_resblocks: int The amount of residual blocks to add in a single block
    input_dim: int The input dimension of the network.
    output_dim: int The ouput dimension of the network.
    activation_fn: Callable | str The activation function to use. Default "relu"
    network_arch: Tuple[int, ...] The network architecture. Default (128, 128)
    (i.e. two linear layers with 128 neurons)

    """
    n_resblocks: int
    input_dim: int
    output_dim: int
    activation_fn: Callable | str = "relu"
    network_arch: Tuple[int, ...] = (128, 128)


@dataclass
class HypernetConfig:
    """
    A configuration for the Q-Hypernetwork

    Attributes
    ----------
    resblock_arch: Tuple[ResblockConfig, ...] The configuration for the residual
        blocks
    layer_dims: Tuple[int, ...] The dimensions for the dynamic network.
    reward_dim: int The reward dimension of the environment. Default 3.
    obs_dim: int The observation dimension of the  environment. Default 3 
    head_hidden_dim: int The hidden dimension of the "Heads". Default 1024.
    activation_fn: Callable | str: The activation function used in the dynamic
        network. Default "relu"
    """
    resblock_arch: Tuple[ResblockConfig, ...]
    layer_dims: Tuple[int, ...]
    reward_dim: int = 3
    obs_dim: int = 3
    head_hidden_dim: int = 1024
    activation_fn: Callable | str = "relu"


def _apply_hyper_init(
        modules: Iterable["Head"],
        weight_shape: Tuple[int, ...],
        bias_shape: Tuple[int, ...],
        method: str = "bias",
        init_type: str = "xavier_uniform",
):

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
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
    ):
        """
        The "Head" that is used to generate the weights for a single linear
        layer

        Parameters
        ----------
        input_dim : int
            The input dimension of the target network
        output_dim : int
            The output dimension of the target network
        hidden_dim: int
            The size of the hidden dimension used in the layers.
        """
        super().__init__()

        self._target_input_dim = input_dim
        self._target_output_dim = output_dim

        self._weight_layer = nn.Linear(hidden_dim, input_dim * output_dim)
        self._bias_layer = nn.Linear(input_dim, output_dim)
        self._scale_layer = nn.Linear(input_dim, output_dim)

    def init_weights(self, weights: torch.Tensor, bias_weights: torch.Tensor):
        """Initialize the weights for the network from the given weights.

        Parameters
        ----------
        weights : torch.Tensor
            The weights for the linear layer.
        bias_weights: torch.Tensor:
            The weights for the bias of the layers.
        """
        with torch.no_grad():
            self._weight_layer.weight.data.copy_(weights)
            self._weight_layer.bias.data.copy_(bias_weights)

            self._bias_layer.weight.data.copy_(weights)
            self._bias_layer.bias.data.copy_(bias_weights)

            self._scale_layer.weight.data.copy_(weights)
            self._weight_layer.bias.data.copy_(bias_weights)

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
        bias = self._bias_layer(x).view(-1, self._target_output_dim)
        scale = self._scale_layer(x).view(-1, self._target_output_dim)
        return weights, bias, scale

    def _initialize_weights(self):
        # Implementation of Bias-HyperInit
        torch.init.zeros_(self._weight_layer.weight)

        pass


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
            apply_activation_to_last=False
        )

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
        weight_dim: int,
        obs_dim: int,
        z_dim: int,
        resblock_arch: Tuple[ResblockConfig, ...],
    ):
        """Generate an "embedding" layer that transfers the current observation
        and preferences over the objectives into latent variable.

        Parameters
        ----------
        weight_dim : int
           The reward dimension 
        obs_dim : int
            The observation dimesion
        z_dim : int
            The latent dimension
        resblock_arch : Tuple[ResblockConfig, ...]
            The configuration for each linear + res + res "block"
        """
        super().__init__()

        self._z_dim = z_dim
        blocks = []

        res_output_dim = None
        blocks.append(
            nn.Linear(weight_dim + obs_dim, resblock_arch[0].input_dim)
        )
        for i, block_config in enumerate(resblock_arch):
            if res_output_dim is not None:
                blocks.append(
                    nn.Linear(res_output_dim, block_config.input_dim)
                )

            res_output_dim = block_config.output_dim

            for _ in range(block_config.n_resblocks):
                blocks.append(ResBlock(
                    input_dim=block_config.input_dim,
                    ouput_dim=block_config.ouput_dim,
                    network_architecture=block_config.net_architecture,
                    activation_fn=block_config.activation_fn
                ))
        self._hypernet = nn.Sequential(*blocks)

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
           The hidden presentation of the input state. Has shape (z_dim, )
           (which was specified in the constructor)
        """

        # Condition the network on the preferences
        network_input = torch.cat(obs, weights, dim=-1)
        return self._hypernet(network_input)


class HyperNet(nn.Module):
    def __init__(
        self,
        config: HypernetConfig
    ):
        """Hypernetwork for a Q-network, that is used to approximate
        the preference conditioned state-action values Q(s, a, w)

        Parameters
        ----------
        config : HypernetConfig
            The configuration for the Q-Hypernetwork
        """
        super().__init__()

        self._config = config
        self._embeddeding = Embedding(
            reward_dim=config.reward_dim,
            obs_dim=config.obs_dim,
            resblock_arch=config.resblock_arch
        )

        self._heads = [
            Head(input_dim=config.resblock_arch[-1].output_dim,
                 hidden_dim=config.head_hidden_dim,
                 output_dim=config.layer_dims[0]
                 )
        ]

        for in_dim, out_dim in common.iter_pairwise(config.layer_dims[1:]):
            self._heads.append(
                Head(
                    input_dim=in_dim, output_dim=out_dim,
                    hidden_dim=config.head_hidden_dim
                )
            )

        _apply_hyper_init(self._heads)

        if callable(config.activation_fn):
            self._activation_fn = config.activation_fn
        else:
            self._activation_fn = nets.get_activation_fn(config.activation_fn)

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
        """

        z = self._embeddeding(obs, prefs)

        # Apply the dynamic pass for each of the generated layers
        out = action
        for i, head in enumerate(self._heads):
            weights, bias, scale = head(z)
            out = torch.bmm(weights, out) * scale + bias
            # Don't apply activation to the last layer
            if i != self._config.n_heads - 1:
                out = self._activation_fn(out)
        return out.view(-1, 1)
