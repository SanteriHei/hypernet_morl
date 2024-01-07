""" Define the hypernetwork model for the MSA-hyper """
from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, Iterable, List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import structured_configs
from ..utils import common, configs, log, nets


class HeadNet(nn.Module):
    def __init__(
            self, *,
            hidden_dim: int,
            target_input_dim: int,
            target_output_dim: int,
            layer_features: Tuple[int, ...],
            n_outputs: int = 1,
            init_method: Literal["uniform", "normal"] = "uniform",
            init_stds: Iterable[float] | float = 0.02
    ):
        """Define a "Head" for a given module, that is responsible for 
        generating all parameters for a neural network 

        Parameters
        ----------
        cfg: structured_configs.HyperNetConfig
            The configuration for the hypernet.
        hidden_dim : int
            The hiddend dimension of the embedding
        target_input_dim : int
            The input dimension of the target network.
        target_output_dim : int
            The ouput dimension of the target network.
        network_arch : Tuple[int, ...]
            The network architecture as the number of neurons in the layers.
        n_outputs : int, optional
            The number of output layers the network has. Default 1.
        init_method: Literal["uniform", "normal"], optional
            The initialization method used for the layers. Default "uniform"
        init_stds : Iterable[float] | float, optional
            The standard deviation(s) used in the initialization. Default 0.05
        """

        # assert init_method in ("uniform", "normal"), \
        #     f"Unknown init method {init_method!r}"

        super().__init__()
        self._logger = log.get_logger("hypernet.headnet")
        self._weight_layers = nn.ModuleList()
        self._bias_layers = nn.ModuleList()
        self._scale_layers = nn.ModuleList()
        self._target_output_dims = []
        self._target_input_dims = []

        self._weight_layers.append(
            nn.Linear(hidden_dim, target_input_dim * layer_features[0])
        )
        self._scale_layers.append(
            nn.Linear(hidden_dim, layer_features[0])
        )
        self._bias_layers.append(
            nn.Linear(hidden_dim, layer_features[0])
        )
        self._target_output_dims.append(layer_features[0])
        self._target_input_dims.append(target_input_dim)

        for in_dim, out_dim in common.iter_pairwise(layer_features):
            self._logger.debug(
                f"Weight: Linear {hidden_dim} -> {out_dim*in_dim}"
            )
            self._logger.debug(
                f"bias, scale: Linear {hidden_dim} -> {out_dim}"
            )
            self._weight_layers.append(
                nn.Linear(hidden_dim, in_dim*out_dim)
            )
            self._bias_layers.append(nn.Linear(hidden_dim, out_dim))
            self._scale_layers.append(nn.Linear(hidden_dim, out_dim))
            self._target_output_dims.append(out_dim)
            self._target_input_dims.append(in_dim)

        # Lastly add the layer(s) that actually ouputs the weights
        for i in range(n_outputs):
            self._logger.debug(
                f"Output {i+1}: Weight: Linear "
                f"{hidden_dim} -> {target_output_dim*layer_features[-1]}"
            )

            self._logger.debug(
                f"Output {i+1}: bias, scale: Linear "
                f"{hidden_dim} -> {target_output_dim}"
            )
            self._weight_layers.append(
                nn.Linear(hidden_dim, layer_features[-1]*target_output_dim)
            )
            self._bias_layers.append(
                nn.Linear(hidden_dim, target_output_dim)
            )
            self._scale_layers.append(
                nn.Linear(hidden_dim, target_output_dim)
            )
            self._target_output_dims.append(target_output_dim)
            self._target_input_dims.append(layer_features[-1])

        self._init_layers(init_method, init_stds)

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """The forward pass of the head. Generates the weights, biases and 
        the scales for the specified target network.

        Parameters
        ----------
        x : torch.Tensor
            The input for the hypernet.

        Returns
        -------
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
            The weights, biases and scales for each layer. The last n_outputs
            layers contain the parameters for the ouput layers.
        """
        iter = zip(self._weight_layers, self._bias_layers, self._scale_layers)
        weights = []
        biases = []
        scales = []
        for i, (weight_l, bias_l, scale_l) in enumerate(iter):
            target_in, target_out = self._target_input_dims[i], self._target_output_dims[i]
            param_w = weight_l(x).view(-1, target_out, target_in)
            param_b = bias_l(x).view(-1, target_out, 1)
            param_s = scale_l(x).view(-1, target_out, 1)
            weights.append(param_w)
            biases.append(param_b)
            scales.append(param_s)
        return weights, biases, scales

    @torch.no_grad
    def _init_layers(
            self, init_method: Literal["uniform", "normal"],
            init_stds: Tuple[float, ...] | float
    ):
        """
        Initialize the layers of a hyper network.

        Parameters
        ----------
        init_stds : Iterable[float] | float
            The initial standard deviations
        """
        if isinstance(init_stds, float):
            tmp = (
                init_stds for _ in range(len(self._bias_layers))
            )
            init_stds = tmp

        iter = zip(
            init_stds, self._weight_layers, self._bias_layers, self._scale_layers
        )

        self._logger.debug(f"Initializing using {init_method!r}")
        for std, weight_l, bias_l, scale_l in iter:
            if init_method == "uniform":
                nn.init.uniform_(weight_l.weight, -std, std)
                nn.init.uniform_(bias_l.weight, -std, std)
                nn.init.uniform_(scale_l.weight, -std, std)
            elif init_method == "normal":
                self._logger.debug(f"normal with std={std}/den")
                den = np.ceil(np.sqrt(np.prod(weight_l.weight.shape)))
                nn.init.normal_(weight_l.weight, mean=0.0, std=std/den)

                den = np.ceil(np.sqrt(np.prod(bias_l.weight.shape)))
                nn.init.normal_(bias_l.weight, mean=0.0, std=std/den)

                den = np.ceil(np.sqrt(np.prod(scale_l.weight.shape)))
                nn.init.normal_(scale_l.weight, mean=0.0, std=std/den)

            # Initialize biases to zeros
            nn.init.zeros_(weight_l.bias)
            nn.init.zeros_(bias_l.bias)
            nn.init.zeros_(scale_l.bias)


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
        warnings.warn(
            "'Head' is deprecated! Use 'HeadNet' instead!",
            category=DeprecationWarning
        )

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
            layer_features: Tuple[int, ...],
            activation_fn: str | Callable = "relu",
            dropout_rates: Iterable[float] | float | None = None
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

        assert (n_layers := len(layer_features)) >= 1, \
            f"Expected atleast 2 layer resblock, got {n_layers} layers instead"

        # Do not apply activation function to the last layer.
        apply_activation = tuple(
            i != len(layer_features) for i in range(len(layer_features) + 1)
        )

        self._network = nets.create_mlp(
            input_dim=input_dim,
            layer_features=layer_features,
            activation_fn=activation_fn,
            apply_activation=apply_activation,
            dropout_rate=dropout_rates
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
        embedding_layers: Tuple[structured_configs.ResblockConfig, ...]
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

        self._hypernet = self._construct_network(embedding_layers)
        self._init_layers()

    def forward(
            self, meta_variable: torch.Tensor
    ) -> torch.Tensor:
        """Generate the hidden presentation of the state weights 
        that is used to generated the weights for a linear layer.

        Parameters
        ----------
        meta_variable: torch.Tensor
            The meta-variable used to generate the weights for the target 
            network.

        Returns
        -------
        torch.Tensor
           The hidden presentation of the input state. Has shape
           (batch_size, ) (which was specified in the constructor)
        """

        # Condition the network on the preferences
        x = self._hypernet(meta_variable)
        return x

    @torch.no_grad()
    def _init_layers(self):
        """
        Initialize the layers using fan-in Kaiming uniform initialization.
        """
        for module in self._hypernet.modules():
            if isinstance(module, nn.Linear):
                # Bit hacky to use a private function for this!
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
                    module.weight)
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
            assert len(block_cfg.layer_features) > 0, \
                "Resblock needs to have atleast one layer"
            blocks.append(
                nn.Linear(
                    block_cfg.input_dim,
                    block_cfg.layer_features[0]
                )
            )
            for ii in range(block_cfg.n_resblocks):
                blocks.append(ResBlock(
                    input_dim=block_cfg.layer_features[0],
                    layer_features=block_cfg.layer_features,
                    activation_fn=block_cfg.activation_fn,
                    dropout_rates=block_cfg.dropout_rates
                ))
        return nn.Sequential(*blocks)


class HyperNet(nn.Module):
    def __init__(
        self,
        cfg: structured_configs.CriticConfig
    ):
        """Hypernetwork for a Q-network, that is used to approximate
        the preference conditioned state-action values Q(s, a, w)

        Parameters
        ----------
        config : HypernetConfig
            The configuration for the Q-Hypernetwork
        """
        super().__init__()
        warnings.warn(
            "HyperNet is deprecated! Use critics.HyperCritic instead",
            category=DeprecationWarning
        )

        self._cfg = configs.as_structured_config(cfg)
        self._embeddeding = Embedding(
            reward_dim=cfg.reward_dim,
            obs_dim=cfg.obs_dim,
            embedding_layers=cfg.resblock_arch
        )

        target_input_dim = self._get_target_input_dim()

        self._heads = nn.ModuleList([
            Head(target_input_dim=target_input_dim,
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
    def config(self) -> structured_configs.CriticConfig:
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
        print("Hypernet generated params:")
        for w, b, s in zip(weights, biases, scales):
            print(f"W: {w.shape} | B {b.shape} | S {s.shape}")

        # Do not apply activation on the last layer
        # (i.e. create mask where last item is false)
        apply_activation = np.arange(len(weights)) < len(weights) - 1

        target_net_input = self._get_target_input(obs, action, prefs)

        out = nets.target_network(
            target_net_input, weights=weights, biases=biases, scales=scales,
            apply_activation=apply_activation, activation_fn=self._activation_fn
        )
        # Remove the singleton dimension.
        return out.squeeze(2)

    def _get_target_input_dim(self) -> int:
        """Get the input dimension for the target network, depending the 
        input configuration the user defined.

        Returns
        ------- 0% 
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
            out = action

        assert out is not None, \
            (f"Unknown critic input config: obs: {self._cofg.use_obs} | "
             f"Action {self._cfg.use_action} |  Prefs {self._cfg.use_prefs}")
        return out
