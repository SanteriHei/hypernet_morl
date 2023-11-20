""" Some utilities for building neural networks """

import warnings
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim

from . import common, log

_LOGGER = log.get_logger("utils.nets")


@torch.no_grad()
def polyak_update(src: nn.Module, dst: nn.Module, tau: float = 0.995):
    """Polyak update that updates the parameters of the destination 
    with the source using soft-updates.

    Parameters
    ----------
    src : nn.Module
        The source from which the parameters are copied from.
    dst : nn.Module
        The destination, to which he parameters are copied to.
    tau : float
        The "speed" of the update. Should be in range (0, 1.0]. Default 0.995.
    """
    assert 0 < tau <= 1, f"Tau must be between 0 and 1, got {tau:.3f} instead"
    for src_param, dst_param in zip(src.parameters(), dst.parameters()):
        dst_param.data.copy_(
            dst_param.data * (1.0 - tau) + src_param.data * tau)


@torch.no_grad()
def init_layers(
        layer: torch.nn.Module,
        init_type: str = "xavier_uniform",
        weight_gain: float = 1.0,
        bias_const: float = 0.0
):
    """Initialize layers of a torch.nn module.

    Parameters
    ----------
    layer : torch.nn.Module
        The layer to initialize
    init_type : str
        The type of initialization. Default "xavier_uniform"
    weight_gain : float
        The gain of the weights.
    bias_const : float
        The constant used for initializing the bias.
    """
    init_fn = get_initialization_fn(init_type)
    if isinstance(layer, nn.Linear):
        init_fn(layer.weight, gain=weight_gain)
        nn.init.constant_(layer.bias, bias_const)


@torch.no_grad
def hyper_init(
        modules: Iterable["Head"],
        weight_shape: Tuple[int, ...],
        bias_shape: Tuple[int, ...],
        method: str = "bias",
        init_type: str = "xavier_uniform"
):

    # TODO: How this is actually applied? How would one apply different
    # initializations for each preferences in any way?
    if method not in ("bias", "weight"):
        raise ValueError((f"Unknown initialization method {method!r} for "
                          "hyper-init! Should be either 'bias' or 'weight'"))
    init_fn = get_initialization_fn(init_type)

    # Generate the common weights for each layer
    bias = torch.zeros(bias_shape)
    weights = torch.zeros(weight_shape)

    if method == "bias":
        init_fn(bias)
    else:
        init_fn(weights)

    for head in modules:
        head.init_weights(weights, bias)


def create_mlp(
        *,
        input_dim: int,
        output_dim: int,
        network_arch: Tuple[int, ...],
        activation_fn: nn.Module | str,
        apply_activation: Tuple[bool, ...] | bool = True
) -> nn.Module:
    """Create a simple fully connected network using the specified architecture
    and activation function

    Parameters
    ----------
    input_dim : int
        The input dimension of the network.
    output_dim : int
        The ouput dimension of the network.
    network_arch : Tuple[int, ...]
        The architecture of the network as a tuple of neurons for each layer.
    activation_fn : nn.Module | str
        The used activation function.
    apply_activation Tuple[bool, ...] | bool, optional
        A boolean indicating if activation function should be applied after 
        each given layer. If a single value is given, it is used after each
        layer. Default True. (i.e apply activation function after each layer)

    Returns
    -------
    nn.Module
        The MLP network.
    """

    if (n_layers := len(network_arch)) < 1:
        raise ValueError(
            f"Expected atleast 1 layer, but got {n_layers} layers"
        )

    if isinstance(apply_activation, bool):
        apply_activation = (
            apply_activation for _ in range(len(network_arch) + 1)
        )

    if isinstance(activation_fn, str):
        activation_fn = get_activation_module(activation_fn)

    layers = []
    architecture = (input_dim, *network_arch, output_dim)

    for use_activation, (in_dim, out_dim) in zip(
            apply_activation, common.iter_pairwise(architecture)
    ):
        _LOGGER.debug(
                f"Linear {in_dim} -> {out_dim} with activation? {use_activation}"
        )
        layers.append(nn.Linear(in_dim, out_dim))
        if use_activation:
            layers.append(activation_fn())

    return nn.Sequential(*layers)


def target_network(
        x: torch.Tensor, *,
        weights: Tuple[torch.Tensor, ...], biases: Tuple[torch.Tensor, ...],
        scales: Tuple[torch.Tensor, ...], apply_activation: Tuple[bool],
        activation_fn: str | Callable = "relu",
) -> torch.Tensor:
    """Apply the forward pass of the target network of the hypernetwork.
    The target network is expected to contain linear layers.

    Parameters
    ----------
    x : torch.Tensor
        The input to the network.
    weights : Tuple[torch.Tensor, ...]
        The weights of each layer.
    biases : Tuple[torch.Tensor, ...]
        The biases of each layer.
    scales : Tuple[torch.Tensor, ...]
        The scales for each layer.
    apply_activation : Tuple[bool]
        mask indicating if the activation function is called after a layer.
    activation_fn : str | Callable
        The activation function to be used after each layer.

    Returns
    -------
    torch.Tensor:
        The output of the target networks forward pass.
    """
    out = x
    iter = zip(weights, biases, scales, apply_activation)
    if isinstance(activation_fn, str):
        activation_fn = get_activation_fn(activation_fn)

    assert callable(activation_fn), \
        f"Activation function is not callable! ({activation_fn})"

    for w, b, scale, use_activation in iter:
        in_var = out.unsqueeze(2) if out.ndim == 2 else out
        out = torch.bmm(w, in_var) * scale + b
        if use_activation:
            out = activation_fn(out)
    return out


def get_initialization_fn(init_name: str) -> Callable:
    """Get an initialization function by name.

    Parameters
    ----------
    init_name : str
        The name of the initialization function.

    Returns
    -------
    Callable
        The corresponding initialization function
    """
    match init_name:
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
            warnings.warn((f"Unknown initialization {init_name!r}, "
                           "initializing to ones instead"))
            init_fn = nn.init.ones_
    return init_fn


def get_activation_fn(fn_name: str) -> Callable:
    """
    Get activation function based on the function name.

    Parameters
    ----------
    fn_name : str
        The name of the activation function

    Returns
    -------
    Callable
        The desired activation function as a function.
    """
    match fn_name:
        case "relu":
            return F.relu
        case "leaky-relu":
            return F.leaky_relu
        case "tanh":
            return F.tanh
        case "sigmoid":
            return F.sigmoid
        case _:
            raise ValueError(f"{fn_name!r} is not supported!")


def get_activation_module(fn_name: str) -> torch.nn.Module:
    """
    Get activation function based on a name

    Parameters
    ----------
    fn_name : str
        The name of the activation function

    Returns
    -------
    torch.nn.Module
        The Module version of the function (i.e. not the torch.Functional)
    """

    match fn_name:
        case "relu":
            return torch.nn.ReLU
        case "leaky-relu":
            return torch.nn.LeakyReLU
        case "tanh":
            return torch.nn.Tanh
        case "sigmoid":
            return torch.nn.Sigmoid
        case _:
            raise ValueError(f"{fn_name!r} is not supported!")


def get_optim_by_name(optim_name: str) -> optim.Optimizer:
    """Get a torch optimizer by its name.

    Parameters
    ----------
    optim_name : str
        The name of the optimizer (in lower-case)

    Returns
    -------
    optim.Optimizer
        The optimizer (un-initialized) that corresponds to the given name
    """
    match optim_name:
        case "adam":
            return optim.Adam
        case "rms_prop":
            return optim.RMSprop
        case "adamax":
            return optim.Adamax
        case "sgd":
            return optim.SGD
        case "lbfgs":
            return optim.LBFGS
        case _:
            raise ValueError(f"Unknown optimizer: {optim_name!r}")
