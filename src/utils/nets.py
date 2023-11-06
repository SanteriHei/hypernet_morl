""" Some utilities for building neural networks """

from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

from . import common


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


def create_mlp(
        *,
        input_dim: int,
        output_dim: int,
        network_arch: Tuple[int, ...],
        activation_fn: nn.Module | str,
        apply_activation_to_last: bool = False
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
    apply_activation_to_last : bool
        If set to True, the activation function will be also applied after the 
        ouput layer. Default False

    Returns
    -------
    nn.Module
        The MLP network.
    """

    if (n_layers := len(network_arch)) < 1:
        raise ValueError(
            f"Expected atleast 1 layer, but got {n_layers} layers"
        )

    if isinstance(activation_fn, str):
        activation_fn = get_activation_module(activation_fn)

    assert isinstance(activation_fn, nn.Module)

    layers = []
    architecture = (input_dim, *network_arch)

    for in_dim, out_dim in common.iter_pairwise(architecture):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation_fn())

    layers.append(nn.Linear(architecture[-1], output_dim))
    if apply_activation_to_last:
        layers.append(activation_fn())

    return nn.Sequential(*layers)


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
            return torch.nn.ReLu
        case "leaky-relu":
            return torch.nn.LeakyReLu
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
            return optim.Ada
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
