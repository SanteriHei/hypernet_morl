""" Some utilities for building neural networks """

import itertools
from typing import Iterable, Callable, Tuple

import torch
from torch import nn
import torch.nn.functional as F

def iter_pairwise(x: Iterable) -> Iterable:
    """Iterate over pairs (s0, s1), (s1, s2), ... (sN-1, sN)

    Parameters
    ----------
    x : Iterable
        The object that should be iterated pairwise

    Returns
    -------
    Iterable
        The iterator that returns the overlapping, sequential pairs of
        items.

    """
    a, b = itertools.tee(x)
    next(b, None)
    return zip(a, b)

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
        activation_fn = get_activation_module(get_activation_module)
    
    assert isinstance(activation_fn, nn.Module)
        
    layers = []
    architecture = (input_dim, *network_arch)

    for in_dim, out_dim in iter_pairwise(architecture):
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
