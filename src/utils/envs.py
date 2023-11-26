""" Define some helpers for the environments"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import numpy.typing as npt
import torch


class TorchWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env: gym.Env, device: str | torch.device | None = None):
        """A wrapper that allows one to pass in and receive torch.Tensors to
        the  environments seamlessly.

        Parameters
        ----------
        env : gym.Env
            The enviroment to which the wrapper is applied to.
        device : str | torch.device | None
            The device where the values should be moved after converting
            to tensors. If None, cpu will be used. Default None.
        """
        gym.utils.RecordConstructorArgs.__init__(self, device=device)
        super().__init__(env)
        self.device = device if device is not None else torch.device("cpu")

    def reset(
            self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Overrides the reset method to return the observations as a torch.Tensor

        Parameters
        ----------
        seed : int | None
            The seed for the enviroment. Default None
        options : Dict[str, Any] | None
            Any possible options for the enviroment. Default None.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, Any]]
            The observation (as a tensor in the desired device), and info
            returned by the enviroment.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info

    def step(
            self, action: npt.NDArray | torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, bool, Dict[str, Any]]:
        """Overrides the step method to allow one to use tensors.

        Parameters
        ----------
        action : npt.NDArray |  torch.Tensor
            The action to take. Can be either a numpy ndarray or a tensor

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, bool, bool, Dict[str, Any]]
            Returns the observation after the action, the received rewards,
            the terminated and truncated bools, and the info from the enviroment.
            The observation and rewards are moved to the specified device.
        """
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        obs, rewards, terminated, truncated, info = self.env.step(action)
        obs = torch.from_numpy(obs).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        return obs, rewards, terminated, truncated, info


def create_env(env_id: str, device: str | torch.device) -> gym.Env:
    """Create new gymnasium enviroment and apply the neccessary wrappers to the 
    enviroment.

    Parameters
    ----------
    env_id : str
        The id of the environment.
    device : str | torch.device
        The device to which the data should be moved to from the environment.

    Returns
    -------
    gym.Env
        The desired enviroment with appropriate wrappers.
    """
    env = mo_gym.make(env_id)
    if isinstance(device, str):
        device = torch.device(device)
    env = mo_gym.MORecordEpisodeStatistics(env)
    env = TorchWrapper(env, device=device)
    return env


def create_vec_envs(env_id: str, device: str | torch.device, n_envs: int = 5):
    envs = mo_gym.MOSyncVectorEnv(
        (lambda env_id: mo_gym.make(env_id) for _ in range(n_envs))
    )
    return envs
