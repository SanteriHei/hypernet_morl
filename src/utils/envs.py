""" Define some helpers for the environments"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
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
            self, action: npt.NDArray |  torch.Tensor
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
        obs = torch.from_numpy(obs).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        return obs, rewards, terminated, truncated, info
