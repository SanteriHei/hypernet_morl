"""Define some helpers for the environments"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium.vector import AsyncVectorEnv, VectorEnv
from gymnasium.vector.utils import batch_space


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


class MOAsyncVectorEnv(AsyncVectorEnv):
    def __init__(self, env_fns: Sequence[Callable], copy: bool = True):
        """Asynchronous vectorized environments implementation for the
        multi-objective environments. Adds the reward-space information
        to the environment.

        Parameters
        ----------
        env_fns : Sequence[Callable]
            Set of functions that generate the environments.
        copy : bool, optional.
            If set to true, the reset and step return a copy of the
            observations. Default True.
        """
        dummy_env = env_fns[0]()
        try:
            self.reward_space = dummy_env.get_wrapper_attr("reward_space")
        except Exception:
            self.reward_space = dummy_env.reward_space
        dummy_env.close()
        AsyncVectorEnv.__init__(self, env_fns, copy=copy)
        self._rewards = batch_space(self.reward_space, self.num_envs)


def extract_env_dims(env: gym.Env) -> Dict[str, int]:
    """Extract environment space dimensions, taking into account the
    vectorized environments.

    Parameters
    ----------
    env : gym.Env
        The environment to look at.

    Returns
    -------
    Dict[str, int]
        The observation dimension, action dimension and reward dimension of the
        corresponding SINGLE environment
    """
    is_vec_env = env.get_wrapper_attr("is_vector_env")
    if is_vec_env:
        return {
            "obs_dim": env.get_wrapper_attr("single_observation_space").shape[0],
            "action_dim": env.get_wrapper_attr("single_action_space").shape[0],
            "reward_dim": env.reward_space.shape[0],
            "num_envs": env.num_envs
        }

    return {
        "obs_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.shape[0],
        "reward_dim": env.reward_space.shape[0],
        "num_envs": 1
    }


def create_env(
    env_id: str,
    device: str | torch.device,
    gamma: float = 1.0,
    **kwargs: Dict[str, Any],
) -> gym.Env:
    """Create new gymnasium enviroment and apply the neccessary wrappers to the
    enviroment.

    Parameters
    ----------
    env_id : str
        The id of the environment.
    device : str | torch.device
        The device to which the data should be moved to from the environment.
    gamma: float, optional
        The discount factor used when tracking the episode statistics.
        Default 1.0
    kwargs: Dict[str, Any]
        Any possible extra keyword arguments passed to environment creation
        function

    Returns
    -------
    gym.Env
        The desired enviroment with appropriate wrappers.
    """
    env = mo_gym.make(env_id, **kwargs)
    if isinstance(device, str):
        device = torch.device(device)

    # Add normalizing
    env = mo_gym.MORecordEpisodeStatistics(env, gamma=gamma)
    env = TorchWrapper(env, device=device)
    return env


def create_vec_envs(
    env_id: str,
    device: str | torch.device,
    gamma: float = 1.0,
    n_envs: int = 5,
    asynchronous: bool = True,
    **kwargs: Mapping[str, Any]
) -> VectorEnv:
    """Creates vectorized environments with the required wrappers.

    Parameters
    ----------
    env_id : str
        The id for the used environment.
    device : str | torch.device
        The device at which the outputs from the environment will be placed to.
    gamma: float, optional
        The discount factor used when tracking the episode statistics.
        Default 1.0
    n_envs : int, optional
        The amount environments to create. Default 5
    asynchronous : bool, optional
        If set to True, the vectorized environments will be run in
        parallel, otherwise they are run in synchronized fashion. Default True.
    kwargs: Mapping[str, Any]
        Any additional arguments that will be passed down to the ´gym.make´
        method.

    Returns
    -------
    VectorEnv
        'n_envs' copies of the environment in vectorized form.
    """
    if asynchronous:
        envs = MOAsyncVectorEnv(
                [_create_env(env_id,  **kwargs) for _ in range(n_envs)]
        )
    else:
        envs = mo_gym.MOSyncVectorEnv(
                [_create_env(env_id,  **kwargs) for _ in range(n_envs)]
        )
    envs = mo_gym.MORecordEpisodeStatistics(envs, gamma=gamma)
    envs = TorchWrapper(envs, device=device)
    return envs


def _create_env(env_id: str, **kwargs: Mapping[str, Any]) -> Callable:
    def _thunk():
        env = mo_gym.make(env_id, **kwargs)
        env = gym.wrappers.AutoResetWrapper(env)
        return env

    return _thunk
