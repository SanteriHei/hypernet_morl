from typing import Any, Mapping, Tuple

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv


class MoHalfCheetah(HalfCheetahEnv, gym.utils.EzPickle):

    """Port of the Multi-objective half-cheetah environment from 
    PGMORL. Importantly, changes the second objective to be the energy
    efficiency, rather than the control cost as in the mo-gymnasium

    Attributes
    ----------
    reward_dim : The reward dimension. 2 in this case.
    reward_space : The reward space, which will be from - infinity to infinity
        in this case.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        gym.utils.EzPickle.__init__(self, **kwargs)
        self.reward_dim = 2
        self.reward_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.reward_dim, )
        )
        self._base_reward = 4.0
        self._alive_bonus = 1.0
        pass

    def step(self, action: npt.NDArray) -> Tuple[
            npt.NDArray, npt.NDArray, bool, bool, Mapping[str, Any]
    ]:
        """Take a step in the environment and observe the outcome. Utilizes
        the energy reward and the reward for the speed.

        Parameters
        ----------
        action : npt.NDArray
            The action to take.

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray, bool, bool, Mapping[str, Any]]
              Returns the observation, (vector) reward, terminated, truncated
              and info.
              In this case the first reward component is the speed reward, 
              while the second component is the energy reward.
        """
        observation, reward, terminated, truncated, info = super().step(action)
        energy_reward = (
            self._base_reward -
            1.0 * np.square(action).sum() + self._alive_bonus
        )
        reward_run = min(self._base_reward, info["x_velocity"]) + self._alive_bonus
        vec_reward = np.array([reward_run, energy_reward])
        info["reward_run"] = reward_run
        return observation, vec_reward, terminated, truncated, info