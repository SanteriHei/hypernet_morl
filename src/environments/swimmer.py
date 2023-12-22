from typing import Any, Mapping, Tuple

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv


class MoSwimmer(SwimmerEnv, gym.utils.EzPickle):
    

    """Port of the Multi-objective Swimmer environment from PGMORL paper.
    The environment has two objectives: The forward speed and the energy 
    efficiency. The optimal value for the energy efficiency is set at 0.3

    Attributes
    ----------
    reward_dim : The dimension of the reward space. 2 in this case.
    reward_space : The Reward space of the environment.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        gym.utils.EzPickle.__init__(self, **kwargs)
        self.reward_dim = 2


        self._ctrl_cost_coeff = 0.15
        self._base_rwd = 0.3

        self.reward_space = gym.spaces.Box(
                low=np.array([-np.inf, -np.inf]),
                high=np.array([np.inf, self._base_rwd]),
                dtype=np.float32
        )


    def step(self, action: npt.NDArray) -> Tuple[
            npt.NDArray, npt.NDArray, bool, bool, Mapping[str, Any]
    ]:
        """Takes a step in the environment and observe the outcome.

        Parameters
        ----------
        action : npt.NDArray
            The action to take.

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray, bool, bool, Mapping[str, Any]]
            Returns the observation, (vector) reward, terminated, truncated
            and info. The infor contains the current, the energy efficiency 
            reward and the current x position
        """
        x_pos_before = self.data.qpos[0]
        action = np.clip(action, a_min=-1.0, a_max=1.0)
        self.do_simulation(action, self.frame_skip)
        x_pos_after = self.data.qpos[0]
        forward_speed = (x_pos_after - x_pos_before) / self.dt
        reward_ctrl = (
                self._base_rwd - self._ctrl_cost_coeff * np.square(action).sum()
        )
        obs = self._get_obs()

        info = {
                "reward_speed": forward_speed,
                "reward_ctrl": reward_ctrl,
                "x_position": x_pos_after
        }

        if self.render_mode == "human":
            self.render()

        return obs, np.array([forward_speed, reward_ctrl]), False, False, info


    def _get_obs(self) -> npt.NDArray:
        """Get the observation. Removes the current x and y positions from the 
        observation.

        Returns
        -------
        npt.NDArray
            The current observation.
        """
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])






