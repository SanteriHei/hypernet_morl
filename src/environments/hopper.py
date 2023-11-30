from typing import Any, Mapping, Tuple

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.envs.mujoco.hopper_v4 import HopperEnv


class MoHopper(HopperEnv, gym.utils.EzPickle):
    """Backport of Multi-objective hopper environment to the Gymnasium. 
    Utilizes the reward definitions from (Insert pgmorl). 

    The reward space is two dimensional, where the first objective is the
    forward speed, and the second objective is the jumping height.

    Attributes
    ----------
    reward_dim : The reward space dimension. 2 in this case.
    reward_space : The reward space.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        gym.utils.EzPickle.__init__(self, **kwargs)
        self.reward_dim = 2
        self.reward_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.reward_dim, )
        )

        self._alive_bonus = 1.0

    def step(
            self, action: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray, bool, bool, Mapping[str, Any]]:

        """Simulate a step in the environment.

        Parameters
        ----------
        action : npt.NDArray
            The action to take.

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray, bool, bool, Mapping[str, Any]]
            The observation, the reward, terminated, truncated and info,
            that contain the x-position, x-velocity and the pure height
            reward (i.e. the jump_height without energy penalty) and the 
            pure enrgy reward (i.e. sum(action^2))
        """
        pos_before = self.data.qpos[0]
        action = np.clip(action, [-2.0, -2.0, -4.0], [2.0, 2.0, 4.0])
        self.do_simulation(action, self.frame_skip)
        pos_after, height, angle = self.data.qpos[0:3]
        x_velocity = (pos_after - pos_before)/self.dt

        other_rewards = self._alive_bonus - 2e-4 * np.square(action).sum()
        reward_run = 1.5 * x_velocity + other_rewards
        jump_height = height - self.init_qpos[1]
        reward_jump = 12.0 * jump_height + other_rewards
        s = self.state_vector()
        terminated = not ((s[1] > 0.4) and abs(s[2]) < np.deg2rad(90) and abs(
            s[3]) < np.deg2rad(90) and abs(s[4]) < np.deg2rad(90) and abs(s[5]) < np.deg2rad(90))

        info = {
            "x_position": pos_after,
            "x_velocity": x_velocity,
            "height_reward":  reward_jump - other_rewards,
            "energy_reward": np.square(action).sum()
        }

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, np.array([reward_run, reward_jump]), terminated, False, info
