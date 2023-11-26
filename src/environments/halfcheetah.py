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
        reward_run = min(self._base_reward,
                         info["x_velocity"]) + self._alive_bonus
        vec_reward = np.array([reward_run, energy_reward])
        info["reward_run"] = reward_run
        return observation, vec_reward, terminated, truncated, info


class MoHopper(gym.utils.EzPickle):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        gym.utils.EzPickle.__init__(self, **kwargs)
        self.reward_dim = 2
        self.reward_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.reward_dim, )
        )

        self._alive_bonus = 1.0

    def step(self, action):
        pos_before = self.sim.data.qpos[0]
        action = np.clip(action, [-2.0, -2.0, -4.0], [2.0, 2.0, 4.0])
        self.do_simulation(action, self.frame_skip)
        pos_after, height, angle = self.sim.data.qpos[0:3]
        x_velocity = (pos_after - pos_before)/self.dt

        other_rewards = self._alive_bonus - 2e-4 * np.square(action).sum()
        reward_run = 1.5 * x_velocity + other_rewards
        jump_height = height - self.init_qpos[1]
        reward_jump = 12.0 * jump_height + other_rewards
        s = self.state_vector()
        terminated = not((s[1] > 0.4) and abs(s[2]) < np.deg2rad(90) and abs(s[3]) < np.deg2rad(90) and abs(s[4]) < np.deg2rad(90) and abs(s[5]) < np.deg2rad(90))

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
        pass
