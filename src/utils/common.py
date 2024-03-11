""" Some common utilities for the algorithms"""
from __future__ import annotations

import itertools
import json
import numbers
import pathlib
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import numpy.typing as npt
import pymoo.util.ref_dirs
import torch
from ruamel.yaml import YAML

from . import pareto


class NumpyEncoder(json.JSONEncoder):
    """
    A custom encoder that converts numpy values into native Python types before
    they are serialized.
    """

    def default(self, obj):
        # Only catch the numeric scalar values
        if (
            isinstance(obj, numbers.Number)
            and np.ndim(obj) == 0
            and isinstance(obj, np.generic)
        ):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            # Let the basemethod handle the rest
            return json.JSONEncoder.default(self, obj)

def dump_json(filepath: pathlib.Path | str, payload: Mapping[str, Any]):
    """Writes the given mapping as a json file. Handles numpy arrays as well.

    Parameters
    ----------
    filepath : pathlib.Path | str
        The destination for the data.
    payload : Mapping[str, Any]
        The data to write.
    """
    fpath = pathlib.Path(filepath)
    if fpath.exists() and fpath.is_dir():
        raise FileExistsError(f"{filepath!r} already exists! (and is not a file)")
    with fpath.open("w") as ofstream:
        json.dump(payload, ofstream, cls=NumpyEncoder)


def dump_yaml(filepath: pathlib.Path | str, payload: Mapping[str, Any]):
    """
    Writes a YAML complient data mapping to a file. YAML 1.2(?) is supported.

    Parameters
    ----------
    filepath : pathlib.Path | str
        The filepath to the destination,
    payload : Mapping[str, Any]
        The data to write.
    """
    yaml = YAML()
    fpath = pathlib.Path(filepath)
    if fpath.exists() and fpath.is_dir():
        raise FileExistsError(f"{filepath!r} already exists! (and is not a file)")
    with fpath.open("w") as ofstream:
        yaml.dump(payload, ofstream)


@contextmanager
def scoped_timer(scope_name: str):
    """Create a timer that times the execution of the code inside the
    context.

    Parameters
    ----------
    scope_name : str
        The name of the scope.
    """
    t0 = time.perf_counter_ns()
    try:
        yield
    finally:
        t1 = time.perf_counter_ns()
        elapsed_time = t1 - t0
        print(f"[{scope_name}]: {elapsed_time/10**9:.3f}s")


def set_thread_count(device: str | torch.device, n_threads: int):
    """
    Set the used thread count for Pytorch. NOTE: has effect only if the 
    device is CPU.

    Parameters
    ----------
    device : str | torch.device
        The device used for the computations.
    n_threads : int
        The number of threads used for the computations.
    """
    if (isinstance(device, torch.device) and device.type == "cpu") or device == "cpu":
        torch.set_num_threads(n_threads)


def set_global_rng_seed(seed: int):
    """Fix the seed for Pytorch's and Numpy's global random generators.

    Parameters
    ----------
    seed : int
        The seed to use.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def deg_to_rad(angle: float) -> float:
    """Convert angle from degrees to radians.

    Parameters
    ----------
    angle : float
        The angle in degrees.

    Returns
    -------
    float
        The corresponding angle in radians.
    """
    return (angle / 180) * np.pi


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


def get_equally_spaced_weights(
    dim: int, n_points: int, seed: int | None = None
) -> npt.NDArray:
    """Generate equally spaced points in 'dim' dimensional space.
    See https://pymoo.org/misc/reference_directions.html for more details

    Parameters
    ----------
    dim : int
        The dimensionality of the space.
    n_points : int
        The amount of points to generate.
    seed : int | None
        The seed for the random number generation. Default None.

    Returns
    -------
    npt.NDArray
        The equally spaced points. Has shape (n_points, dim)
    """
    return pymoo.util.ref_dirs.get_reference_directions(
        name="energy", n_dim=dim, n_points=n_points, seed=seed
    )


@dataclass
class ReplaySample:
    """A sample from the replay buffer.

    Attributes
    ----------
    obs : npt.NDArray | torch.Tensor A set of observations. Has shape
        (n_samples, obs_dim)
    actions : npt.NDArray | torch.Tensor A set of actions. Has shape
        (n_samples, action_dim)
    rewards : npt.NDArray | torch.Tensor A set of rewards. Has shape
        (n_samples, reward_dim)
    prefs : npt.NDArray | torch.Tensor A set of preferences over the objectives.
        Has shape (n_samples, reward_dim)
    next_obs : npt.NDArray | torch.Tensor A set of next observations after
        taking the specified actions. Has shape (n_samples, obs_dim)
    dones : npt.NDArray | torch.Tensor A set of states of the episodes, where
        true denotes that the episode finished with that action. Has shape
        (n_samples, )
    """

    obs: npt.NDArray | torch.Tensor
    actions: npt.NDArray | torch.Tensor
    rewards: npt.NDArray | torch.Tensor
    prefs: npt.NDArray | torch.Tensor
    next_obs: npt.NDArray | torch.Tensor
    dones: npt.NDArray | torch.Tensor

    def as_tensors(self, device: torch.device) -> ReplaySample:
        """Convert the current sample from the buffer to torch tensors and
        move them to the desired device.

        Parameters
        ----------
        device : torch.device
            The device to which the data should be moved to.

        Returns
        -------
        ReplaySample:
            A copy of the current ReplaySample that has moved all the data
            to tensors in the preferred device.
        """
        if isinstance(self.obs, np.ndarray):
            obs = torch.from_numpy(self.obs).float().to(device)
        else:
            obs = self.obs.detach().clone().to(device)

        if isinstance(self.actions, np.ndarray):
            actions = torch.from_numpy(self.actions).float().to(device)
        else:
            actions = self.actions.detach().clone().to(device)

        if isinstance(self.rewards, np.ndarray):
            rewards = torch.from_numpy(self.rewards).float().to(device)
        else:
            rewards = self.rewards.detach().clone().to(device)

        if isinstance(self.prefs, np.ndarray):
            prefs = torch.from_numpy(self.prefs).float().to(device)
        else:
            prefs = self.prefs.detach().clone().to(device)

        if isinstance(self.next_obs, np.ndarray):
            next_obs = torch.from_numpy(self.next_obs).float().to(device)
        else:
            next_obs = self.next_obs.detach().clone().to(device)

        if isinstance(self.dones, np.ndarray):
            dones = torch.from_numpy(self.dones).float().to(device)
        else:
            dones = self.dones.detach().clone().to(device)

        return ReplaySample(
            obs=obs,
            actions=actions,
            rewards=rewards,
            prefs=prefs,
            next_obs=next_obs,
            dones=dones,
        )


class HistoryBuffer:
    def __init__(
        self,
        *,
        total_timesteps: int,
        eval_freq: int,
        n_eval_prefs: int,
        obs_dim: int,
        action_dim: int,
        reward_dim: int,
        n_critics: int | None = None
    ):
        """Defines a buffer that contains all the visited states, taken
        actions, received rewards etc. Furthermore, stores all the evaluation
        results of the policy.

        Parameters
        ----------
        total_timesteps : int
            The total amount of timesteps the algorithm is trained for.
        eval_freq : int
            The evaluation frequency for the policy
        n_eval_prefs : int
            The amount evaluation preferences used for evaluating the policy.j
        obs_dim : int
            The dimensionality of the environments observation space.
        action_dim : int
            The dimensionality of the environments action space.
        reward_dim : int
            The dimensionality of the reward space.
        """
        
        # The basics
        self._obs = np.empty((total_timesteps, obs_dim), dtype=np.float64)
        self._next_obs = np.empty((total_timesteps, obs_dim), dtype=np.float64)
        self._actions = np.empty((total_timesteps, action_dim), dtype=np.float64)
        self._rewards = np.empty((total_timesteps, reward_dim), dtype=np.float64)
        self._prefs = np.empty((total_timesteps, reward_dim), dtype=np.float64)
        self._dones = np.empty((total_timesteps,), dtype=bool)
        self._episodes = np.empty((total_timesteps,), dtype=np.uint64)

        self._step_ptr = 0

        # returns
        n_points = (total_timesteps // (eval_freq * 5)) * n_eval_prefs
        self._avg_disc_returns = np.empty((n_points, reward_dim), dtype=np.float64)
        self._sd_disc_returns = np.empty((n_points, reward_dim), dtype=np.float64)
        self._avg_returns = np.empty((n_points, reward_dim), dtype=np.float64)
        self._sd_returns = np.empty((n_points, reward_dim), dtype=np.float64)
        self._global_step = np.empty((n_points,), dtype=np.uint64)
        self._point_ptr = 0

        # losses
        self._loss_prefs = []
        if n_critics is not None: 
            self._critic_losses = {f"critic_{i}": [] for i in range(n_critics)}
        else:
            self._critic_losses = {}
        self._policy_losses = []
    

    def append_losses(
        self, 
        prefs: torch.Tensor | npt.NDArray,
        critic_losses: List[torch.Tensor],
        policy_losses: torch.Tensor
    ):

        if isinstance(prefs, torch.Tensor):
            prefs = prefs.cpu().numpy()
        if isinstance(policy_losses, torch.Tensor):
            policy_losses = policy_losses.cpu().numpy()

        self._loss_prefs.append(prefs)
        self._policy_losses.append(policy_losses)

        for i, loss in enumerate(critic_losses):
            if isinstance(loss, torch.Tensor):
                loss = loss.cpu().numpy()
            if f"critic_{i}" not in self._critic_losses:
                self._critic_losses[f"critic_{i}"] = []
            self._critic_losses[f"critic_{i}"].append(loss)

    def append_step(
        self,
        obs: torch.Tensor | npt.NDArray,
        action: torch.Tensor | npt.NDArray,
        rewards: torch.Tensor | npt.NDArray,
        prefs: torch.Tensor | npt.NDArray,
        next_obs: torch.Tensor | npt.NDArray,
        done: bool,
        episode: int,
    ):
        """Appends information from the taken step.j

        Parameters
        ----------
        obs : torch.Tensor | npt.NDArray
            The observation before taking the step.
        action : torch.Tensor | npt.NDArray
            The action taken from the observation.
        rewards : torch.Tensor | npt.NDArray
            The received reward from the action.
        prefs : torch.Tensor | npt.NDArray
            The preferences used when taking the action.
        next_obs : torch.Tensor | npt.NDArray
            The next observation after the action.
        done : bool
            Indicating if the episode ended or not.
        episode : int
            The episode during which the step was taken.
        """
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(prefs, torch.Tensor):
            prefs = prefs.cpu().numpy()
        if isinstance(next_obs, torch.Tensor):
            next_obs = next_obs.cpu().numpy()

        # Check for batch dimension
        if obs.ndim == 2:
            for i in range(obs.shape[0]):
                self._append_single(
                        obs=obs[i, :],
                        action=action[i, :],
                        rewards=rewards[i, :], 
                        prefs=prefs[i, :],
                        next_obs=next_obs[i, :],
                        done=done[i],
                        episode=episode
                    )
        else:
            self._append_single(
                    obs=obs,
                    action=action,
                    rewards=rewards, 
                    prefs=prefs,
                    next_obs=next_obs,
                    done=done,
                    episode=episode
                )

    def append_avg_returns(
        self, *, 
        avg_disc_returns: List[float], 
        sd_disc_returns: List[float], 
        avg_returns: List[float],
        sd_returns: List[float],
        step: int
    ):
        """Append new average returns from the evaluation of the agent.

        Parameters
        ----------
        avg_disc_returns : List[float]
            The average returns from the evaluation.
        sd_disc_returns : List[float]
            The standard deviations of the returns from the evaluation.
        step : int
            The step at which the agent was evaluated.
        """
        avg_disc_returns = np.asarray(avg_disc_returns)
        sd_disc_returns = np.asarray(sd_disc_returns)
        avg_returns = np.asarray(avg_returns)
        sd_returns = np.asarray(sd_returns)

        end_idx = self._point_ptr + avg_returns.shape[0]
        assert end_idx <= self._sd_disc_returns.shape[0], (
            "Overindexing! has store for "
            f"{self._sd_disc_returns.shape[0]} points, "
            f"tried to fit {end_idx} "
            "points instead!"
        )

        self._avg_disc_returns[self._point_ptr : end_idx, :] = avg_disc_returns
        self._sd_disc_returns[self._point_ptr : end_idx, :] = sd_disc_returns
        self._avg_returns[self._point_ptr : end_idx, :] = avg_returns
        self._sd_returns[self._point_ptr : end_idx, :] = sd_returns

        self._global_step[self._point_ptr : end_idx] = step
        self._point_ptr = end_idx

    def save_losses(self, save_path: str | pathlib.Path):
        save_path = pathlib.Path(save_path)

        if len(self._critic_losses) == 0:
            warnings.warn("No critic losses stored! Skipping the saving...")
            return
        
        if save_path.is_dir():
            import uuid
            filename = f"losses-{str(uuid.uuid4()[:8])}.npz"
            warnings.warn((f"{str(save_path)} is a directory, while path was "
                           f"expected! Saving to {str(save_path)}/{filename} "
                           "instead!"))
            save_path /= filename
            

        if save_path.exists():
            warnings.warn(f"{str(save_path)} Already exists! Overriding it...")

        
        # Convert the losses to arrays, and save them
        prefs = np.asarray(self._loss_prefs)
        policy_losses = np.asarray(self._policy_losses)
        critic_losses = {
                key: np.asarray(losses) for key, losses in 
                self._critic_losses.items()
        }
        np.savez(
                save_path,
                prefs=prefs,
                policy_losses=policy_losses,
                **critic_losses
        )

        

    def save_history(self, save_path: str | pathlib.Path):
        """Saves the step history.

        Parameters
        ----------
        save_path : str | pathlib.Path
            The path to which the data will be saved to. Should point to a
            .npz file.
        """
        save_path = pathlib.Path(save_path)

        if save_path.exists() or save_path.is_file():
            warnings.warn(f"{str(save_path)} Already exists! Overriding it...")
        np.savez(
            save_path,
            obs=self._obs,
            actions=self._actions,
            prefs=self._prefs,
            rewards=self._rewards,
            next_obs=self._next_obs,
            dones=self._dones,
            episodes=self._episodes,
        )

    def pareto_front_to_json(
        self, use_discounted_returns: bool = True
    ) -> Tuple[List[Dict[str, float | int]], List[Dict[str, float | int]]]:
        """Convert the pareto-front data into json format.

        Parameters
        ----------
        use_discounted_returns: bool, optional
            If set to true, discounted returns will be used. Otherwise, the 
            raw returns are returned.

        Returns
        -------
        Tuple[List[Dict[str, float | int]], List[Dict[str, float | int]]]
              Returns the pareto-front data in json. First returned value
              contains all the points, while the second contains only the
              non-dominated points.
        """
        # Convert the pareto-front to json compliant form

        def _pfront_to_json(returns, return_sds, prefs, steps):
            out = []
            for i in range(returns.shape[0]):
                return_dct = {}
                for ii in range(returns.shape[0]):
                    return_dct[f"avg_disc_return_{ii}"] = float(returns[i, ii])
                    return_dct[f"std_disc_return_{ii}"] = float(return_sds[i, ii])
                    return_dct[f"pref_{ii}"] = float(prefs[i, ii])
                return_dct["global_step"] = int(steps[i])

                # return_dct = {
                #     f"avg_disc_return_{j}": float(returns[i, j])
                #     for j in range(returns.shape[1])
                # }
                # return_sd_dct = {
                #     f"std_disc_return_{j}": float(return_sds[i, j])
                #     for j in range(return_sds.shape[1])
                # }
                # return_dct.update(return_sd_dct)
                # return_dct.update({"global_step": int(steps[i])})
                out.append(return_dct)
            return out

        pareto_front = []
        non_dom_pareto_front = []
        #  First, find the values where the global steps change.
        idx = np.where(np.diff(self._global_step) != 0)[0] + 1

        idx = np.concatenate(
            (np.asarray([0]), idx, np.asarray([self._global_step.shape[0]])), axis=0
        )
    
        # Select the approriate returns
        if use_discounted_returns:
            avg_returns = self._avg_disc_returns
            sd_returns = self._sd_disc_returns
        else:
            avg_returns = self._avg_returns
            sd_returns = self._sd_returns

        for i in range(idx.shape[0] - 1):
            start_idx = idx[i]
            end_idx = idx[i + 1]
            selected_returns = avg_returns[start_idx:end_idx, :]
            return_sds = sd_returns[start_idx:end_idx, :]
            global_steps = self._global_step[start_idx:end_idx]
            prefs = self._prefs[start_idx:end_idx, :]

            # Store the unfiltered pareto-front
            json_pfront = _pfront_to_json(
                    selected_returns, return_sds, prefs, global_steps
            )
            pareto_front.extend(json_pfront)

            # Store the pareto-front containing only the non-dominated
            # indices
            non_dom_inds = pareto.get_non_pareto_dominated_inds(
                selected_returns, remove_duplicates=True
            )
            non_dom_avg_returns = selected_returns[non_dom_inds, :]
            non_dom_return_sds = return_sds[non_dom_inds, :]
            non_dom_steps = global_steps[non_dom_inds]
            non_dom_prefs = prefs[non_dom_inds]
            json_non_dom_pfront = _pfront_to_json(
                non_dom_avg_returns, non_dom_return_sds, 
                non_dom_prefs, non_dom_steps
            )
            non_dom_pareto_front.extend(json_non_dom_pfront)
        return pareto_front, non_dom_pareto_front

    def _append_single(
        self,
        *,
        obs: torch.Tensor,
        action: torch.Tensor | npt.NDArray,
        rewards: torch.Tensor | npt.NDArray,
        prefs: torch.Tensor | npt.NDArray,
        next_obs: torch.Tensor | npt.NDArray,
        done: bool,
        episode: int,
    ):
        self._obs[self._step_ptr, ...] = obs
        self._actions[self._step_ptr, ...] = action
        self._prefs[self._step_ptr, ...] = prefs
        self._rewards[self._step_ptr, ...] = rewards
        self._next_obs[self._step_ptr, ...] = next_obs
        self._dones[self._step_ptr] = done
        self._episodes[self._step_ptr] = episode
        self._step_ptr += 1


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        obs_dim: int,
        action_dim: int,
        reward_dim: int,
        use_torch: bool = True,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ):
        """Create a FIFO style replay buffer.

        Parameters
        ----------
        capacity : int
            The maximum capacity of the buffer. After this, the older samples
            will be overwritten by the new samples.
        obs_dim : int
            The dimension of the observations.
        action_dim : int
            The dimension of the actions.
        reward_dim : int
            The dimension of the rewards.
        seed : int | None
            The seed used for the random number generator. Defaulf None
        """
        if use_torch:
            self._device = torch.device("cpu" if device is None else device)
            self._obs = torch.empty((capacity, obs_dim), dtype=torch.float32).to(device)
            self._actions = torch.empty((capacity, action_dim), dtype=torch.float32).to(
                device
            )
            self._rewards = torch.empty((capacity, reward_dim), dtype=torch.float32).to(
                device
            )

            self._prefs = torch.empty((capacity, reward_dim), dtype=torch.float32).to(
                device
            )
            self._next_obs = torch.empty((capacity, obs_dim), dtype=torch.float32).to(
                device
            )

            # Store as float's to ensure that one can use them in basic
            # arithmetic operations
            self._dones = torch.empty((capacity,), dtype=torch.float32).to(device)

        else:
            # defined only for compliance
            self._device = None
            self._obs = np.empty((capacity, obs_dim))
            self._actions = np.empty((capacity, action_dim))
            self._rewards = np.empty((capacity, reward_dim))
            self._prefs = np.empty((capacity, reward_dim))
            self._next_obs = np.empty((capacity, obs_dim))
            self._dones = np.empty((capacity,), dtype=bool)

        self._ptr = 0
        self._len = 0
        self._capacity = capacity
        self._rng = np.random.default_rng(seed)

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise AttributeError(
                ("Running buffer in Numpy mode! No concept " "of device exists")
            )
        return self._device

    def append(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        rewards: torch.Tensor,
        prefs: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
    ):
        """Add a new sample to the buffer. If the buffer is full, the oldest
        sample will be overwritten.

        Parameters
        ----------
        obs : torch.Tensor
            The observation to add.
        action : torch.Tensor
            The taken action .
        rewards : torch.Tensor
            The rewards to add.
        prefs : torch.Tensor
            The current preferences over the objectives.
        next_obs : torch.Tensor
            The next observation after the taking the action.
        done : bool
            The state of the episode (i.e. is it finished or not).
        """

        # check if the data has batch dimension
        if obs.ndim == 2:
            done = torch.tensor(done, dtype=torch.float32, device=self._device)
            for i in range(obs.shape[0]):
                self._append_single(
                    obs=obs[i, :],
                    action=action[i, :],
                    rewards=rewards[i, :],
                    prefs=prefs[i, :],
                    next_obs=next_obs[i, :],
                    done=done[i],
                )
        else:
            self._append_single(
                obs=obs,
                action=action,
                rewards=rewards,
                prefs=prefs,
                next_obs=next_obs,
                done=done,
            )

    def sample(self, n_samples: int) -> ReplaySample:
        """Sample the buffer for a set of observation, action, reward,
        preference, next observation done tuples.

        Parameters
        ----------
        n_samples : int
            The amount of samples to select. Should be at most the lenght of
            the buffer

        Returns
        -------
        ReplaySample
            A sample from the buffer, that contains a set of
            (obs, action, reward, prefs, next_obs, done) tuples, where
            the shapes will be (
                    (n_samples, obs_dim), (n_samples, action_dim),
                    (n_samples, reward_dim), (n_samples, reward_dim),
                    (n_samples, obs_dim), (n_samples, )
            ) correspondingly
        """

        if n_samples > self._len:
            raise ValueError(
                (
                    f"'n_samples' should be less or equal to the "
                    f"length of the buffer. Got {n_samples} "
                    f"(len(buffer) = {self._len})"
                )
            )
        indexes = self._rng.choice(self._len, size=n_samples, replace=False)

        obs = self._obs[indexes, :]
        actions = self._actions[indexes, :]
        rewards = self._rewards[indexes, :]
        prefs = self._prefs[indexes, :]
        next_obs = self._next_obs[indexes, :]
        dones = self._dones[indexes]

        return ReplaySample(
            obs=obs,
            actions=actions,
            rewards=rewards,
            prefs=prefs,
            next_obs=next_obs,
            dones=dones,
        )

    def __len__(self) -> int:
        """Return the current length of the buffer. If the buffer is full
        (i.e. one is overwriting previous entries), the length == capacity


        Returns
        -------
        int
            The current lenght of the buffer.
        """
        return self._len

    def _append_single(
        self,
        *,
        obs: torch.Tensor,
        action: torch.Tensor,
        rewards: torch.Tensor,
        prefs: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        self._obs[self._ptr, ...] = obs
        self._actions[self._ptr, ...] = action
        self._rewards[self._ptr, ...] = rewards
        self._prefs[self._ptr, ...] = prefs
        self._next_obs[self._ptr, ...] = next_obs
        self._dones[self._ptr] = done

        self._ptr = (self._ptr + 1) % self._capacity

        if self._len < self._capacity:
            self._len += 1
