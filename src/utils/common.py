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
from typing import Any, Dict, Iterable, List, Literal, Mapping, Tuple

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


def get_preference_sampler(
    sampler_type: Literal["normal", "uniform", "static"],
    reward_dim: int,
    device: str | torch.device | None = None,
    seed: int | None = None,
    **kwargs: Mapping[str, Any],
):
    """Create and return a preference sampler based on the expected sampling
    type

    Parameters
    ----------
    sampler_type : Literal["normal", "uniform", "static"]
        The type of sampler. Can be one of "normal", "uniform" and "static".
    reward_dim : int
        The dimensionality of the reward.
    device : str | torch.device | None
        The device in which the results should be returned in.
    seed : int | None
        The seed used for the generator that is used for the random sampling.

    Returns
    -------
    Any
        The constructed preference sampler.
    """
    sampler = None
    match sampler_type:
        case "normal":
            sampler = PreferenceSampler(reward_dim, device=device, seed=seed, **kwargs)
        case "uniform":
            sampler = UniformSampler(reward_dim, device=device, seed=seed, **kwargs)
        case "static":
            sampler = StaticSampler(reward_dim, device=device, seed=seed, **kwargs)
        case _:
            raise ValueError(
                (
                    f"Unknown sampler type {sampler_type!r}! Should "
                    "be one of 'normal', 'uniform' or 'static'"
                )
            )

    return sampler


def set_thread_count(device: str | torch.device, n_threads: int):
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


class StaticSampler:
    N_POINTS: int = 50
    SAMPLING_TYPE: str = "choice"

    def __init__(
        self,
        reward_dim: int,
        device: str | torch.device | None = None,
        seed: int | None = None,
        **kwargs: Mapping[str, Any],
    ):
        """A static sampling scheme for the preferences, were the preferences
        are sampled from a predetermined set of points in order.

        Parameters
        ----------
        reward_dim : int
            The dimensionality of the reward space.
        device : str | torch.device | None
            The device where the tensors are stored to.
        seed : int | None
            The seed used to generate the points.
            (NOTE: has practically no effect in 2 dimensional spaces)
        """
        self._reward_dim = reward_dim
        self._device = torch.device("cpu" if device is None else device)
        self._n_points = kwargs.pop("n_points", StaticSampler.N_POINTS)
        self._sampling_type = kwargs.pop("sampling_type", StaticSampler.SAMPLING_TYPE)
        self._uneven_weighting = kwargs.pop("uneven_weighting", False)


        assert len(kwargs) == 0, f"Unknown kwargs: {kwargs}"

        if self._sampling_type == "choice":
            self._generator = torch.Generator(device=self._device)
            if seed is not None:
                self._generator.manual_seed(seed)

            if self._uneven_weighting:
                w = torch.ones(self._n_points, device=self._device)
                w[self._n_points//2:] = 1.75
                self._weights = w / torch.linalg.vector_norm(w, ord=1)
            else:
                self._weights = torch.ones(
                        self._n_points, device=self._device
                ) / self._n_points
        else:
            self._generator = None

        self._ptr = 0
        ref_preferences = pymoo.util.ref_dirs.get_reference_directions(
            name="energy", n_dim=reward_dim, n_points=self._n_points, seed=seed
        )
        self._ref_preferences = torch.from_numpy(ref_preferences).to(
            device=self._device, dtype=torch.float32
        )

    @property
    def reward_dim(self) -> int:
        """Return the used reward dimensionality"""
        return self._reward_dim

    @property
    def device(self) -> torch.device:
        """Return the currently used device"""
        return self._device

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """Sample preferences from the defined preference space.

        Parameters
        ----------
        n_samples : int
            The amount of samples to select.

        Returns
        -------
        torch.Tensor
            The sampled preferences.
        """

        
        match self._sampling_type:
            case "choice":
                return self._rnd_choice(n_samples)
            case "sequence":
                return self._get_sequence(n_samples)
            case _:
                raise ValueError("Unknown samping type!")

    def _rnd_choice(self, n_samples: int) -> torch.Tensor:
        idx = torch.multinomial(
                self._weights, n_samples,
                replacement=False, generator=self._generator
        )
        return self._ref_preferences[idx, :]

    def _get_sequence(self, n_samples) -> torch.Tensor:
        assert n_samples <= self._n_points, (
            "Cannot sample more than the amount of reference points "
            f"({self._n_points}) at a time!"
        )

        out = torch.full(
            (n_samples, self._reward_dim),
            torch.inf,
            dtype=torch.float32,
            device=self._device,
        )
        out_ptr = 0
        while out_ptr < n_samples:
            
            if self._ptr == self._n_points:
                self._ptr = 0

            new_ptr = min(self._n_points, self._ptr + out.shape[0] - out_ptr)

            n_points = new_ptr - self._ptr

            out[out_ptr:out_ptr+n_points] = self._ref_preferences[self._ptr:new_ptr]
            self._ptr = new_ptr
            out_ptr += n_points
        assert not out.isinf().any().item(), "Output contains Inf's!"
        return out

class UniformSampler:
    def __init__(
        self,
        reward_dim: int,
        device: str | torch.device | None = None,
        seed: int | None = None,
        **kwargs: Mapping[str, Any],
    ):
        """Create a random sampler that chooses samples with uniform random
        distribution from the preferece space.

        Parameters
        ----------
        reward_dim : int
            The dimensionality of the rewards.
        device : str | torch.device | None
            The device where the tensors will be stored. If None, "cpu" is used,
        seed : int | None
            The seed used to initialize the PRNG. Default None.
        """
        self._reward_dim = reward_dim
        self._device = torch.device("cpu" if device is None else device)

        # Use a local generator to manage the random state instead of the
        # global PRNG
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(seed)

    @property
    def reward_dim(self) -> int:
        """Return the used reward dimension"""
        return self._reward_dim

    @property
    def device(self) -> torch.device:
        """Return currently used device"""
        return self._device

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample preferences from the specified preference space.

        Parameters
        ----------
        n_samples : int
            The amount of preference samples to draw.

        Returns
        -------
        torch.Tensor
            The sampled preferences.
        """

        # If one is using only two rewards, just sample the single weight
        # from a uniform distribution, and calculate the second preference
        # based on that.
        if self._reward_dim == 2:
            pref_0 = torch.rand(
                size=(n_samples, 1), device=self._device, dtype=torch.float32
            )
            pref_1 = 1 - pref_0
            prefs = torch.concat((pref_0, pref_1), axis=-1)
            assert (
                (prefs.sum() - 1).abs() < 1e-7
            ).all(), f"Not all prefs sum to one: ({prefs.sum(axis=-1).min()}, {prefs.sum(axis=-1).max()})"
            return prefs
            # return torch.concat((pref_0, pref_1), axis=-1)

        # Otherwise, just sample the preferences from the uniform distribution
        # and normalize them.
        prefs = torch.rand(
            size=(n_samples, self._reward_dim),
            generator=self._generator,
            device=self._device,
            dtype=torch.float32,
        )
        prefs /= prefs.sum()
        return prefs


class PreferenceSampler:
    def __init__(
        self,
        reward_dim: int,
        angle_rad: float,
        w: npt.NDArray | torch.Tensor | None = None,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ):
        """Create a simple preference sampler that can be used to
        sample normalized preferences from a (possibly) restricted part of the
        space.

        Parameters
        ----------
        reward_dim : int
            The dimension of the rewards.
        angle_rad : float
            The angle that is used to restrict the weight sampling (in radians)
        w : [TODO:parameter]
            [TODO:description]
        device: str | torch.device | None, optional
            The device where the tensors will be stored. If None, "cpu" is used
            as the default. Default None
        seed: int | None, optional
            The seed used to initialize the PRNG. Default None.
        """
        self._reward_dim = reward_dim
        self._angle = angle_rad

        self._device = torch.device("cpu" if device is None else device)

        # Use a generator to manage the random state instead of the global PRNG
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(seed)

        if w is None:
            w = torch.ones(self._reward_dim, device=self._device)
        elif isinstance(w, npt.ndarray):
            w = torch.from_numpy(w).do(self._device)
        self._w = w / torch.norm(w)

    @property
    def reward_dim(self) -> int:
        """Return the used reward dimension"""
        return self._reward_dim

    @property
    def device(self) -> torch.device:
        """Return the currently used device"""
        return self._device

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample preferences from the specified preference space.

        Parameters
        ----------
        n_samples : int
            The amount of preference samples to draw.

        Returns
        -------
        torch.Tensor
            The sampled weights.
        """
        samples = torch.normal(
            torch.zeros(n_samples, self._reward_dim, device=self._device),
            generator=self._generator,
        )

        # Remove fluctutation on dir w.
        samples = samples - (samples @ self._w).view(-1, 1) * self._w.view(1, -1)
        samples = samples / torch.norm(samples, dim=1, keepdim=True)

        # Calculate the angle of the sampled vector, and shift it
        s_angle = torch.rand(n_samples, 1, device=self._device) * self._angle

        w_sample = torch.tan(s_angle) * samples + self._w.view(1, -1)
        w_sample = w_sample / torch.norm(w_sample, dim=1, keepdim=True, p=1)
        return w_sample.float().to(self._device)


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
        self._obs = np.empty((total_timesteps, obs_dim), dtype=np.float64)
        self._next_obs = np.empty((total_timesteps, obs_dim), dtype=np.float64)
        self._actions = np.empty((total_timesteps, action_dim), dtype=np.float64)
        self._rewards = np.empty((total_timesteps, reward_dim), dtype=np.float64)
        self._prefs = np.empty((total_timesteps, reward_dim), dtype=np.float64)
        self._dones = np.empty((total_timesteps,), dtype=bool)
        self._episodes = np.empty((total_timesteps,), dtype=np.uint64)

        self._step_ptr = 0

        n_points = (total_timesteps // (eval_freq * 5)) * n_eval_prefs
        self._avg_returns = np.empty((n_points, reward_dim), dtype=np.float64)
        self._sd_returns = np.empty((n_points, reward_dim), dtype=np.float64)
        self._global_step = np.empty((n_points,), dtype=np.uint64)
        self._point_ptr = 0

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

        self._obs[self._step_ptr, ...] = obs
        self._actions[self._step_ptr, ...] = action
        self._prefs[self._step_ptr, ...] = prefs
        self._rewards[self._step_ptr, ...] = rewards
        self._next_obs[self._step_ptr, ...] = next_obs
        self._dones[self._step_ptr] = done
        self._episodes[self._step_ptr] = episode
        self._step_ptr += 1

    def append_avg_returns(
        self, avg_returns: List[float], sd_returns: List[float], step: int
    ):
        """Append new average returns from the evaluation of the agent.

        Parameters
        ----------
        avg_returns : List[float]
            The average returns from the evaluation.
        sd_returns : List[float]
            The standard deviations of the returns from the evaluation.
        step : int
            The step at which the agent was evaluated.
        """
        avg_returns = np.asarray(avg_returns)
        sd_returns = np.asarray(sd_returns)
        end_idx = self._point_ptr + avg_returns.shape[0]
        assert end_idx <= self._sd_returns.shape[0], (
            "Overindexing! has store for "
            f"{self._sd_returns.shape[0]} points, "
            f"tried to fit {end_idx} "
            "points instead!"
        )

        self._avg_returns[self._point_ptr : end_idx, :] = avg_returns
        self._sd_returns[self._point_ptr : end_idx, :] = sd_returns
        self._global_step[self._point_ptr : end_idx] = step
        self._point_ptr = end_idx

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
        self,
    ) -> Tuple[List[Dict[str, float | int]], List[Dict[str, float | int]]]:
        """Convert the pareto-front data into json format.

        Returns
        -------
        Tuple[List[Dict[str, float | int]], List[Dict[str, float | int]]]
              Returns the pareto-front data in json. First returned value
              contains all the points, while the second contains only the
              non-dominated points.
        """
        # Convert the pareto-front to json compliant form

        def _pfront_to_json(returns, return_sds, steps):
            out = []
            for i in range(returns.shape[0]):
                return_dct = {
                    f"avg_disc_return_{j}": float(returns[i, j])
                    for j in range(returns.shape[1])
                }
                return_sd_dct = {
                    f"std_disc_return_{j}": float(return_sds[i, j])
                    for j in range(return_sds.shape[1])
                }
                return_dct.update(return_sd_dct)
                return_dct.update({"global_step": int(steps[i])})
                out.append(return_dct)
            return out

        pareto_front = []
        non_dom_pareto_front = []
        #  First, find the values where the global steps change.
        idx = np.where(np.diff(self._global_step) != 0)[0] + 1

        idx = np.concatenate(
            (np.asarray([0]), idx, np.asarray([self._global_step.shape[0]])), axis=0
        )
        for i in range(idx.shape[0] - 1):
            start_idx = idx[i]
            end_idx = idx[i + 1]
            avg_returns = self._avg_returns[start_idx:end_idx, :]
            return_sds = self._sd_returns[start_idx:end_idx, :]
            global_steps = self._global_step[start_idx:end_idx]

            # Store the unfiltered pareto-front
            json_pfront = _pfront_to_json(avg_returns, return_sds, global_steps)
            pareto_front.extend(json_pfront)

            # Store the pareto-front containing only the non-dominated
            # indices
            non_dom_inds = pareto.get_non_pareto_dominated_inds(
                avg_returns, remove_duplicates=True
            )
            non_dom_avg_returns = avg_returns[non_dom_inds, :]
            non_dom_return_sds = return_sds[non_dom_inds, :]
            non_dom_steps = global_steps[non_dom_inds]
            json_non_dom_pfront = _pfront_to_json(
                non_dom_avg_returns, non_dom_return_sds, non_dom_steps
            )
            non_dom_pareto_front.extend(json_non_dom_pfront)
        return pareto_front, non_dom_pareto_front


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
        self._obs[self._ptr, ...] = obs
        self._actions[self._ptr, ...] = action
        self._rewards[self._ptr, ...] = rewards
        self._prefs[self._ptr, ...] = prefs
        self._next_obs[self._ptr, ...] = next_obs
        self._dones[self._ptr] = done

        self._ptr = (self._ptr + 1) % self._capacity

        if self._len < self._capacity:
            self._len += 1

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
