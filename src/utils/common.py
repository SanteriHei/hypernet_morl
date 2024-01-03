""" Some common utilities for the algorithms"""
from __future__ import annotations

import itertools
import json
import numbers
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import numpy.typing as npt
import pymoo.util.ref_dirs
import torch
from ruamel.yaml import YAML


class NumpyEncoder(json.JSONEncoder):
    """
    A custom encoder that converts numpy values into native Python types before
    they are serialized.
    """
    def default(self, obj):
        
        # Only catch the numeric scalar values
        if (
                isinstance(obj, numbers.Number) and np.ndim(obj) == 0 and 
                isinstance(obj, np.generic)
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
        raise FileExistsError(
            f"{filepath!r} already exists! (and is not a file)"
        )
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
        raise FileExistsError(
            f"{filepath!r} already exists! (and is not a file)"
        )
    with fpath.open("w") as ofstream:
        yaml.dump(payload, ofstream)

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
    return (angle/180) * np.pi


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


class WeightSampler:

    def __init__(
            self, reward_dim: int, angle_rad: float,
            w: npt.NDArray | torch.Tensor | None = None,
            device: str | torch.device | None = None,
            seed: int | None = None
    ):
        """Create a simple weight sampler that can be used to 
        sample normalized weights from a (possibly) restricted part of the 
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
    def device(self) -> torch.device:
        """ Return the currently used device """
        return self._device

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample weight from the specified weight space.

        Parameters
        ----------
        n_samples : int
            The amount of weight samples to draw.

        Returns
        -------
        torch.Tensor
            The sampled weights.
        """
        samples = torch.normal(
                torch.zeros(n_samples, self._reward_dim, device=self._device),
                generator=self._generator
        )

        # Remove fluctutation on dir w.
        samples = (
            samples - (samples @ self._w).view(-1, 1) * self._w.view(1, -1)
        )
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
            obs=obs, actions=actions, rewards=rewards,
            prefs=prefs, next_obs=next_obs, dones=dones
        )


class ReplayBuffer:

    def __init__(
            self, capacity: int, *,
            obs_dim: int,
            action_dim: int,
            reward_dim: int,
            use_torch: bool = True,
            device: str | torch.device | None = None,
            seed: int | None = None
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
            self._obs = torch.empty(
                (capacity, obs_dim), dtype=torch.float32
            ).to(device)
            self._actions = torch.empty(
                (capacity, action_dim), dtype=torch.float32
            ).to(device)
            self._rewards = torch.empty(
                (capacity, reward_dim), dtype=torch.float32
            ).to(device)

            self._prefs = torch.empty(
                (capacity, reward_dim), dtype=torch.float32
            ).to(device)
            self._next_obs = torch.empty(
                (capacity, obs_dim), dtype=torch.float32
            ).to(device)

            # Store as float's to ensure that one can use them in basic
            # arithmetic operations
            self._dones = torch.empty(
                (capacity,), dtype=torch.float32
            ).to(device)

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
            raise AttributeError(("Running buffer in Numpy mode! No concept "
                                  "of device exists"))
        return self._device

    def append(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            rewards: torch.Tensor,
            prefs: torch.Tensor,
            next_obs: torch.Tensor,
            done: bool
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

    def sample(
            self, n_samples: int
    ) -> ReplaySample:
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
            raise ValueError((f"'n_samples' should be less or equal to the "
                              f"length of the buffer. Got {n_samples} "
                              f"(len(buffer) = {self._len})"))
        indexes = self._rng.choice(self._len, size=n_samples, replace=False)

        obs = self._obs[indexes, :]
        actions = self._actions[indexes, :]
        rewards = self._rewards[indexes, :]
        prefs = self._prefs[indexes, :]
        next_obs = self._next_obs[indexes, :]
        dones = self._dones[indexes]

        return ReplaySample(
            obs=obs, actions=actions, rewards=rewards, prefs=prefs,
            next_obs=next_obs, dones=dones
        )

    def __len__(self) -> int:
        """Return the current lenght of the buffer. If the buffer is full 
        (i.e. one is overwriting previous entries), the length == capacity


        Returns
        -------
        int
            The current lenght of the buffer.
        """
        return self._len
