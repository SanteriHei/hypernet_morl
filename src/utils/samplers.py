import warnings
from typing import Any, Literal, Mapping

import numpy.typing as npt
import pymoo.util.ref_dirs
import torch

from . import common


class SingleSampler:
    def __init__(
        self,
        reward_dim: int,
        device: str | torch.device | None = None,
        **kwargs: Mapping[str, Any],
    ):
        self._reward_dim = reward_dim
        self._device = torch.device("cpu" if device is None else device)

        pref = kwargs.pop("pref", None)
        if pref is None:
            pref = torch.rand(
                self._reward_dim, dtype=torch.float32, device=self._device
            )
            pref = pref / torch.linalg.norm(pref, ord=1)
        else:
            pref = torch.tensor(pref, dtype=torch.float32, device=self._device)

        if ((tot_sum := pref.sum()) - 1.0).abs() > 1e-15:
            warnings.warn(f"Given preference {pref} doesn't sum to 1 ({tot_sum:.3f})")
        self._pref = pref

    @property
    def reward_dim(self) -> int:
        """Return the used reward dimensionality"""
        return self._reward_dim

    @property
    def device(self) -> torch.device:
        """Return the currently used device"""
        return self._device

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """Sample a new preference.

        Parameters
        ----------
        n_samples : int
            The amount of preferences to sample

        Returns
        -------
        torch.Tensor
            The sampled preferences.
        """
        if n_samples == 1:
            return self._pref
        return torch.stack([self._pref for _ in range(n_samples)])


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
                w[self._n_points // 2 :] = 1.75
                self._weights = w / torch.linalg.vector_norm(w, ord=1)
            else:
                self._weights = (
                    torch.ones(self._n_points, device=self._device) / self._n_points
                )
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
            self._weights, n_samples, replacement=False, generator=self._generator
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

            out[out_ptr : out_ptr + n_points] = self._ref_preferences[
                self._ptr : new_ptr
            ]
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
                (prefs.sum(axis=-1) - 1).abs() < 1e-7
            ).all(), f"Not all prefs sum to one: ({prefs.sum(axis=-1).min()}, {prefs.sum(axis=-1).max()})"
            return prefs

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
    _DEFAULT_ANGLE_DEG: float = 45.0

    def __init__(
        self,
        reward_dim: int,
        w: npt.NDArray | torch.Tensor | None = None,
        device: str | torch.device | None = None,
        seed: int | None = None,
        **kwargs: Mapping[str, Any],
    ):
        """Create a simple preference sampler that can be used to
        sample normalized preferences from a (possibly) restricted part of the
        space.

        Parameters
        ----------
        reward_dim : int
            The dimension of the rewards.
        angle_deg : float
            The angle that is used to restrict the weight sampling (in degrees)
        w : [TODO:parameter]
            [TODO:description]
        device: str | torch.device | None, optional
            The device where the tensors will be stored. If None, "cpu" is used
            as the default. Default None
        seed: int | None, optional
            The seed used to initialize the PRNG. Default None.
        """
        self._reward_dim = reward_dim

        if "angle_deg" not in kwargs:
            warnings.warn(
                (
                    "'angle_deg' not specified for preference samnpler! "
                    f"Using {self._DEFAULT_ANGLE_DEG} deg angle as default"
                )
            )
            angle_deg = self._DEFAULT_ANGLE_DEG
        else:
            angle_deg = kwargs.pop("angle_deg")
        self._angle = common.deg_to_rad(angle_deg)
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
        case "single":
            sampler = SingleSampler(reward_dim, device=device, seed=seed, **kwargs)
        case _:
            raise ValueError(
                (
                    f"Unknown sampler type {sampler_type!r}! Should "
                    "be one of 'normal', 'uniform' or 'static'"
                )
            )
    return sampler
