# NOTE: ADD LICENSE HERE

# The performance metrics are taken from morl-baselines to ensure
# reproducability (www.github.com/LucasAlegre/morl-baselines)
import copy
from typing import Callable, List

import numpy as np
import numpy.typing as npt
from pymoo.indicators.distance_indicator import (
    DistanceIndicator as MooDistanceIndicator,
)
from pymoo.indicators.hv import HV as MooHyperVolume
from pymoo.indicators.igd_plus import IGDPlus as MooIGDPlus

UtilityFunc = Callable[[npt.ArrayLike, npt.ArrayLike], float]


def _igd_plus_max_dist(z, a, norm=None):
    """Define the IGD+ metric for the maximization task. Essentially, this
    is the equation 17 from Ishibuchi, Hisao, Hiroyuki Masuda, Yuki Tanigaki,
    and Yusuke Nojima. 2015. “Modified Distance Calculation in Generational
    Distance and Inverted Generational Distance.” In Evolutionary Multi-Criterion
    Optimization, https://doi.org/10.1007/978-3-319-15892-1_8. where a-z is
    changed to z-a.

    Parameters
    ----------
    z : npt.NDArray
        The reference set
    a : npt.NDArray
        The current pareto-front approximat
    norm : float
        The value used for normalizing the values

    Returns
    -------
    float
        The IGD+ metric for a maximization task
    """
    d = z - a
    d[d < 0] = 0
    d = d / norm
    return np.sqrt((d**2).sum(axis=1))


class IGDPlusMax(MooDistanceIndicator):
    def __init__(self, pf, **kwargs):
        super().__init__(pf, _igd_plus_max_dist, 1, **kwargs)


def get_igd_plus_max(
        ref_set: npt.NDArray, points: List[npt.ArrayLike]
) -> float:
    """Calculate the IGD+ metric for a maximization task.

    Parameters
    ----------
    ref_set : npt.NDArray
        The set of reference points.
    points : List[npt.ArrayLike]
        The current approximation of the pareto-front.

    Returns
    -------
    float
        The IGB+ metric for a maximization task.
    """
    return IGDPlusMax(ref_set)(np.array(points))


def get_igd_plus(
        ref_set: npt.NDArray, points: List[npt.ArrayLike]
) -> float:
    """Calculate the IGB+ metric for a minization task.

    Parameters
    ----------
    ref_set : npt.NDArray
        The set of reference points.
    points : List[npt.ArrayLike]
        The current approximation of the pareto-front.

    Returns
    -------
    float
        The IGB+ metric for a minimization task.
    """
    return MooIGDPlus(ref_set)(np.array(points))


def get_hypervolume(ref_point: npt.NDArray, points: List[npt.ArrayLike]) -> float:
    """
    Calculates the hypervolume metric for a given set of points and
    a given reference point (using pymoo)

    Parameters
    ----------
    ref_point : npt.NDArray
        The reference point
    points : List[npt.ArrayLike]
        The points to calculate the hypervolume to.

    Returns
    -------
    float
        The hypervolume indicator
    """
    return MooHyperVolume(ref_point=ref_point * -1)(np.array(points) * -1)


def get_sparsity(pareto_front: List[np.ndarray]) -> float:
    """
    Calculates the sparsity metric from PGMORL (insert paper details)
    Note that lower sparsity is better (i.e. we prefer dense approximations)

    Parameters
    ----------
    front : List[np.ndarray]
        The current pareto front to compute the sparsity on.
    Returns
    -------
    float
        The sparsity of the current front.
    """
    if len(pareto_front) < 2:
        return 0.0

    sparsity_value = 0.0
    m = len(pareto_front[0])
    front = np.array(pareto_front)
    for dim in range(m):
        objs_i = np.sort(copy.deepcopy(front.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity_value += np.square(objs_i[i] - objs_i[i - 1])
    sparsity_value /= len(front) - 1
    return sparsity_value


def get_expected_utility(
    front: List[npt.NDArray],
    prefs_set: List[npt.NDArray],
    utility_fn: UtilityFunc = np.dot,
) -> float:
    """
    Calculates the 'Expected Utility' of the pareto-front.
    Similar to R-metrics in MOO, but requires only one PDF approximation
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and
    P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Parameters
    ----------
    pareto_front : List[npt.NDArray]
        The current pareto-front approximation.
    prefs_set: List[npt.NDArray]
        The preferences used for the utility calculation.
    utility_fn : Callable[[npt.NDArrayLike, npt.NDArrayLike], float], Optional
        The utility function. Default np.dot

    Returns
    -------
    float
        The eum metric.
    """
    maxs = []
    for weights in prefs_set:
        scalarized_front = np.array([utility_fn(weights, point) for point in front])
        maxs.append(np.max(scalarized_front))

    return np.mean(np.array(maxs), axis=0)


def maximum_utility_loss(
    pareto_front: List[np.ndarray],
    reference_set: List[np.ndarray],
    weights_set: np.ndarray,
    utility_fn: UtilityFunc = np.dot,
) -> float:
    """Calculates the maximum utility metric

    Maximum utility loss of the policies on the PF for various weights.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and
    P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Parameters
    ----------
    pareto_front : List[np.ndarray]
        The current pareto-front approximation
    reference_set : List[np.ndarray]
        The reference set (e.g. True pareto front) to compute the mul on
    weights_set : np.ndarray
        The weights used for the utility computation
    utility_fn : Callable[[npt.NDArrayLike, npt.ArrayLike], float], optional
        The used utility function, Default np.dot

    Returns
    -------
    float
        The mul metric.
    """
    max_scalarized_values_ref = [
        np.max([utility_fn(weight, point) for point in reference_set])
        for weight in weights_set
    ]
    max_scalarized_values = [
        np.max([utility_fn(weight, point) for point in pareto_front])
        for weight in weights_set
    ]
    utility_losses = [
        max_scalarized_values_ref[i] - max_scalarized_values[i]
        for i in range(len(max_scalarized_values))
    ]
    return np.max(utility_losses)
