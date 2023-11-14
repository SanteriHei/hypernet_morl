""" Utilities for pareto front calculations. 
NOTE: Code taken from morl-baselines https://github.com/LucasAlegre/morl-baselines"""
from __future__ import annotations

from typing import List

import numpy as np
import numpy.typing as npt


def get_non_pareto_dominated_inds(
        candidates: np.ndarray | List[float], remove_duplicates: bool = True
) -> npt.NDArray:
    """A batched and fast version of the Pareto coverage set algorithm.
    
    Parameters
    ----------
    candidates: npt.NDArray | List[float]
        The candidate vector.s
    remove_duplicates: bool, optional
        Whether to remove duplicate vectors. Defaults to True.

    Returns
    -------
    npt.NDArray
        The indices of the elements that should be kept to form the Pareto
        front or coverage set.
    """
    candidates = np.array(candidates)
    uniques, indcs, invs, counts = np.unique(
        candidates, return_index=True, return_inverse=True,
        return_counts=True, axis=0
    )

    res_eq = np.all(candidates[:, None, None] <= candidates, axis=-1).squeeze()
    res_g = np.all(candidates[:, None, None] < candidates, axis=-1).squeeze()
    c1 = np.sum(res_eq, axis=-1) == counts[invs]
    c2 = np.any(~res_g, axis=-1)
    if remove_duplicates:
        to_keep = np.zeros(len(candidates), dtype=bool)
        to_keep[indcs] = 1
    else:
        to_keep = np.ones(len(candidates), dtype=bool)

    return np.logical_and(c1, c2) & to_keep


def filter_pareto_dominated(
        candidates: npt.NDarray | List[float], remove_duplicates: bool = True
) -> npt.NDArray:
    """A batched and fast version of the Pareto coverage set algorithm.

    Parameters:
        candidates: npt.NDArray
            The candidate vectors.
        remove_duplicates: bool, optional
            If set to True, duplicate candidates are removed, otherwise they 

    Returns:
        npt.NDArray: A Pareto coverage set.
    """
    candidates = np.array(candidates)
    if len(candidates) < 2:
        return candidates
    return candidates[get_non_pareto_dominated_inds(
        candidates, remove_duplicates=remove_duplicates
    )]
