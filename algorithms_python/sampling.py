"""Sampling helpers for DAG based models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

@dataclass
class IndependentDagSample:
    prob: float
    sample: np.ndarray

def sample_ind_dag(
    whe_sam: bool,
    probs: np.ndarray,
    samples: np.ndarray | None = None,
    log: bool = False,
    rng: np.random.Generator | None = None,
) -> IndependentDagSample:
    """Sample an edge set assuming independent Bernoulli draws.

    Parameters
    ----------
    whe_sam:
        When ``True`` the ``probs`` matrix is used to randomly sample a
        support set.  Otherwise the provided ``samples`` indices are used.
    probs:
        Matrix with inclusion probabilities for each ordered pair.
    samples:
        Optional array of linear indices (row-major order) describing the
        selected edges.  Used when ``whe_sam`` is ``False``.
    log:
        When ``True`` the logarithm of the probability is returned.
    rng:
        Optional ``numpy`` random number generator.
    """

    rng = rng or np.random.default_rng()
    probs = np.asarray(probs, dtype=float)
    d = probs.shape[0]

    if whe_sam:
        draws = rng.random(size=(d, d))
        samples = np.flatnonzero((draws < probs).flatten(order="F"))
    elif samples is None:
        raise ValueError("`samples` must be provided when `whe_sam` is False")

    flat_probs = probs.flatten(order="F")
    if log:
        prob = float(np.log(flat_probs[samples]).sum())
    else:
        prob = float(np.prod(flat_probs[samples]))

    return IndependentDagSample(prob=prob, sample=np.asarray(samples, dtype=int))

__all__ = ["IndependentDagSample", "sample_ind_dag"]
