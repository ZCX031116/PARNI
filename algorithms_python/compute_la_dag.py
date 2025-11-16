"""Laplace approximation helpers for DAG models."""
from __future__ import annotations

from typing import Dict, Protocol

import numpy as np

from .structures import LaplaceApproximation

class LogLikelihood(Protocol):
    def __call__(self, la: LaplaceApproximation, hyper_par: Dict[str, any]) -> LaplaceApproximation:
        ...

def compute_la_dag(gamma: np.ndarray, hyper_par: Dict[str, any]) -> LaplaceApproximation:
    """Evaluate the Laplace approximation for a DAG configuration."""

    la = LaplaceApproximation(curr=np.asarray(gamma, dtype=int), p_gam=int(np.sum(gamma)))
    log_llh = hyper_par["log_llh"]
    la = log_llh(la, hyper_par)
    log_m_prior = hyper_par["log_m_prior"]
    log_m = log_m_prior(la.p_gam, hyper_par["h"], hyper_par["p"])
    la.lmp = float(log_m)
    la.log_post = float(la.llh + log_m)
    return la

__all__ = ["compute_la_dag", "LaplaceApproximation"]
