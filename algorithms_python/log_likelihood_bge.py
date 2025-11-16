"""BGe score helpers.

The original R code evaluates marginal likelihood tables using the
``BiDAG`` package.  The Python translation expects the caller to provide
an equivalent scoring callable via ``hyper_par['score_function']``.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable

import numpy as np

from .dag_utils import model_encoding
from .structures import LaplaceApproximation

ScoreFn = Callable[[int, np.ndarray], float]

def _score_node(score_fn: ScoreFn, node: int, parents: np.ndarray) -> float:
    parents = np.asarray(parents, dtype=int)
    return float(score_fn(node, parents))

def log_llh_bge_table(la: LaplaceApproximation, hyper_par: Dict[str, any]) -> LaplaceApproximation:
    gamma = la.curr
    permi_pars = hyper_par["permi_pars"]
    tables = hyper_par["tables"]
    p = hyper_par["p"]
    score_fn: ScoreFn = hyper_par["score_function"]

    A = np.zeros(p)
    for j in range(p):
        parents_j = np.where(gamma[:, j] == 1)[0]
        if parents_j.size == 0:
            A[j] = tables[j][1][0]
            continue
        perm_parents = permi_pars[j]
        missing = np.setdiff1d(parents_j, perm_parents, assume_unique=True)
        if missing.size == 0:
            m = model_encoding(gamma[perm_parents, j])
            A[j] = tables[j][1][m - 1]
        elif missing.size == 1:
            idx = int(np.where(tables[j][0] == missing[0])[0][0])
            m = model_encoding(gamma[perm_parents, j])
            A[j] = tables[j][idx][m - 1]
        else:
            A[j] = _score_node(score_fn, j, parents_j)

    la.llh = float(A.sum())
    la.A = A
    la.p_gam = int(gamma.sum())
    return la

def log_llh_bge_update_table(
    changes: Iterable[int],
    la_old: LaplaceApproximation,
    la: LaplaceApproximation,
    hyper_par: Dict[str, any],
) -> LaplaceApproximation:
    A = la_old.A.copy()
    gamma = la.curr
    permi_pars = hyper_par["permi_pars"]
    tables = hyper_par["tables"]
    score_fn: ScoreFn = hyper_par["score_function"]

    num_parents = gamma.sum(axis=0)
    for j in changes:
        if num_parents[j] == 0:
            A[j] = tables[j][1][0]
            continue
        parents_j = np.where(gamma[:, j] == 1)[0]
        perm_parents = permi_pars[j]
        missing = np.setdiff1d(parents_j, perm_parents, assume_unique=True)
        if missing.size == 0:
            m = model_encoding(gamma[perm_parents, j])
            A[j] = tables[j][1][m - 1]
        elif missing.size == 1:
            idx = int(np.where(tables[j][0] == missing[0])[0][0])
            m = model_encoding(gamma[perm_parents, j])
            A[j] = tables[j][idx][m - 1]
        else:
            A[j] = _score_node(score_fn, j, parents_j)

    la.llh = float(A.sum())
    la.A = A
    la.p_gam = int(gamma.sum())
    return la

__all__ = ["log_llh_bge_table", "log_llh_bge_update_table"]
