"""Log-likelihood computations for DAG models."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .dag_utils import model_encoding
from .structures import LaplaceApproximation

def _solve_quadratic(V: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    chol = np.linalg.cholesky(V)
    sol = np.linalg.solve(chol, x)
    quad = float(np.dot(sol, sol))
    log_det = float(np.log(np.diag(chol)).sum())
    return log_det, quad

def log_llh_dag(la: LaplaceApproximation, hyper_par: Dict[str, any]) -> LaplaceApproximation:
    gamma = la.curr
    XtX = hyper_par["XtX"]
    g = hyper_par["g"]
    p = gamma.shape[0]
    n = hyper_par["n"]

    p_gamma = int(gamma.sum())
    log_sqrt_det_sigma = np.zeros(p)
    A = -n * np.log(np.diag(XtX) / 2.0) / 2.0

    num_parents = gamma.sum(axis=0)
    nodes_connected = np.where(num_parents > 0)[0]

    for j in nodes_connected:
        parents_j = np.where(gamma[:, j] == 1)[0]
        Vg = XtX[np.ix_(parents_j, parents_j)].astype(float)
        Vg[np.diag_indices_from(Vg)] += 1.0 / g
        log_det, quad = _solve_quadratic(Vg, XtX[parents_j, j])
        log_sqrt_det_sigma[j] = log_det
        A[j] = -n * np.log((XtX[j, j] - quad) / 2.0) / 2.0

    log_llh = -p_gamma * np.log(g) / 2.0 - log_sqrt_det_sigma.sum() + A.sum()
    la.llh = float(log_llh)
    la.p_gam = p_gamma
    la.A = A
    la.log_det_sigma = log_sqrt_det_sigma
    return la

def log_llh_dag_table(la: LaplaceApproximation, hyper_par: Dict[str, any]) -> LaplaceApproximation:
    gamma = la.curr
    XtX = hyper_par["XtX"]
    g = hyper_par["g"]
    p = gamma.shape[0]
    n = hyper_par["n"]
    permi_pars = hyper_par["permi_pars"]
    tables = hyper_par["tables"]

    p_gamma = int(gamma.sum())
    log_sqrt_det_sigma = np.zeros(p)
    A = -n * np.log(np.diag(XtX) / 2.0) / 2.0

    num_parents = gamma.sum(axis=0)
    nodes_connected = np.where(num_parents > 0)[0]

    for j in nodes_connected:
        parents_j = np.where(gamma[:, j] == 1)[0]
        perm_parents = permi_pars[j]
        missing = np.setdiff1d(parents_j, perm_parents, assume_unique=True)

        if missing.size == 0:
            m = model_encoding(gamma[perm_parents, j])
            A[j] = tables[j][1][m - 1]
            continue

        if missing.size == 1:
            idx = np.where(tables[j][0] == missing[0])[0][0]
            m = model_encoding(gamma[perm_parents, j])
            A[j] = tables[j][idx][m - 1]
            continue

        Vg = XtX[np.ix_(parents_j, parents_j)].astype(float)
        Vg[np.diag_indices_from(Vg)] += 1.0 / g
        log_det, quad = _solve_quadratic(Vg, XtX[parents_j, j])
        log_sqrt_det_sigma[j] = log_det
        A[j] = -parents_j.size * np.log(g) / 2.0 - n * np.log((XtX[j, j] - quad) / 2.0) / 2.0

    la.llh = float(-log_sqrt_det_sigma.sum() + A.sum())
    la.p_gam = p_gamma
    la.A = A
    la.log_det_sigma = log_sqrt_det_sigma
    return la

__all__ = ["log_llh_dag", "log_llh_dag_table"]
