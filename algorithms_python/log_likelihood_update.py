"""Incremental log-likelihood updates for the PARNI sampler."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .dag_utils import model_encoding
from .log_likelihood import _solve_quadratic
from .structures import LaplaceApproximation

def log_llh_dag_update_table(
    changes: Iterable[int],
    la_old: LaplaceApproximation,
    la: LaplaceApproximation,
    hyper_par: Dict[str, any],
) -> LaplaceApproximation:
    log_det_sigma = la_old.log_det_sigma.copy()
    A = la_old.A.copy()

    gamma = la.curr
    XtX = hyper_par["XtX"]
    g = hyper_par["g"]
    n = hyper_par["n"]
    permi_pars = hyper_par["permi_pars"]
    tables = hyper_par["tables"]

    num_parents = gamma.sum(axis=0)

    for j in changes:
        if num_parents[j] == 0:
            log_det_sigma[j] = 0.0
            A[j] = tables[j][1][0]
            continue

        parents_j = np.where(gamma[:, j] == 1)[0]
        perm_parents = permi_pars[j]
        missing = np.setdiff1d(parents_j, perm_parents, assume_unique=True)

        if missing.size == 0:
            m = model_encoding(gamma[perm_parents, j])
            A[j] = tables[j][1][m - 1]
            log_det_sigma[j] = 0.0
            continue

        if missing.size == 1:
            idx = int(np.where(tables[j][0] == missing[0])[0][0])
            m = model_encoding(gamma[perm_parents, j])
            A[j] = tables[j][idx][m - 1]
            log_det_sigma[j] = 0.0
            continue

        Vg = XtX[np.ix_(parents_j, parents_j)].astype(float)
        Vg[np.diag_indices_from(Vg)] += 1.0 / g
        log_det, quad = _solve_quadratic(Vg, XtX[parents_j, j])
        log_det_sigma[j] = log_det
        A[j] = -parents_j.size * np.log(g) / 2.0 - n * np.log((XtX[j, j] - quad) / 2.0) / 2.0

    la.llh = float(-log_det_sigma.sum() + A.sum())
    la.p_gam = int(gamma.sum())
    la.A = A
    la.log_det_sigma = log_det_sigma
    return la

__all__ = ["log_llh_dag_update_table"]
