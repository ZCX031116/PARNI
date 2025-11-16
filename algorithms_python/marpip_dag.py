"""Pre-compute marginal PIPs and score tables for DAG models."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .dag_utils import h_to_permi_pars, model_encoding

def marpips_dag_h(hyper_par: Dict[str, any], kappa: float = 0.0) -> Dict[str, any]:
    n = hyper_par["n"]
    p = hyper_par["p"]
    permi_pars = hyper_par["permi_pars"]

    pips = np.zeros((p, p))
    tables: List[List[np.ndarray]] = []

    for j in range(p):
        result = enumerate_full_models(j, permi_pars[j], hyper_par)
        pips[:, j] = result["PIPs"]
        tables.append(result["table"])

    pips = pips * (1 - 2 * kappa) + kappa
    ratios = pips / (1 - pips)
    pips = ratios / (ratios + ratios.T + 1)

    return {"tables": tables, "PIPs": pips}

def enumerate_full_models(j: int, pa_j: np.ndarray, hyper_par: Dict[str, any]):
    X = hyper_par["X"]
    g = hyper_par["g"]
    n = hyper_par["n"]
    p = hyper_par["p"]
    h = hyper_par["h"]
    log_m_prior = hyper_par["log_m_prior"]
    max_p = hyper_par["max_p"]

    xtx = hyper_par["XtX"]
    inv_g = 1.0 / g

    pips = np.zeros(p)
    available = np.setdiff1d(np.arange(p), np.concatenate(([j], pa_j)))
    index = np.concatenate(([0, 0], available))
    table: List[np.ndarray] = [index]

    xj_txj = xtx[j, j]
    log_llh_null = -n * np.log(xj_txj / 2.0) / 2.0
    lmp_null = log_m_prior(0, h, max_p)

    C = 1.0
    max_C = log_llh_null + lmp_null
    p_pa = len(pa_j)
    k_power = 2 ** p_pa

    for idx in range(1, len(index)):
        k = index[idx]
        if k == 0:
            pa_j_new = pa_j
        else:
            pa_j_new = np.concatenate((pa_j, [k]))
        p_pa_new = len(pa_j_new)
        if p_pa_new == 0:
            table.append(np.array([log_llh_null]))
            continue

        V = xtx[np.ix_(pa_j_new, pa_j_new)].astype(float)
        V[np.diag_indices_from(V)] += inv_g
        x_paj_xj_t = xtx[pa_j_new, j]

        log_llhs = np.full(k_power, log_llh_null)
        curr = np.zeros(p_pa_new, dtype=int)
        if p_pa_new != p_pa:
            curr[-1] = 1

        for mask in range(k_power):
            if p_pa > 0:
                bits = ((mask >> np.arange(p_pa)) & 1).astype(int)
                curr[:p_pa] = bits
            else:
                curr[:] = 0

            included = np.nonzero(curr)[0]
            if included.size == 0:
                continue

            parents_incl = pa_j_new[included]
            V_sub = V[np.ix_(included, included)]
            x_curr = x_paj_xj_t[included]
            chol = np.linalg.cholesky(V_sub)
            sol = np.linalg.solve(chol, x_curr)
            quad = float(np.dot(sol, sol))
            log_det = float(np.log(np.diag(chol)).sum())

            log_llh = -included.size * np.log(g) / 2.0 - n * np.log((xj_txj - quad) / 2.0) / 2.0 - log_det
            lmp = log_m_prior(included.size, h, max_p)
            prob_curr = np.exp(log_llh + lmp - max_C)

            if prob_curr > 1:
                pips /= prob_curr
                C /= prob_curr
                max_C = log_llh + lmp
                prob_curr = 1.0

            pips[parents_incl] += prob_curr

            if p_pa_new != p_pa:
                encoded = model_encoding(curr[:-1])
            else:
                encoded = model_encoding(curr)
            log_llhs[encoded - 1] = log_llh
            C += prob_curr

        table.append(log_llhs)

    return {"table": table, "PIPs": pips / C}

__all__ = ["marpips_dag_h", "enumerate_full_models"]
