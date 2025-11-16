"""BGe variant of the marginal PIP pre-computation."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .dag_utils import model_encoding

def marpips_bge_h(hyper_par: Dict[str, any], kappa: float = 0.0) -> Dict[str, any]:
    p = hyper_par["p"]
    permi_pars = hyper_par["permi_pars"]

    pips = np.zeros((p, p))
    tables: List[List[np.ndarray]] = []

    for j in range(p):
        result = enumerate_full_models_bge(j, permi_pars[j], hyper_par)
        pips[:, j] = result["PIPs"]
        tables.append(result["table"])

    pips = pips * (1 - 2 * kappa) + kappa
    ratios = pips / (1 - pips)
    pips = ratios / (ratios + ratios.T + 1)

    return {"tables": tables, "PIPs": pips}

def enumerate_full_models_bge(j: int, pa_j: np.ndarray, hyper_par: Dict[str, any]):
    p = hyper_par["p"]
    score_fn = hyper_par["score_function"]

    pips = np.zeros(p)
    available = np.setdiff1d(np.arange(p), np.concatenate(([j], pa_j)))
    index = np.concatenate(([0, 0], available))
    table: List[np.ndarray] = [index]

    log_llh_null = float(score_fn(j, np.array([], dtype=int)))
    C = np.exp(log_llh_null)
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
            log_llh = float(score_fn(j, parents_incl))
            prob_curr = np.exp(log_llh)
            pips[parents_incl] += prob_curr

            if p_pa_new != p_pa:
                encoded = model_encoding(curr[:-1])
            else:
                encoded = model_encoding(curr)
            log_llhs[encoded - 1] = log_llh
            C += prob_curr

        table.append(log_llhs)

    return {"table": table, "PIPs": pips / C}

__all__ = ["marpips_bge_h", "enumerate_full_models_bge"]
