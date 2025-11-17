"""Pointwise Python translation of the PARNI-DAG sampler."""
from __future__ import annotations

import time
from typing import Callable, Dict, List

import numpy as np

from .compute_la_dag import compute_la_dag
from math import lgamma

from .dag_utils import h_to_permi_pars
from .log_likelihood import log_llh_dag_table
from .log_likelihood_bge import log_llh_bge_table, log_llh_bge_update_table
from .log_likelihood_update import log_llh_dag_update_table
from .marpip_dag import marpips_dag_h
from .marpip_bge import marpips_bge_h
from .sampling import sample_ind_dag
from .structures import LaplaceApproximation
from .update_la_dag import update_la_dag

Array = np.ndarray

def _setup_hyper_par(alg_par: Dict[str, any], hyper_par: Dict[str, any]) -> Dict[str, any]:
    hyper = hyper_par.copy()
    hyper.setdefault("XtX", hyper["X"] @ hyper["X"].T)
    hyper.setdefault("max_p", hyper["p"] * (hyper["p"] - 1) // 2)
    hyper["permi_pars"] = h_to_permi_pars(np.asarray(alg_par["H"]))
    h = hyper["h"]
    p = hyper["p"]
    if np.ndim(h) == 0 or len(np.atleast_1d(h)) == 1:
        h_val = float(np.atleast_1d(h)[0])
        hyper["log_m_prior"] = lambda p_gam, *_: p_gam * (np.log(h_val) - np.log(1 - h_val))
    else:
        alpha, beta = h
        hyper["log_m_prior"] = lambda p_gam, *_: lgamma(p_gam + alpha) + lgamma(p - p_gam + beta) - lgamma(p + alpha + beta)
    return hyper

def parni(alg_par: Dict[str, any], hyper_par: Dict[str, any]) -> Dict[str, any]:
    hyper = _setup_hyper_par(alg_par, hyper_par)
    p = hyper["p"]
    max_p = hyper["max_p"]

    N = alg_par["N"]
    Nb = alg_par["Nb"]
    n_chain = alg_par["n_chain"]
    store_chains = alg_par.get("store_chains", False)
    omega = alg_par.get("omega_init", 0.9)
    bal_fun = alg_par.get("bal_fun", lambda x: np.minimum(1.0, x))
    kappa = alg_par.get("kappa", 0.0)
    f = alg_par.get("eval_f")
    rng = alg_par.get("rng", np.random.default_rng())

    use_bge = hyper.get("use_bge", False)
    if use_bge:
        hyper["log_llh"] = log_llh_bge_table
        hyper["log_llh_update"] = log_llh_bge_update_table
        res = marpips_bge_h(hyper, kappa)
    else:
        hyper["log_llh"] = log_llh_dag_table
        hyper["log_llh_update"] = log_llh_dag_update_table
        res = marpips_dag_h(hyper, kappa)
    hyper["tables"] = res["tables"]

    approx_PIPs = res["PIPs"]
    PIPs = approx_PIPs.copy()
    A = np.minimum(PIPs / (1 - PIPs), 1)
    np.fill_diagonal(A, 0)
    D = np.minimum((1 - PIPs) / PIPs, 1)
    np.fill_diagonal(D, 0)

    swap_idx = np.zeros(p * p, dtype=int)
    for idx in range(p * p):
        row = idx % p
        col = idx // p
        swap_idx[idx] = col + row * p
    hyper["swap_idx"] = swap_idx

    chains: List[List[Array]] = [[] for _ in range(n_chain)] if store_chains else []
    LAs: List[LaplaceApproximation] = []
    log_posts = np.full((N + 1, n_chain), np.nan)
    model_sizes = np.full((N + 1, n_chain), np.nan)
    estm_PIPs = np.zeros((p, p))
    sum_PIPs = np.zeros((p, p))
    infs = np.zeros((2, max_p + 1))

    for i in range(n_chain):
        gamma_init = alg_par.get("gamma_init", np.zeros((p, p), dtype=int))
        la = compute_la_dag(gamma_init, hyper)
        LAs.append(la)
        log_posts[0, i] = la.log_post
        model_sizes[0, i] = la.p_gam
        if store_chains:
            chains[i].append(gamma_init.copy())

    acc_times = 0
    mut = 0
    ESJD = 0.0
    k_sizes = 0
    sum_f = 0.0
    eval_f = f is not None

    start_total = time.time()
    start_iter = None

    for iter in range(1, N + 1):
        if iter == Nb + 1:
            start_iter = time.time()
        for i in range(n_chain):
            la = LAs[i]
            curr = la.curr
            eta = (1 - curr) * A + curr * D
            neighs = sample_ind_dag(True, eta, rng=rng)
            k = neighs.sample
            if k.size == 0:
                continue
            updates = update_la_dag(
                la,
                k,
                hyper,
                bal_fun,
                PIPs,
                thinning_rate=omega,
                omega=0.5,
                rng=rng,
            )
            JD = updates["JD"]
            acc_rate = updates["acc_rate"]
            thinned_k_size = updates["thinned_k_size"]
            k_sizes += thinned_k_size
            if JD > 0 and rng.random() < acc_rate:
                la_prop = updates["LA_prop"]
                LAs[i] = la_prop
                curr = la_prop.curr
                if iter > Nb:
                    acc_times += 1
                    mut += 1
            model_sizes[iter, i] = LAs[i].p_gam
            log_posts[iter, i] = LAs[i].log_post
            if iter > Nb:
                estm_PIPs += curr
                infs[0, JD] += 1
                infs[1, JD] += acc_rate
                if eval_f:
                    sum_f += f(curr)
                ESJD += acc_rate * JD
            if store_chains:
                chains[i].append(curr.copy())
        sum_curr = np.add.reduce([la.curr for la in LAs]) / n_chain
        sum_PIPs += sum_curr
        if iter > Nb:
            PIPs = kappa + (1 - 2 * kappa) * sum_PIPs / iter
            np.fill_diagonal(PIPs, 0)
            A = np.minimum(PIPs / (1 - PIPs), 1)
            np.fill_diagonal(A, 0)
            D = np.minimum((1 - PIPs) / PIPs, 1)
            np.fill_diagonal(D, 0)

    end_total = time.time()
    if start_iter is None:
        start_iter = start_total
    end_iter = time.time()

    c = (N - Nb) * n_chain
    infs[1] = np.divide(infs[1], infs[0], out=np.zeros_like(infs[1]), where=infs[0] > 0)
    infs[0] = np.divide(infs[0], c, out=np.zeros_like(infs[0]), where=c > 0)

    return {
        "chains": chains,
        "infs": infs,
        "log_post_trace": log_posts,
        "model_size_trace": model_sizes,
        "omega": omega,
        "acc_rate": acc_times / c if c else 0.0,
        "mut_rate": mut / c if c else 0.0,
        "estm_PIPs": estm_PIPs / c if c else estm_PIPs,
        "emp_PIPs": sum_PIPs / N,
        "approx_PIPs": approx_PIPs,
        "ad_PIPs": PIPs,
        "eval_f": sum_f / c if eval_f and c else None,
        "ESJD": ESJD / c if c else 0.0,
        "k_sizes": k_sizes / (N * n_chain),
        "CPU_time": np.array(
            [
                (end_total - start_total) / 60.0,
                (end_iter - start_total) / 60.0,
                (end_total - start_iter) / 60.0,
            ]
        ),
    }

__all__ = ["parni"]
