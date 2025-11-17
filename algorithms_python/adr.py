"""Adaptive add-delete-reverse sampler with multiple chains."""
from __future__ import annotations

import time
from math import lgamma
from typing import Dict, List

import numpy as np

from .compute_la_dag import compute_la_dag
from .dag_utils import h_to_permi_pars, is_dag_adjmat
from .log_likelihood import log_llh_dag_table
from .marpip_dag import marpips_dag_h
from .structures import LaplaceApproximation

Array = np.ndarray

def _init_hyper_par(alg_par: Dict[str, any], hyper_par: Dict[str, any]) -> Dict[str, any]:
    hyper = hyper_par.copy()
    h = hyper["h"]
    p = hyper["p"]

    if np.ndim(h) == 0 or len(np.atleast_1d(h)) == 1:
        h_val = float(np.atleast_1d(h)[0])
        hyper["log_m_prior"] = lambda p_gam, *_: p_gam * (np.log(h_val) - np.log(1 - h_val))
    else:
        alpha, beta = h
        hyper["log_m_prior"] = lambda p_gam, *_: lgamma(p_gam + alpha) + lgamma(p - p_gam + beta) - lgamma(p + alpha + beta)
    hyper["log_llh"] = log_llh_dag_table
    hyper["permi_pars"] = h_to_permi_pars(np.asarray(alg_par["H"]))
    tables = marpips_dag_h(hyper)
    hyper["tables"] = tables["tables"]
    return hyper

def adr(alg_par: Dict[str, any], hyper_par: Dict[str, any]) -> Dict[str, any]:
    hyper = _init_hyper_par(alg_par, hyper_par)
    p = hyper["p"]
    max_p = p * (p - 1) // 2
    hyper["max_p"] = max_p

    N = alg_par["N"]
    Nb = alg_par["Nb"]
    n_chain = alg_par["n_chain"]
    store_chains = alg_par.get("store_chains", False)
    f = alg_par.get("f")
    rng = alg_par.get("rng", np.random.default_rng())

    randon_gamma_init = bool(alg_par.get("randon_gamma_init", False))
    h = hyper["h"]
    if np.ndim(h) == 0 or len(np.atleast_1d(h)) == 1:
        h_exp = float(np.atleast_1d(h)[0])
    else:
        alpha, beta = h
        h_exp = alpha / (alpha + beta)

    log_posts = np.full((N + 1, n_chain), np.nan)
    model_sizes = np.full((N + 1, n_chain), np.nan)
    estm_PIPs = np.zeros((p, p))
    infs = np.zeros((2, max_p))
    chains: List[List[Array]] = [[] for _ in range(n_chain)] if store_chains else []
    LAs: List[LaplaceApproximation] = []

    for i in range(n_chain):
        if randon_gamma_init:
            while True:
                gamma = (rng.random((p, p)) < h_exp).astype(int)
                np.fill_diagonal(gamma, 0)
                if is_dag_adjmat(gamma):
                    break
        else:
            gamma = np.zeros((p, p), dtype=int)
        la = compute_la_dag(gamma, hyper)
        LAs.append(la)
        log_posts[0, i] = la.log_post
        model_sizes[0, i] = la.p_gam
        if store_chains:
            chains[i].append(gamma.copy())

    acc_times = 0
    ESJD = 0.0
    propose_DAG = 0
    sum_f = 0.0
    eval_f = f is not None

    start_total = time.time()
    start_iter = None

    for iter in range(1, N + 1):
        if iter == Nb + 1:
            start_iter = time.time()
        for i in range(n_chain):
            la = LAs[i]
            curr = la.curr.copy()
            p_gam = la.p_gam
            if 1 <= p_gam <= max_p - 1:
                if rng.random() < 1 / 3:
                    edges = np.argwhere(curr == 1)
                    if edges.size == 0:
                        continue
                    edge = edges[rng.integers(len(edges))]
                    change = [edge[0] + edge[1] * p, edge[1] + edge[0] * p]
                    log_prop = 0.0
                    log_reverse = 0.0
                elif rng.random() < 1 / 2:
                    possible = curr + curr.T
                    np.fill_diagonal(possible, 1)
                    empties = np.argwhere(possible == 0)
                    if empties.size == 0:
                        continue
                    edge = empties[rng.integers(len(empties))]
                    change = [edge[0] + edge[1] * p]
                    log_prop = -np.log(len(empties))
                    log_reverse = -np.log(p_gam + 1)
                else:
                    connected = np.argwhere(curr == 1)
                    if connected.size == 0:
                        continue
                    edge = connected[rng.integers(len(connected))]
                    change = [edge[0] + edge[1] * p]
                    log_prop = -np.log(len(connected))
                    log_reverse = -np.log(2) - np.log(max_p - (len(connected) - 1))
            elif p_gam < 1:
                possible = np.zeros((p, p), dtype=int)
                np.fill_diagonal(possible, 1)
                empties = np.argwhere(possible == 0)
                edge = empties[rng.integers(len(empties))]
                change = [edge[0] + edge[1] * p]
                log_prop = -np.log(len(empties))
                log_reverse = -np.log(3)
            else:
                connected = np.argwhere(curr == 1)
                edge = connected[rng.integers(len(connected))]
                change = [edge[0] + edge[1] * p]
                log_prop = -np.log(len(connected))
                log_reverse = -np.log(3) - np.log(2)

            prop = curr.copy()
            for idx in change:
                r = idx % p
                c = idx // p
                prop[r, c] = 1 - prop[r, c]

            if is_dag_adjmat(prop):
                propose_DAG += 1
                la_prop = compute_la_dag(prop, hyper)
                log_ratio = la_prop.log_post + log_reverse - la.log_post - log_prop
                acc_rate = min(1.0, np.exp(log_ratio))
                if rng.random() < acc_rate:
                    LAs[i] = la_prop
                    curr = prop
                    p_gam = la_prop.p_gam
                    if iter > Nb:
                        acc_times += 1
                JD = len(change)
                if iter > Nb:
                    estm_PIPs += curr
                    infs[0, JD] += 1
                    infs[1, JD] += acc_rate
                    if eval_f:
                        sum_f += f(curr)
                    ESJD += acc_rate * JD
            model_sizes[iter, i] = p_gam
            log_posts[iter, i] = LAs[i].log_post
            if store_chains:
                chains[i].append(curr.copy())

    end_total = time.time()
    if start_iter is None:
        start_iter = start_total
    end_iter = time.time()

    c = (N - Nb) * n_chain
    infs[1] = np.divide(infs[1], infs[0], out=np.zeros_like(infs[1]), where=infs[0] > 0)
    infs[0] = infs[0] / c

    return {
        "chains": chains,
        "infs": infs,
        "log_post_trace": log_posts,
        "model_size_trace": model_sizes,
        "acc_rate": acc_times / c,
        "propose_DAG_rate": propose_DAG / (N * n_chain),
        "estm_PIPs": estm_PIPs / c,
        "f": sum_f / c if eval_f else None,
        "ESJD": ESJD / c,
        "CPU_time": np.array([
            (end_total - start_total) / 60.0,
            (end_iter - start_total) / 60.0,
            (end_total - start_iter) / 60.0,
        ]),
    }

__all__ = ["adr"]
