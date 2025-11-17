"""Neighbourhood update kernel used inside the PARNI sampler."""
from __future__ import annotations

from typing import Callable, Dict, Iterable

import numpy as np

from .dag_utils import is_dag_adjmat
from .log_likelihood_update import log_llh_dag_update_table
from .structures import LaplaceApproximation

Array = np.ndarray

Moves = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

def get_moves() -> Array:
    return Moves.copy()

def get_omega_vec(omega: float, change_idx: int, M: Array) -> Array:
    counts = np.abs(M - M[change_idx])
    return (omega ** counts.sum(axis=1)) * ((1 - omega) ** (2 - counts.sum(axis=1)))

def update_la_dag(
    la: LaplaceApproximation,
    k: Array,
    hyper_par: Dict[str, any],
    bal_fun: Callable[[float | Array], float | Array],
    pips: Array,
    thinning_rate: float,
    omega: float,
    rng: np.random.Generator | None = None,
) -> Dict[str, any]:
    rng = rng or np.random.default_rng()
    temp = la.curr.copy()
    la_temp = la.copy()
    log_post_temp = la.log_post
    llh_temp = la.llh
    lmp_temp = la.lmp

    max_p = hyper_par["max_p"]
    p = hyper_par["p"]
    h = hyper_par["h"]

    log_llh_update = hyper_par["log_llh_update"]
    log_m_prior = hyper_par["log_m_prior"]
    swap_idx = hyper_par["swap_idx"]

    k = np.asarray(k, dtype=int)
    k_swap = swap_idx[k]
    swaps = np.isin(k_swap, k)

    k_size = k.size
    grouped_k = np.column_stack((k, np.where(swaps, k_swap, np.inf)))
    grouped_k = grouped_k[grouped_k[:, 1] > grouped_k[:, 0]]

    if grouped_k.size == 0:
        grouped_k = np.column_stack((k, np.full_like(k, np.inf)))

    mask = rng.random(grouped_k.shape[0]) < thinning_rate
    thinned_k = grouped_k[mask]
    if thinned_k.size == 0:
        thinned_k = grouped_k[:1]
    thinned_k_size = thinned_k.shape[0]

    prob_prop = 0.0
    rev_prob_prop = 0.0
    prob_k_odds = 1.0
    JD = 0

    prod_bal_con = 0.0
    rev_prod_bal_con = 0.0

    for k_j in rng.permutation(thinned_k):
        if np.isfinite(k_j[1]):
            change_pair = tuple(int(x) for x in k_j)
            idxs = np.array(change_pair, dtype=int)
            temp_k = temp.flat[idxs]
            la_temps = [None] * 4
            prob_change_k_ratio = np.ones(4)
            odd_k_change = np.ones(4)
            mar_eff = pips.flat[idxs]
            prob_change = np.full(4, (1 - omega) ** 2)
            M = get_moves()

            for i in range(1, 4):
                temp_change = temp.copy()
                temp_change.flat[idxs[0]] = abs(M[i, 0] - temp_k[0])
                temp_change.flat[idxs[1]] = abs(M[i, 1] - temp_k[1])
                if temp_change.flat[idxs].sum() < 2 and is_dag_adjmat(temp_change):
                    la_temp_change = log_llh_update(
                        changes=np.unique(np.ceil((idxs + 1) / p).astype(int) - 1),
                        la_old=la_temp,
                        la=LaplaceApproximation(curr=temp_change, p_gam=int(temp_change.sum())),
                        hyper_par=hyper_par,
                    )
                    la_temps[i] = la_temp_change
                    llh_temp_change = la_temp_change.llh
                    lmp_temp_change = log_m_prior(la_temp_change.p_gam, h, max_p)
                    log_post_temp_change = llh_temp_change + lmp_temp_change
                    prob_ratio = np.exp(log_post_temp_change - log_post_temp)
                    odd_k_change[i] = np.prod((mar_eff / (1 - mar_eff)) ** ((2 * temp_k - 1) * M[i]))
                    prob_change_k_ratio[i] = prob_ratio * odd_k_change[i]
                    prob_change[i] = omega ** M[i].sum() * (1 - omega) ** (2 - M[i].sum()) * bal_fun(prob_change_k_ratio[i])
                else:
                    prob_change[i] = 0.0
            bal_const = prob_change.sum()
            if bal_const == 0:
                continue
            prob_change /= bal_const
            change_idx = np.searchsorted(np.cumsum(prob_change), rng.random())
            if change_idx == 0:
                rev_bal_const = bal_const
                prob_prop += np.log(prob_change[0])
                rev_prob_prop += np.log(prob_change[0])
                prod_bal_con += np.log(bal_const)
                rev_prod_bal_con += np.log(rev_bal_const)
                continue
            la_temp = la_temps[change_idx]
            log_post_temp = la_temp.llh + log_m_prior(la_temp.p_gam, h, max_p)
            temp = la_temp.curr
            JD += 2
            prob_k_odds *= odd_k_change[change_idx]
            rev_ratio = prob_change_k_ratio / prob_change_k_ratio[change_idx]
            rev_ratio[np.isnan(rev_ratio)] = 1.0
            rev_prob_change = bal_fun(rev_ratio) * get_omega_vec(omega, change_idx, M)
            rev_bal_const = rev_prob_change.sum()
            rev_prob_change /= rev_bal_const
            prob_prop += np.log(prob_change[change_idx])
            rev_prob_prop += np.log(rev_prob_change[0])
            prod_bal_con += np.log(bal_const)
            rev_prod_bal_con += np.log(rev_bal_const)
        else:
            idx = int(k_j[0])
            temp_change = temp.copy()
            temp_k = temp_change.flat[idx]
            temp_change.flat[idx] = 1 - temp_k
            if is_dag_adjmat(temp_change):
                la_temp_change = log_llh_update(
                    changes=np.unique(np.ceil((idx + 1) / p).astype(int) - 1),
                    la_old=la_temp,
                    la=LaplaceApproximation(curr=temp_change, p_gam=int(temp_change.sum())),
                    hyper_par=hyper_par,
                )
                llh_temp_change = la_temp_change.llh
                lmp_temp_change = log_m_prior(int(temp_change.sum()), h, max_p)
            else:
                llh_temp_change = 0.0
                lmp_temp_change = -np.inf
            log_post_temp_change = llh_temp_change + lmp_temp_change
            prob_change_ratio = np.exp(log_post_temp_change - log_post_temp)
            mar_eff = pips.flat[idx]
            odd_k_change = (mar_eff / (1 - mar_eff)) ** (2 * temp_k - 1)
            prob_change = omega * bal_fun(prob_change_ratio * odd_k_change)
            prob_keep = (1 - omega) * bal_fun(1)
            bal_const = prob_change + prob_keep
            prob_change /= bal_const
            if rng.random() < prob_change:
                rev_prob_change = omega * bal_fun(1 / prob_change_ratio / odd_k_change)
                rev_prob_keep = (1 - omega) * bal_fun(1)
                rev_bal_const = rev_prob_change + rev_prob_keep
                rev_prob_change /= rev_bal_const
                prob_prop += np.log(prob_change)
                rev_prob_prop += np.log(rev_prob_change)
                temp = temp_change
                JD += 1
                log_post_temp = log_post_temp_change
                llh_temp = llh_temp_change
                lmp_temp = lmp_temp_change
                la_temp = la_temp_change
                prob_k_odds *= odd_k_change
            else:
                rev_bal_const = bal_const
                prob_prop += np.log(1 - prob_change)
                rev_prob_prop += np.log(1 - prob_change)
            prod_bal_con += np.log(bal_const)
            rev_prod_bal_con += np.log(rev_bal_const)

    la_prop = la_temp
    la_prop.lmp = lmp_temp
    la_prop.log_post = log_post_temp

    acc_rate = min(1.0, np.exp(prod_bal_con - rev_prod_bal_con))

    return {
        "LA_prop": la_prop,
        "JD": JD,
        "acc_rate": acc_rate,
        "thinned_k_size": thinned_k_size,
    }

__all__ = ["update_la_dag"]
