"""Helpers for using the PARNI sampler as a one-step proposal kernel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable

import numpy as np

from .compute_la_dag import compute_la_dag
from .parni import _setup_hyper_par
from .marpip_dag import marpips_dag_h
from .marpip_bge import marpips_bge_h
from .sampling import sample_ind_dag
from .structures import LaplaceApproximation
from .update_la_dag import update_la_dag
from .log_likelihood import log_llh_dag_table
from .log_likelihood_update import log_llh_dag_update_table
from .log_likelihood_bge import log_llh_bge_table, log_llh_bge_update_table

Array = np.ndarray


@dataclass
class ParniContext:
    """Container holding reusable information for single step proposals."""

    hyper_par: Dict[str, Any]
    PIPs: Array
    A: Array
    D: Array
    thinning_rate: float
    proposal_omega: float
    bal_fun: Callable[[float | Array], float | Array]
    rng: np.random.Generator


def _default_bal_fun(x: float | Array) -> float | Array:
    return np.minimum(1.0, x)


def _resolve_score_function(bge_obj: Any) -> Callable[[int, Array], float]:
    if bge_obj is None:
        raise ValueError("A `bge_obj` instance is required when `pips_mode` is 'bge'.")

    candidate_methods: Iterable[str] = (
        "local_score",
        "score_local",
        "local_score_family",
        "local_score_gaussian",
        "calc_local_score",
        "score",
    )
    method = None
    for name in candidate_methods:
        if hasattr(bge_obj, name):
            method = getattr(bge_obj, name)
            break
    if method is None:
        raise AttributeError(
            "Unable to locate a scoring method on the supplied BGe object. "
            "Expected one of: " + ", ".join(candidate_methods)
        )

    def _score(node: int, parents: Array) -> float:
        parents = np.asarray(parents, dtype=int)
        return float(method(node, parents))

    return _score


def parni_prepare_context(
    *,
    X_p_n: Array,
    h: float | Array,
    bge_obj: Any | None = None,
    H: Array | None = None,
    kappa: float = 0.0,
    omega: float = 0.9,
    pips_mode: str = "bge",
    rng: np.random.Generator | None = None,
    score_function: Callable[[int, Array], float] | None = None,
    proposal_omega: float = 0.5,
    bal_fun: Callable[[float | Array], float | Array] | None = None,
) -> ParniContext:
    """Pre-compute the information required by :func:`parni_propose_one_step`.

    Parameters
    ----------
    X_p_n:
        Data matrix arranged as ``(p, n)`` matching the convention used by the
        translated PARNI code.
    h:
        Hyper-parameter controlling the sparsity prior.
    bge_obj:
        Optional BGe scoring object.  Required when ``pips_mode`` is ``"bge"``.
    H:
        Optional reachability matrix describing the admissible parents for each
        node.  When omitted a fully connected graph (excluding self edges) is
        assumed.
    kappa:
        Ridge parameter used when smoothing the marginal PIPs.
    omega:
        Thinning rate used when sub-sampling neighbourhood moves.
    pips_mode:
        One of ``"bge"`` or ``"dag"`` determining which marginal PIP tables are
        pre-computed.
    rng:
        Optional ``numpy.random.Generator`` reused across calls.
    score_function:
        Optional callable overriding the method used to score local families in
        the BGe case.
    proposal_omega:
        Value passed to :func:`update_la_dag` controlling the per-edge move
        probabilities (defaults to the value used in the full sampler).
    bal_fun:
        Balancing function used inside :func:`update_la_dag`.
    """

    rng = rng or np.random.default_rng()
    X_p_n = np.asarray(X_p_n, dtype=float)
    p, n = X_p_n.shape

    if H is None:
        H = np.ones((p, p), dtype=int)
        np.fill_diagonal(H, 0)

    XtX = X_p_n @ X_p_n.T
    hyper_par: Dict[str, Any] = {
        "X": X_p_n,
        "XtX": XtX,
        "p": p,
        "n": n,
        "g": 1.0,
        "h": h,
    }

    if pips_mode.lower() == "bge":
        score_fn = score_function or _resolve_score_function(bge_obj)
        hyper_par["score_function"] = score_fn
        hyper_par["use_bge"] = True
    else:
        hyper_par["use_bge"] = False

    alg_par: Dict[str, Any] = {"H": np.asarray(H, dtype=int)}
    hyper = _setup_hyper_par(alg_par, hyper_par)

    if hyper.get("use_bge", False):
        hyper["log_llh"] = log_llh_bge_table
        hyper["log_llh_update"] = log_llh_bge_update_table
        res = marpips_bge_h(hyper, kappa)
    else:
        hyper["log_llh"] = log_llh_dag_table
        hyper["log_llh_update"] = log_llh_dag_update_table
        res = marpips_dag_h(hyper, kappa)
    hyper["tables"] = res["tables"]

    approx_PIPs = res["PIPs"].copy()
    PIPs = approx_PIPs.copy()
    np.fill_diagonal(PIPs, 0)
    A = np.minimum(PIPs / (1 - PIPs), 1.0)
    D = np.minimum((1 - PIPs) / PIPs, 1.0)
    np.fill_diagonal(A, 0)
    np.fill_diagonal(D, 0)

    context = ParniContext(
        hyper_par=hyper,
        PIPs=PIPs,
        A=A,
        D=D,
        thinning_rate=float(omega),
        proposal_omega=float(proposal_omega),
        bal_fun=bal_fun or _default_bal_fun,
        rng=rng,
    )
    return context


def parni_make_LA_from_G(gamma: Array, context: ParniContext) -> LaplaceApproximation:
    """Compute the Laplace approximation state for ``gamma``."""

    return compute_la_dag(np.asarray(gamma, dtype=int), context.hyper_par)


def parni_propose_one_step(
    la: LaplaceApproximation,
    context: ParniContext,
    rng: np.random.Generator | None = None,
) -> Dict[str, Any]:
    """Perform a single neighbourhood update using the PARNI kernel."""

    rng = rng or context.rng
    curr = la.curr
    eta = (1 - curr) * context.A + curr * context.D
    neighs = sample_ind_dag(True, eta, rng=rng)
    k = neighs.sample
    if k.size == 0:
        return {
            "LA_prop": la,
            "log_q_forward": 0.0,
            "log_q_reverse": 0.0,
            "JD": 0,
            "acc_rate": 1.0,
        }

    updates = update_la_dag(
        la=la,
        k=k,
        hyper_par=context.hyper_par,
        bal_fun=context.bal_fun,
        pips=context.PIPs,
        thinning_rate=context.thinning_rate,
        omega=context.proposal_omega,
        rng=rng,
    )
    la_prop = updates["LA_prop"]
    return {
        "LA_prop": la_prop,
        "log_q_forward": float(updates.get("log_q_forward", 0.0)),
        "log_q_reverse": float(updates.get("log_q_reverse", 0.0)),
        "JD": updates.get("JD", 0),
        "acc_rate": updates.get("acc_rate", 1.0),
    }


__all__ = [
    "ParniContext",
    "parni_prepare_context",
    "parni_make_LA_from_G",
    "parni_propose_one_step",
]
