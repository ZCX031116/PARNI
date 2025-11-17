"""Logistic helper functions with epsilon bounds.

This module mirrors the behaviour of the `logit_e.R` helper in the
original R implementation.  The functions clamp their inputs away from
0 and 1 by a user provided epsilon to avoid numerical issues when the
logistic (or its inverse) is evaluated at the extremes.
"""
from __future__ import annotations

import numpy as np

def logit_eps(x: np.ndarray | float, eps: float) -> np.ndarray | float:
    """Return the logit transform with epsilon clamping.

    Parameters
    ----------
    x:
        Scalar or ``numpy`` array with values in :math:`[0, 1]`.
    eps:
        Small positive number used to keep the argument away from 0 and 1.
    """

    x_arr = np.asarray(x, dtype=float)
    clipped = np.clip(x_arr, 2 * eps, 2 * (1 - eps))
    result = np.log(clipped - eps) - np.log(1 - clipped - eps)
    if np.isscalar(x):
        return float(result)
    return result

def inv_logit_eps(y: np.ndarray | float, eps: float) -> np.ndarray | float:
    """Inverse of :func:`logit_eps`.

    Parameters
    ----------
    y:
        Scalar or array of logit transformed values.
    eps:
        Same epsilon used in :func:`logit_eps`.
    """

    y_arr = np.asarray(y, dtype=float)
    ey = np.exp(-y_arr)
    result = (eps * ey - eps + 1.0) / (ey + 1.0)
    if np.isscalar(y):
        return float(result)
    return result

__all__ = ["logit_eps", "inv_logit_eps"]
