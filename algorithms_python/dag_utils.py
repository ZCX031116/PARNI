"""Utility helpers for working with adjacency matrices of DAGs.

This module bundles several helpers that appear throughout the original
R implementation:

``is_dag_adjmat``
    Check whether a binary adjacency matrix encodes a DAG.
``h_to_permi_pars``
    Convert a reachability matrix ``H`` into per–node parent lists.
``model_encoding`` / ``model_decoding``
    Map between binary inclusion vectors and Gray-code compatible integer
    encodings.  These helpers are used when tabulating local scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

def is_dag_adjmat(W: np.ndarray) -> bool:
    """Return ``True`` iff ``W`` encodes a directed acyclic graph."""

    W = np.asarray(W, dtype=int)
    num_precedent = W.sum(axis=0)
    d = W.shape[0]
    remaining = W.copy()

    while d > 1:
        if np.all(num_precedent > 0):
            return False
        leaves = np.where(num_precedent == 0)[0]
        remaining = np.delete(np.delete(remaining, leaves, axis=0), leaves, axis=1)
        d -= len(leaves)
        if d <= 1:
            break
        num_precedent = remaining.sum(axis=0)
    return True

def h_to_permi_pars(H: np.ndarray) -> List[np.ndarray]:
    """Return per-node parent indices implied by ``H``."""

    H = np.asarray(H, dtype=int)
    parents: List[np.ndarray] = []
    for col in range(H.shape[1]):
        parents.append(np.where(H[:, col] == 1)[0])
    return parents

def convert_to_binary(n: int) -> List[int]:
    if n > 1:
        digits = convert_to_binary(n // 2)
    else:
        digits = []
    digits.append(n % 2)
    return digits

def gray_code(idx: int, vector: Iterable[int]) -> np.ndarray:
    """Return the Gray-code representation of ``idx``.

    The ``vector`` argument provides the template length and is updated in
    place (matching the behaviour of the R helper).
    """

    vector = np.array(list(vector), dtype=int)
    binary = convert_to_binary(idx)
    g = np.array(binary, dtype=int)
    if len(g) > 1:
        for i in range(1, len(binary)):
            g[i] = (binary[i] + binary[i - 1]) % 2
    start = len(vector) - len(g)
    vector[start:] = g
    return vector

def model_encoding(curr_part: np.ndarray) -> int:
    curr_part = np.asarray(curr_part, dtype=int)
    powers = 2 ** np.arange(curr_part.size)
    return int(curr_part.dot(powers) + 1)

def model_decoding(p: int, m: int) -> np.ndarray:
    bits = convert_to_binary(m - 1)
    bits = [0] * (p - len(bits)) + bits
    return np.array(list(reversed(bits)), dtype=int)

__all__ = [
    "is_dag_adjmat",
    "h_to_permi_pars",
    "convert_to_binary",
    "gray_code",
    "model_encoding",
    "model_decoding",
]
