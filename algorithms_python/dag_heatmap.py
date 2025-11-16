"""Convenience plotting utilities for DAG adjacency matrices."""
from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def _prep_dataframe(W: np.ndarray) -> pd.DataFrame:
    p = W.shape[0]
    grid = pd.DataFrame(
        {
            "X": np.repeat(np.arange(1, p + 1), p),
            "Y": np.tile(np.arange(1, p + 1), p),
            "Z": W.T.reshape(-1),
        }
    )
    grid["X"] = grid["X"].astype(str)
    grid["Y"] = grid["Y"].astype(str)
    return grid

def dag_heatmap(W: np.ndarray, text_bound: float = 0.05, low_col: str = "white", high_col: str = "red", true_graph: np.ndarray | None = None):
    """Plot posterior edge probabilities."""

    data = _prep_dataframe(W)
    data["label"] = data["Z"].where(lambda z: np.abs(z) > text_bound).round(2)

    pivot = data.pivot(index="Y", columns="X", values="Z")
    ax = sns.heatmap(pivot, cmap=sns.color_palette([low_col, high_col]), cbar_kws={"label": "PEP"})

    for y, x, val in data[["Y", "X", "label"]].dropna().itertuples(index=False):
        ax.text(int(x) - 0.5, pivot.shape[0] - int(y) + 0.5, f"{val:.2f}", ha="center", va="center", fontsize=6)

    ax.set_xlabel("Children nodes")
    ax.set_ylabel("Parent nodes")

    if true_graph is not None:
        mask = true_graph.T.astype(bool).ravel()
        ys, xs = np.where(true_graph.T)
        ax.scatter(xs + 0.5, pivot.shape[0] - ys - 0.5, facecolors="none", edgecolors="black", linewidths=0.5)

    return ax

def dag_heatmap_cor(W: np.ndarray, text_bound: float = 0.05, low_col: str = "white", high_col: str = "red", true_graph: np.ndarray | None = None):
    data = _prep_dataframe(W)
    data["label"] = data["Z"].where(lambda z: np.abs(z) > text_bound).round(2)
    pivot = data.pivot(index="Y", columns="X", values="Z")
    ax = sns.heatmap(pivot, cmap=sns.color_palette([low_col, high_col]), cbar_kws={"label": "correlation"})
    for y, x, val in data[["Y", "X", "label"]].dropna().itertuples(index=False):
        ax.text(int(x) - 0.5, pivot.shape[0] - int(y) + 0.5, f"{val:.2f}", ha="center", va="center", fontsize=6)
    ax.set_xlabel("X")
    ax.set_ylabel("X")
    if true_graph is not None:
        ys, xs = np.where(true_graph.T)
        ax.scatter(xs + 0.5, pivot.shape[0] - ys - 0.5, facecolors="none", edgecolors="black", linewidths=0.5)
    return ax

def dag_heatmap_true(W: np.ndarray, low_col: str = "white", high_col: str = "red"):
    data = _prep_dataframe(W)
    data["Z"] = data["Z"].map({0: "disconnected", 1: "connected"})
    pivot = data.pivot(index="Y", columns="X", values="Z")
    ax = sns.heatmap(pivot, cmap=sns.color_palette([low_col, high_col]), cbar=False)
    ax.set_xlabel("Children nodes")
    ax.set_ylabel("Parent nodes")
    for (y, x), value in np.ndenumerate(pivot.values):
        ax.text(x + 0.5, pivot.shape[0] - y - 0.5, value, ha="center", va="center", fontsize=6)
    return ax

__all__ = ["dag_heatmap", "dag_heatmap_cor", "dag_heatmap_true"]
