# ===== Standard Library =====
import os
import sys
import math
import random
import shutil
import tempfile
import warnings
import hashlib
from pathlib import Path
from itertools import permutations
from multiprocessing import Manager
from copy import deepcopy

# ===== Third-Party Libraries =====
import numpy as np
import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from adjustText import adjust_text
from collections import defaultdict

# ===== TRUST Framework =====
from trust.utils.bge import BGe
from adjustText import adjust_text
import json
from build_H import build_H
from parni_dag import (
    parni_prepare_context,
    parni_make_LA_from_G,
    parni_step_one,
)
from dataclasses import dataclass

def pairwise_linear_ce(edge_weights):
    """Returns the pairwise causal effect given the matrix of edge weights.

    Args:
        edge_weights (np.array): (d, d) Weights of the linear model

    Returns:
        effects (np.array): (d, d) matrix of pairwise causal effects
    """
    d = edge_weights.shape[0]
    effects = np.linalg.inv(np.eye(d) - edge_weights) 
    return effects

def causal_effect_ij(edge_weights: np.ndarray, i: int, j: int) -> float:
    """
    返回 (I - B)^{-1}[i, j]，只解一列而不做整逆。
    等价于解 A x = e_j，取 x[i]。
    """
    B = np.asarray(edge_weights, dtype=float)
    d = B.shape[0]
    assert B.shape == (d, d)
    assert 0 <= i < d and 0 <= j < d

    A = np.eye(d, dtype=float) - B           # A = I - B
    e_j = np.zeros(d, dtype=float); e_j[j] = 1.0

    # numpy.linalg.solve 通常在内部走 LU 分解，比 inv 稳定
    try:
        x = np.linalg.solve(A, e_j)
    except np.linalg.LinAlgError as e:
        raise RuntimeError("A = I - B 奇异或病态，无法求解") from e

    return float(x[i])
    
def pairwise_linear_ce_no_params(g_samples, data, bge_model, params_per_graph=10,avg=True, return_B=False, R = None):
    """
    Returns the pairwise (linear) causal effect, averaged over the DAG samples.
    """
    if R is None:
        R = bge_model.calc_R(data)
    N, d = data.shape
    B = [[] for _ in range(d)]
    cnt = 0
    for G_sample in g_samples:
        # print(cnt)
        # cnt = cnt+1
        # i is the child
        for i in range(d):
            parents_mask = G_sample[:, i].astype(bool)
            if np.any(parents_mask):
                l = np.sum(parents_mask) + 1
                parents_child_mask = np.copy(parents_mask)
                parents_child_mask[i] = True
                R22 = R[i, i]
                R12 = R[parents_mask, i]
                R21 = R[i, parents_mask]
                R11 = R[parents_mask, :][:, parents_mask]
                loc = np.linalg.inv(R11) @ R12
                deg_free = bge_model.alpha_w + N - d + l
                shape = np.linalg.inv(
                    deg_free /
                    (R22 - R21 @ np.linalg.inv(R11) @ R12
                     ) *
                    R11
                )
                dist = st.multivariate_t(loc=loc, shape=shape, df=deg_free)
                bs = np.expand_dims(dist.rvs(params_per_graph), axis=0) if params_per_graph == 1 else dist.rvs(params_per_graph)
                for b in bs:
                    column = np.zeros(d)
                    column[parents_mask] = b
                    B[i].append(column)
            else:
                for _ in range(params_per_graph):
                    B[i].append(np.zeros(d))
    
    B = np.array(B)  # (d-col, num_total_samples, d-row)
    B = np.swapaxes(np.swapaxes(B, 0, 1), 1, 2)
    effects = [np.linalg.inv(np.eye(d) - B_sample) for B_sample in B]
    avg_effects = np.mean(np.array(effects), axis=0)
    if return_B:
        if avg:
            return B, avg_effects
        else:
            return B, effects
    else:
        if avg:
            return avg_effects
        else:
            return effects

def log_and_print(message, file=None, console_output=True):
    if console_output:
        print(message)
    if file is not None:
        file.write(str(message) + "\n")

def plot_edge_frequency_and_weight_avg_graphs(ce_samples, gs_np, thetas_np, i, j, threshold, save_path, file_name):
    """Plot edge frequency and average-weight graphs for CE[i, j] > threshold."""
    d = gs_np.shape[1]
    edge_count = np.zeros((d, d))
    edge_weight_sum = np.zeros((d, d))

    count = 0  # Number of valid samples

    for G_sample, theta_sample, ce_mat in zip(gs_np, thetas_np, ce_samples):
        if not np.isfinite(ce_mat[i, j]):
            continue
        if ce_mat[i, j] <= threshold:
            continue
        count += 1
        for src in range(d):
            for tgt in range(d):
                if G_sample[src, tgt] == 1:
                    edge_count[src, tgt] += 1
                    edge_weight_sum[src, tgt] += theta_sample[src, tgt]
    with open(file_name, "a") as f:
        log_and_print(f"{count} samples where CE[{i},{j}] > {threshold}", f)

    # Normalize frequency to [0, 1]
    edge_freq = edge_count / count if count > 0 else edge_count

    # Average edge weight where count > 0
    edge_weight_avg = np.zeros_like(edge_weight_sum)
    np.divide(edge_weight_sum, edge_count, out=edge_weight_avg, where=edge_count > 0)

    # Plot Edge Frequency Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(edge_freq, cmap='Reds', interpolation='nearest')
    plt.title(f"Edge Frequency (CE[{i},{j}] > {threshold})")
    plt.colorbar(label='Edge Appearance Frequency')
    plt.xlabel("Target Node")
    plt.ylabel("Source Node")
    plt.savefig(save_path / f"edge_frequency_CE_{i}_{j}_gt_{threshold}.png", dpi=300)
    plt.close()

    # Plot Average Edge Weight Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(edge_weight_avg, cmap='coolwarm', interpolation='nearest')
    plt.title(f"Average Edge Weight (CE[{i},{j}] > {threshold})")
    plt.colorbar(label='Average Edge Weight')
    plt.xlabel("Target Node")
    plt.ylabel("Source Node")
    plt.savefig(save_path / f"edge_weight_avg_CE_{i}_{j}_gt_{threshold}.png", dpi=300)
    plt.close()

class Multilevel:
    def __init__(self, bge_model, data, X, i, j, save_dir, output_file, max_outer_iter,
                    p_edge=0.3, params_per_graph=50, rng=None, structure_kernel = "Structure_MCMC", joint_kernel = False):
        self.bge_model = bge_model
        self.R = bge_model.calc_R(data)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.data = data
        self.X = sorted(X)
        self.i = i
        self.j = j
        self.sigma = 1.0
        self.mu = 0.0
        self.save_dir = save_dir
        self.output_file = output_file
        self.p_edge = p_edge
        self.params_per_graph = params_per_graph
        self.max_outer_iter=max_outer_iter
        self.structure_kernel = structure_kernel
        self.joint_kernel = joint_kernel
        d = data.shape[1]

        if self.structure_kernel == "PARNI":
            print("Preparing PARNI context...")
            X_p_n = self.data.T 
            p = X_p_n.shape[0]
            H = None
            kappa_ws = 0.0   
            self.parni_ctx = parni_prepare_context(
                X_p_n=X_p_n,
                h=0.2,
                bge_obj=self.bge_model,
                H=H,          
                kappa=kappa_ws,     
                omega=0.5,
                pips_mode="bge"  
            )
            
    def log_post_over_weights(self, adjacency_matrix, weight_matrix, bge_model, data):
        log_p = 0.0
        N, d = data.shape
        R = self.R
        alpha_w = bge_model.alpha_w
        alpha_w_prime = alpha_w + N 
        for i in range(d):
            parents = np.where(adjacency_matrix[:, i])[0].tolist()
            l = len(parents) + 1 
            if not parents:
                if np.any(weight_matrix[:, i] != 0):
                    return -np.inf 
                else:
                    continue
            R11 = R[np.ix_(parents, parents)]
            R12 = R[parents, i]
            R21 = R[i, parents]
            R22 = R[i, i]
            try:
                R11_inv = np.linalg.inv(R11)
            except np.linalg.LinAlgError:
                return -np.inf 
            loc = R11_inv @ R12
            denom = R22 - R21 @ R11_inv @ R12
            if denom <= 1e-10:  
                return -np.inf
            deg_free = alpha_w_prime - d + l
            shape_matrix = np.linalg.inv((deg_free / denom) * R11)
            eigenvalues = np.linalg.eigvalsh(shape_matrix)
            b_i = weight_matrix[parents, i]
            try:
                dist = st.multivariate_t(loc=loc, shape=shape_matrix, df=deg_free)
                log_p += dist.logpdf(b_i)
            except np.linalg.LinAlgError:
                return -np.inf
        return log_p

    def log_posterior_with_weights(self,adjacency_matrix, weight_matrix, bge_model, data, p_edge = 0.3, mll_score = None):
        num_edges = np.sum(adjacency_matrix)
        d = adjacency_matrix.shape[0]
        num_possible_edges = d * (d - 1)
        prior = num_edges * np.log(p_edge) + (num_possible_edges - num_edges) * np.log(1 - p_edge)
        if np.isneginf(prior) or np.isnan(prior):
            prior = -100.0
        log_prior_g = prior
        if mll_score is None:
            mll_score = bge_model.mll(adjacency_matrix, data)
        log_post_w = self.log_post_over_weights(adjacency_matrix, weight_matrix, bge_model, data)
        return (log_prior_g + mll_score + log_post_w), log_prior_g, mll_score, log_post_w

    def score_state(self, adjacency_matrix, weight_matrix, level=0, mll_score = None):
        G_copy = np.array([np.copy(adjacency_matrix)])
        pairwise_effect = causal_effect_ij(weight_matrix, self.i, self.j)
        if pairwise_effect < level:
            return -np.inf, pairwise_effect, -np.inf, -np.inf, -np.inf
        log_score_val,log_prior_g, mll_score,log_post_w = self.log_posterior_with_weights(
            adjacency_matrix, weight_matrix, 
            bge_model=self.bge_model,
            data=self.data,
            p_edge=self.p_edge,
            mll_score = mll_score
        )
        return log_score_val, pairwise_effect,log_post_w, log_prior_g, mll_score

    def initialize_edge_weight_matrix(self, adjacency_matrix):
        G_batch = np.array(adjacency_matrix)
        Bs, d_care = pairwise_linear_ce_no_params(G_batch, self.data, self.bge_model,
                                          params_per_graph=1,
                                          avg=False, return_B=True)
        return Bs

    @staticmethod
    def check_acyclic(adj_matrix):
        G = nx.DiGraph(adj_matrix)
        return nx.is_directed_acyclic_graph(G)
    
    @staticmethod
    def build_reachability(adj: np.ndarray) -> np.ndarray:
        """Floyd–Warshall transitive closure, returns a boolean matrix reach[i, j] indicating if i→j is reachable."""
        reach = adj.astype(bool).copy()
        d = adj.shape[0]
        for k in range(d):
            reach |= np.outer(reach[:, k], reach[k, :])
        return reach

    @staticmethod
    def propose_new_structure(adj: np.ndarray,
                            w:   np.ndarray,
                            sigma: float,
                            mu:    float):
        def _norm_logpdf(x, mu, sigma):
            var = float(sigma) * float(sigma)
            return -0.5*np.log(2.0*np.pi*var) - 0.5*((x - mu)*(x - mu))/var
        d       = adj.shape[0]
        reach   = Multilevel.build_reachability(adj)
        edges   = list(zip(*np.where(adj == 1)))
        new_adj = deepcopy(adj)
        new_w   = deepcopy(w)
        del_list = edges
        add_list = [(i, j) for i in range(d) for j in range(d)
                    if i != j and new_adj[i, j] == 0 and not reach[j, i]]
        rev_list = []
        for (u, v) in edges:
            if new_adj[v, u] == 1:
                continue
            new_adj[u, v] = 0
            reach_wo = Multilevel.build_reachability(new_adj)
            if not reach_wo[u, v]:
                rev_list.append((u, v))
            new_adj[u, v] = 1
        op_types = []
        if del_list: op_types.append(0)
        if add_list: op_types.append(1)
        if rev_list: op_types.append(2)
        if not op_types:
            return adj, w, 1.0, 1.0
        op_type = random.choice(op_types)
        t_fwd   = len(op_types)
        # ---- remove ---------------------------------------------------------------
        if op_type == 0:
            num_del = len(del_list)
            u, v = random.choice(del_list)
            old_w = w[u, v]
            new_adj[u, v] = 0
            new_w[u, v]   = 0.0
            reach_new = Multilevel.build_reachability(new_adj)
            add_after = [(i, j) for i in range(d) for j in range(d)
                        if i != j and new_adj[i, j] == 0 and not reach_new[j, i]]
            num_add_after = len(add_after)
            def count_reversible_edges(adj_):
                cnt = 0
                for x, y in zip(*np.where(adj_ == 1)):
                    if adj_[y, x] == 1:
                        continue
                    adj_[x, y] = 0
                    reach_wo = Multilevel.build_reachability(adj_)
                    if not reach_wo[x, y]:
                        cnt += 1
                    adj_[x, y] = 1
                return cnt
            num_rev_after = count_reversible_edges(new_adj)
            t_rev = (1 if (new_adj==1).any() else 0) \
                + (1 if num_add_after>0 else 0) \
                + (1 if num_rev_after>0 else 0)
            log_q_fwd = -np.log(t_fwd) - np.log(num_del)
            log_q_rev = -np.log(max(t_rev,1)) \
                        -np.log(max(num_add_after,1)) \
                        + _norm_logpdf(old_w, mu, sigma)
        # ---- add ---------------------------------------------------------------
        elif op_type == 1:
            num_add = len(add_list)
            u, v = random.choice(add_list)
            drawn = np.random.normal(loc=mu, scale=sigma)
            new_adj[u, v] = 1
            new_w[u, v]   = drawn
            num_del_after = int((new_adj==1).sum())
            reach_new     = Multilevel.build_reachability(new_adj)
            add_after     = [(i, j) for i in range(d) for j in range(d)
                            if i != j and new_adj[i, j] == 0 and not reach_new[j, i]]
            def count_reversible_edges(adj_):
                cnt = 0
                for x, y in zip(*np.where(adj_ == 1)):
                    if adj_[y, x] == 1:
                        continue
                    adj_[x, y] = 0
                    reach_wo = Multilevel.build_reachability(adj_)
                    if not reach_wo[x, y]:
                        cnt += 1
                    adj_[x, y] = 1
                return cnt
            num_rev_after = count_reversible_edges(new_adj)
            t_rev = (1 if num_del_after>0 else 0) \
                + (1 if len(add_after)>0 else 0) \
                + (1 if num_rev_after>0 else 0)
            log_q_fwd = -np.log(t_fwd) - np.log(num_add) + _norm_logpdf(drawn, mu, sigma)
            log_q_rev = -np.log(max(t_rev,1)) - np.log(max(num_del_after,1))
        # ---- reverse -----------------------------------------------------------
        else:
            num_rev = len(rev_list)
            u, v = random.choice(rev_list)
            new_adj[u, v] = 0
            new_adj[v, u] = 1
            new_w[v, u]   = new_w[u, v]
            new_w[u, v]   = 0.0
            def count_reversible_edges(adj_):
                cnt = 0
                for x, y in zip(*np.where(adj_ == 1)):
                    if adj_[y, x] == 1:
                        continue
                    adj_[x, y] = 0
                    reach_wo = Multilevel.build_reachability(adj_)
                    if not reach_wo[x, y]:
                        cnt += 1
                    adj_[x, y] = 1
                return cnt
            num_rev_after = count_reversible_edges(new_adj)
            del_after = list(zip(*np.where(new_adj == 1)))
            reach_new = Multilevel.build_reachability(new_adj)
            add_after = [(i, j) for i in range(d) for j in range(d)
                        if i != j and new_adj[i, j] == 0 and not reach_new[j, i]]
            t_rev = (1 if len(del_after)>0 else 0) \
                + (1 if len(add_after)>0 else 0) \
                + (1 if num_rev_after>0 else 0)
            log_q_fwd = -np.log(t_fwd) - np.log(num_rev)
            log_q_rev = -np.log(max(t_rev,1)) - np.log(max(num_rev_after,1))

        return new_adj, new_w, log_q_fwd, log_q_rev
 
    @staticmethod 
    def propose_new_weights(adjacency_matrix, weight_matrix, edges, step_size=0.8, rng=None): 
        if rng is None: 
            rng = np.random.default_rng() 
        new_adj = deepcopy(adjacency_matrix) 
        new_w = deepcopy(weight_matrix) 
        if len(edges) == 0: 
            return new_adj, new_w, 0.0, 0.0
        u, v = rng.choice(edges)
        b_old = float(new_w[u, v])
        sigma_fwd = float(step_size * max(1.0, abs(b_old))) 
        if not np.isfinite(sigma_fwd) or sigma_fwd <= 0.0:
            return new_adj, new_w, -np.inf, 0.0
        eps  = rng.normal(scale=sigma_fwd)
        b_new = b_old + eps 
        new_w[u, v] = b_new 
        sigma_rev = float(step_size * max(1.0, abs(b_new))) 
        if not np.isfinite(sigma_rev) or sigma_rev <= 0.0: 
            return new_adj, new_w, p_prop, -np.inf
        LOG2PI = math.log(2.0 * math.pi)
        def _norm_logpdf(x, mu, sigma):
            if not np.isfinite(sigma) or sigma <= 0.0:
                return -np.inf
            z = (x - mu) / sigma
            if not np.isfinite(z):
                return -np.inf
            if abs(z) > math.sqrt(np.finfo(float).max):
                return -np.inf
            return -0.5 * z * z - 0.5 * LOG2PI - math.log(sigma)
        p_prop = _norm_logpdf(b_new, b_old, sigma_fwd) # q(b' | b) 
        p_cur = _norm_logpdf(b_old, b_new, sigma_rev) # q(b | b') 
        return new_adj, new_w, p_prop, p_cur
    
    @staticmethod 
    def _edge_delta(G: np.ndarray, Gp: np.ndarray):
        Aset, Rset, Kset = set(), set(), set()
        U, V = G.shape
        for u in range(U):
            for v in range(V):
                if u == v:
                    continue
                gu, gpu = int(G[u, v]), int(Gp[u, v])
                if gu == 0 and gpu == 1:
                    Aset.add((u, v))
                elif gu == 1 and gpu == 0:
                    Rset.add((u, v))
                elif gu == 1 and gpu == 1:
                    Kset.add((u, v))
        return Aset, Rset, Kset

    def propose_new_state_Structure_MCMC(self, adjacency_matrix, weight_matrix, p_structure=0.5, step_size=1):
        """
        With probability p_structure, perform a structure perturbation; otherwise, perform a weight perturbation.
        """
        if np.random.rand() < p_structure:
            new_adj,new_w, p_prop, p_cur = Multilevel.propose_new_structure(adjacency_matrix, weight_matrix, self.sigma, self.mu)
            return new_adj, new_w, p_prop, p_cur, "structure"
        else:
            edges = list(zip(*np.where(adjacency_matrix == 1)))
            new_adj, new_w, p_prop, p_cur = Multilevel.propose_new_weights(adjacency_matrix, weight_matrix,edges,  step_size)
            return new_adj, new_w, p_prop, p_cur, "weight"
    
    def propose_new_state_PARNI(self, LA, weight_matrix, step_size=1):
        def _norm_logpdf(x, mu, sigma):
            var = float(sigma) * float(sigma)
            return -0.5*np.log(2.0*np.pi*var) - 0.5*((x - mu)*(x - mu))/var
        p_structure = 0.5
        if np.random.rand() < p_structure:
            current_adj = LA.curr.astype(int)
            info = parni_step_one(LA, self.parni_ctx, rng=self.rng, proposal_only=True)
            LA_prop    = info["LA_prop"]
            new_adj    = LA_prop.curr.astype(int)
            LA_prop_llh = float(LA_prop.llh)
            log_qG_fwd = info["log_qG_fwd"]
            log_qG_rev = info["log_qG_rev"]
            A, R, K = Multilevel._edge_delta(current_adj, new_adj)
            new_w = np.copy(weight_matrix)
            log_qw_fwd = 0.0
            log_qw_rev = 0.0
            for e in A:
                u, v = e
                b_sample = self.rng.normal(loc=self.mu, scale=self.sigma)
                new_w[u, v] = b_sample
                log_qw_fwd += _norm_logpdf(b_sample, self.mu, self.sigma)
            for e in R:
                u, v = e
                old_w = weight_matrix[u, v]
                new_w[u, v] = 0.0
                log_qw_rev += _norm_logpdf(old_w, self.mu, self.sigma)
            logp_prop = log_qG_fwd + log_qw_fwd
            logp_cur  = log_qG_rev + log_qw_rev
            return new_adj, new_w, logp_prop, logp_cur, LA_prop, "structure"
        else:
            current_adj = LA.curr.astype(int)
            new_adj = current_adj
            edges = list(zip(*np.where(new_adj == 1)))
            new_w = np.copy(weight_matrix)
            LA_prop = LA
            if len(edges) == 0: 
                return new_adj, new_w, 0.0, 0.0, LA_prop, "weight"
            new_adj, new_w, log_qk_fwd, log_qk_rev = Multilevel.propose_new_weights(
                current_adj, new_w, edges=edges, step_size=step_size, rng=rng
            )
            logp_prop = log_qk_fwd
            logp_cur = log_qk_rev
            
            return new_adj, new_w, logp_prop, logp_cur, LA_prop, "weight"

    def propose_new_state_Joint_Structure_MCMC(self, adjacency_matrix, weight_matrix, p_structure=0.5, step_size=1):
        new_adj, new_w, log_qG_fwd, log_qG_rev = Multilevel.propose_new_structure(
            adjacency_matrix, weight_matrix, self.sigma, self.mu
        )
        A, R, K = Multilevel._edge_delta(adjacency_matrix, new_adj)
        K_list = list(K)
        m_edges = len(K_list)
        if m_edges > 0:
            new_adj, new_w, log_qk_fwd, log_qk_rev = Multilevel.propose_new_weights(
                new_adj, new_w, edges=K_list, step_size=step_size, rng=rng
            )
        else:
            log_qk_fwd = log_qk_rev = 0.0
        logp_prop = log_qG_fwd + log_qk_fwd
        logp_cur  = log_qG_rev + log_qk_rev
        return new_adj, new_w, logp_prop, logp_cur

    def propose_new_state_Joint_PARNI(self, LA, weight_matrix, step_size=1):
        def _norm_logpdf(x, mu, sigma):
            var = float(sigma) * float(sigma)
            return -0.5*np.log(2.0*np.pi*var) - 0.5*((x - mu)*(x - mu))/var
        current_adj = LA.curr.astype(int)
        info = parni_step_one(LA, self.parni_ctx, rng=self.rng, proposal_only=True)
        LA_prop    = info["LA_prop"]
        new_adj    = LA_prop.curr.astype(int)
        LA_prop_llh = float(LA_prop.llh)
        log_qG_fwd = info["log_qG_fwd"]
        log_qG_rev = info["log_qG_rev"]
        A, R, K = Multilevel._edge_delta(current_adj, new_adj)
        new_w = np.copy(weight_matrix)
        log_qw_fwd = 0.0
        log_qw_rev = 0.0
        for e in A:
            u, v = e
            b_sample = self.rng.normal(loc=self.mu, scale=self.sigma)
            new_w[u, v] = b_sample
            log_qw_fwd += _norm_logpdf(b_sample, self.mu, self.sigma)
        for e in R:
            u, v = e
            old_w = weight_matrix[u, v]
            new_w[u, v] = 0.0
            log_qw_rev += _norm_logpdf(old_w, self.mu, self.sigma)
        K_list = list(K)
        m_edges = len(K_list)
        if m_edges > 0:
            new_adj, new_w, log_qk_fwd, log_qk_rev = Multilevel.propose_new_weights(
                new_adj, new_w, edges=K_list, step_size=step_size, rng=rng
            )
        else:
            log_qk_fwd = log_qk_rev = 0.0
        logp_prop = log_qG_fwd + log_qw_fwd + log_qk_fwd
        logp_cur  = log_qG_rev + log_qw_rev + log_qk_rev
        return new_adj, new_w, logp_prop, logp_cur, LA_prop

    def mcmc_sampling(self, 
                      initial_adj, initial_w, 
                      iterations=5000, burn_in=500, level=0):
        acceptance_ratios = []
        current_adj = initial_adj
        current_w   = initial_w
        current_log_score, current_ace,log_post_w, log_prior_g, mll_score = self.score_state(current_adj, current_w, level=level)
        p_G = log_prior_g
        p_X_G = mll_score
        p_B_G_X = log_post_w
        step_size = 0.8
        ACC = 0
        ACC_weight = 0
        ACC_structure = 0
        weight_moves = 0
        structure_moves = 0
        if self.structure_kernel == "PARNI":
            LA = parni_make_LA_from_G(current_adj.astype(int), self.parni_ctx)

        for it in range(iterations):
            # 1) propose  
            target_acc = 0.3
            if it % 100 == 0 and it > burn_in:
                cur_acc = np.mean(acceptance_ratios[-100:])
                if cur_acc < target_acc / 2:     step_size *= 0.8
                elif cur_acc > target_acc * 2:   step_size *= 1.2
            if self.structure_kernel == "Structure_MCMC":
                if self.joint_kernel:
                    new_adj, new_w, logp_prop, logp_cur = self.propose_new_state_Joint_Structure_MCMC(
                        current_adj, current_w, p_structure=0.5, step_size=step_size
                    )
                else:   
                    new_adj, new_w, logp_prop, logp_cur, move = self.propose_new_state_Structure_MCMC(
                        current_adj, current_w, p_structure=0.5, step_size=step_size
                    )
                pairwise_effect  = causal_effect_ij(new_w, self.i, self.j)
                if Multilevel.check_acyclic(new_adj) == False:
                    print("Proposed graph is cyclic!")
                if pairwise_effect < level:
                    log_acceptance_ratio = -np.inf
                    acceptance_ratio = 0.0
                else:
                    proposed_log_score, proposed_ace,log_post_w, log_prior_g, mll_score = self.score_state(new_adj, new_w, level=level)
                    if np.isinf(proposed_log_score):
                        log_acceptance_ratio = -np.inf
                        acceptance_ratio = 0.0
                    else:
                        log_acceptance_ratio = (proposed_log_score + logp_cur) - (current_log_score + logp_prop)
                        if log_acceptance_ratio >= 0:
                            acceptance_ratio = 1.0
                        else:
                            acceptance_ratio = np.exp(log_acceptance_ratio)
                if np.random.rand() < acceptance_ratio:
                    current_adj = new_adj
                    current_w   = new_w
                    current_log_score = proposed_log_score
                    current_ace = proposed_ace
                    p_G = log_prior_g
                    p_X_G = mll_score
                    p_B_G_X = log_post_w
                    ACC += 1
                    if move == "structure":
                        ACC_structure += 1 
                    if move == "weight":
                        ACC_weight += 1
                    if move == "weight":
                        weight_moves += 1
                    if move == "structure":
                        structure_moves += 1
                else:
                    if move == "weight":
                        weight_moves += 1
                    if move == "structure":
                        structure_moves += 1
                acceptance_ratios.append(acceptance_ratio)

            elif self.structure_kernel == "PARNI":
                LA_prev = LA
                if self.joint_kernel:
                    new_adj, new_w, logp_prop, logp_cur, LA_prop = self.propose_new_state_Joint_PARNI(
                        LA, current_w, step_size=step_size
                    )
                else:
                    new_adj, new_w, logp_prop, logp_cur, LA_prop,move = self.propose_new_state_PARNI(
                        LA, current_w, step_size=step_size
                    )
                LA_prop_llh = float(LA_prop.llh)
                pairwise_effect  = causal_effect_ij(new_w, self.i, self.j)
                if Multilevel.check_acyclic(new_adj) == False:
                    print("Proposed graph is cyclic!")
                if pairwise_effect < level:
                    log_acceptance_ratio = -np.inf
                    acceptance_ratio = 0.0
                    LA = LA_prev
                else:
                    proposed_log_score, proposed_ace,log_post_w, log_prior_g, mll_score = self.score_state(
                        new_adj, new_w, level=level, mll_score = LA_prop_llh
                    )
                    if np.isinf(proposed_log_score):
                        log_acceptance_ratio = -np.inf
                        acceptance_ratio = 0.0
                    else:
                        log_acceptance_ratio = (proposed_log_score + logp_cur) - (current_log_score + logp_prop)
                        if log_acceptance_ratio >= 0:
                            acceptance_ratio = 1.0
                        else:
                            acceptance_ratio = np.exp(log_acceptance_ratio)
                if np.random.rand() < acceptance_ratio:
                    current_adj = new_adj
                    current_w   = new_w
                    current_log_score = proposed_log_score
                    current_ace = proposed_ace
                    p_G = log_prior_g
                    p_X_G = mll_score
                    p_B_G_X = log_post_w
                    LA = LA_prop
                    ACC += 1
                    if move == "structure":
                        ACC_structure += 1 
                    if move == "weight":
                        ACC_weight += 1
                    if move == "weight":
                        weight_moves += 1
                    if move == "structure":
                        structure_moves += 1
                else:
                    if move == "weight":
                        weight_moves += 1
                    if move == "structure":
                        structure_moves += 1
                    acceptance_ratios.append(acceptance_ratio)

        # mean_acc_ratio = np.mean(acceptance_ratios[burn_in:]) if len(acceptance_ratios[burn_in:]) > 0 else 0.0
        acc_rate = ACC / iterations
        # print(structure_moves, weight_moves, structure_moves + weight_moves, iterations)
        # print("------------------")
        return current_adj, current_w, acc_rate, ACC_structure/structure_moves if structure_moves > 0 else 0, ACC_weight/weight_moves if weight_moves > 0 else 0

    def compute_sample_graph_parallel(self, G, W, mcmc_iterations, level):
        """
        When called in parallel, perform MCMC on the given initial structure G,
        and return the adjacency and weight matrices of the last sample.
        """
        d = G.shape[0]
        return self.mcmc_sampling(
            initial_adj=G,
            initial_w=W,
            iterations=mcmc_iterations,
            burn_in=int(mcmc_iterations*0.1),
            level=level,
        )

    def draw_graphs_in_grid(self, G_samples, ace_values, iteration, level, n_samples=9):
        selected_indices = random.sample(range(len(G_samples)), min(n_samples, len(G_samples)))
        selected_graphs = [G_samples[i] for i in selected_indices]
        selected_ace_values = [ace_values[i] for i in selected_indices]
        fig, axes = plt.subplots(3, 3, figsize=(36, 36))
        axes = axes.flatten()
        for idx, (G, ace, ax) in enumerate(zip(selected_graphs, selected_ace_values, axes)):
            graph = nx.from_numpy_array(G, create_using=nx.DiGraph())
            is_acyclic = nx.is_directed_acyclic_graph(graph)
            title = f"ACE: {ace:.4f}"
            if not is_acyclic:
                title += " (Contains Cycle)"
            pos = nx.kamada_kawai_layout(graph)
            nx.draw(graph, pos=pos, ax=ax, with_labels=True, node_color='lightblue',
                    font_size=10, node_size=500, arrows=True)
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        plt.suptitle(f"Graphs at Iteration {iteration} (Level {level})", fontsize=16)
        output_file = f"{self.save_dir}/graphs_iteration_{iteration}_level_{level}.png"
        plt.savefig(output_file)
        plt.close()
        with open(self.output_file, "a") as f:
            log_and_print(f"Graphs saved to {output_file}", f)

    @staticmethod
    def random_dag_topo(num_nodes, p = 0.3):
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        nodes = list(range(num_nodes))
        random.shuffle(nodes)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if random.random() < p:
                    G.add_edge(nodes[i], nodes[j])
        return G
       
    @staticmethod
    def sample_random_graphs(n, d, p = 0.3,
                            size_switch = 24):
        gen_one = Multilevel.random_dag_topo     
        graphs = [nx.to_numpy_array(gen_one(d, p)) for _ in range(n)]
        return graphs

    def calculate_probability(self, n, mcmc_iterations=5000):
        """
        Multilevel (or adaptive) filtering. This only shows the same overall logic as the original,
        with the core difference in compute_sample_graph_parallel.
        Here, we sample (adj, w), and can still use the i->j causal effect for layered filtering.
        """
        with open(self.output_file, "a") as f:
            log_and_print("Starting adaptive leveling...",f)
        probability_list = []
        S = 0  
        iteration = 0
        idx_target = 0
        current_level = -np.inf
        last_level = -np.inf
        num_target = len(self.X)
        suffix = f"(n={n})"
        G_samples = self.sample_random_graphs(n, self.data.shape[1], p=self.p_edge)
        W_samples = list(self.initialize_edge_weight_matrix(G_samples))
        while True:
            if (self.max_outer_iter is not None) and (iteration >= self.max_outer_iter):
                remaining = len(self.X) - idx_target
                probability_list.extend([-50] * remaining)
                print("reach max_outer_iter, break")
                return probability_list
            with open(self.output_file, "a") as f:
                log_and_print(f"Current level: {current_level}",f)
            iteration += 1
            results = []
            with tqdm_joblib(tqdm(desc="Parallel ACE Computation", total=n)) as progress_bar:
                results = Parallel(n_jobs=-1)(
                    delayed(self.compute_sample_graph_parallel)(G, W, mcmc_iterations, current_level)
                    for G,W in zip(G_samples, W_samples)
                )
            index = 0
            tmp = 0
            mark = True
            Number = int(0.1 * n)
            new_adjs = [r[0] for r in results]
            new_ws = [r[1] for r in results]
            acc_rates = [r[2] for r in results]
            acc_structure_rates = [r[3] for r in results]
            acc_weight_rates = [r[4] for r in results]
            with open(self.output_file, "a") as f:
                log_and_print(f"Average acceptance rate in this iteration: {np.mean(acc_rates)}", f)
                log_and_print(f"Average structure acceptance rate in this iteration: {np.mean(acc_structure_rates)}", f)
                log_and_print(f"Average weight acceptance rate in this iteration: {np.mean(acc_weight_rates)}", f)
            Graph_pair_with_ace = []
            ce_matrices = [pairwise_linear_ce(w) for w in new_ws]
            for adj, w, ce_matrix in zip(new_adjs, new_ws, ce_matrices):
                approx_avg_effect = ce_matrix[self.i, self.j]
                Graph_pair_with_ace.append([approx_avg_effect, adj, ce_matrix, w])
            for g_adj in new_adjs:
                if not Multilevel.check_acyclic(g_adj):
                    with open(self.output_file, "a") as f:
                        log_and_print(f"Sampled cyclic graph detected",f)
                    mark = False
            if mark:
                with open(self.output_file, "a") as f:
                    log_and_print("All graphs are acyclic", f)
            ace_list = [x[0] for x in Graph_pair_with_ace]
            #self.draw_graphs_in_grid([x[1] for x in Graph_pair_with_ace], ace_list,iteration, current_level)
            sorted_graph_pair_with_ace = sorted(Graph_pair_with_ace, key=lambda x: x[0], reverse=True)
            percentile_90 = sorted_graph_pair_with_ace[Number][0]
            with open(self.output_file, "a") as f:
                log_and_print(f"ace_list: max: {max(ace_list)}, min: {min(ace_list)}, 90th Percentile of ACE values: {percentile_90}, it's the {Number}th of sorted ace_values", f)
            last_level = current_level
            current_level = percentile_90
            #-----------------------
            bins = 30
            plt.figure(figsize=(10, 6))
            plt.hist(ace_list, bins=bins, color='blue', edgecolor='black', alpha=0.7)
            plt.axvline(x=current_level, color='red', linestyle='--', linewidth=2, label=f'Current_level: {current_level}')
            for target_value in self.X[idx_target:]:
                plt.axvline(x=target_value, color='green', linestyle='--', linewidth=2, label=f'Target ACE: {target_value}')
            plt.axvline(x=last_level, color='yellow', linestyle='--', linewidth=2, label=f'Last_level: {last_level}')
            plt.title('Distribution of ACE Values (Adaptive)')
            plt.xlabel('ACE Value')
            plt.ylabel('Frequency')
            plt.legend()
            filename = f'{self.save_dir}/adaptive_ace_values_{suffix}(level={current_level}).png'
            plt.savefig(filename)
            plt.close()
            #-----------------------
            proportion = sum(1 for item in sorted_graph_pair_with_ace if item[0] >= current_level) / len(sorted_graph_pair_with_ace)
            if proportion == 0:
                return 0
            new_G_samples = [item[1] for item in sorted_graph_pair_with_ace if item[0] >= current_level]
            new_W_samples = [item[3] for item in sorted_graph_pair_with_ace if item[0] >= current_level]
            with open(self.output_file, "a") as f:
                log_and_print(f"90th Percentile {current_level} proportion: {proportion}", f)
            #-----------------------
            last_valid_idx = None
            for i, target_value in enumerate(self.X[idx_target:], start=idx_target):
                if current_level >= target_value:
                    last_valid_idx = i
                    final_proportion = sum(1 for item in sorted_graph_pair_with_ace if item[0] > target_value) / n
                    with open(self.output_file, "a") as f:
                        log_and_print(f"90th Percentile {current_level} exceeds target_ACE {target_value}, final proportion={final_proportion}", f)
                    tmp = 0 if final_proportion == 0 else S + np.log(final_proportion)
                    probability_list.append(tmp)
                    graphs = [item[1] for item in sorted_graph_pair_with_ace if item[0] > target_value]
                    ce_matrices = [item[2] for item in sorted_graph_pair_with_ace if item[0] > target_value]
                    weights = [item[3] for item in sorted_graph_pair_with_ace if item[0] > target_value]
                    if len(ce_matrices) > 0:
                        ce_samples = np.stack(ce_matrices)
                        gs_np = np.stack(graphs)
                        thetas_np = np.stack(weights)
                        save_path = self.save_dir / f"CE_gt_{target_value:.4f}"
                        save_path.mkdir(parents=True, exist_ok=True)
                        plot_edge_frequency_and_weight_avg_graphs(
                            ce_samples=ce_samples,
                            gs_np=gs_np,
                            thetas_np=thetas_np,
                            i=self.i,
                            j=self.j,
                            threshold=target_value,
                            save_path=save_path,
                            file_name=self.output_file
                        )
            if last_valid_idx is not None:
                idx_target = last_valid_idx+1 
                if idx_target == len(self.X):
                    np.save(self.save_dir / f"samples_graphs_level_{current_level}.npy", new_G_samples)
                    np.save(self.save_dir / f"samples_weights_level_{current_level}.npy", new_W_samples)
                    return probability_list
            #-----------------------
            S += np.log(proportion)
            new_samples = list(zip(new_G_samples, new_W_samples))
            while len(new_samples) < n:
                new_samples.append(random.choice(new_samples))
            G_samples, W_samples = zip(*new_samples)
            G_samples = list(G_samples)
            W_samples = list(W_samples)
        return np.exp(S)

    def calculate_level_probability(self, n, mcmc_iterations=5000, num_levels=6):
        """
        Multilevel (or adaptive) filtering. This only shows the same overall logic as the original,
        with the core difference in compute_sample_graph_parallel.
        Here, we sample (adj, w), and can still use the i->j causal effect for layered filtering.
        """
        with open(self.output_file, "a") as f:
            log_and_print("Starting adaptive leveling...",f)
        probability_list = []
        level_list = []
        S = 0  
        iteration = 0
        idx_target = 0
        current_level = -np.inf
        last_level = -np.inf
        num_target = len(self.X)
        suffix = f"(n={n})"
        G_samples = self.sample_random_graphs(n, self.data.shape[1], p=self.p_edge)
        W_samples = list(self.initialize_edge_weight_matrix(G_samples))

        for _ in range(num_levels):
            with open(self.output_file, "a") as f:
                log_and_print(f"Current level: {current_level}",f)
            iteration += 1
            results = []
            with tqdm_joblib(tqdm(desc="Parallel ACE Computation", total=n)) as progress_bar:
                results = Parallel(n_jobs=-1)(
                    delayed(self.compute_sample_graph_parallel)(G, W, mcmc_iterations, current_level)
                    for G,W in zip(G_samples, W_samples)
                )
            index = 0
            tmp = 0
            mark = True
            Number = int(0.1 * n)
            new_adjs = [r[0] for r in results]
            new_ws = [r[1] for r in results]
            Graph_pair_with_ace = []
            ce_matrices = [pairwise_linear_ce(w) for w in new_ws]
            for adj, w, ce_matrix in zip(new_adjs, new_ws, ce_matrices):
                approx_avg_effect = ce_matrix[self.i, self.j]
                Graph_pair_with_ace.append([approx_avg_effect, adj, ce_matrix, w])
            for g_adj in new_adjs:
                if not Multilevel.check_acyclic(g_adj):
                    with open(self.output_file, "a") as f:
                        log_and_print(f"Sampled cyclic graph detected",f)
                    mark = False
            if mark:
                with open(self.output_file, "a") as f:
                    log_and_print("All graphs are acyclic", f)
            ace_list = [x[0] for x in Graph_pair_with_ace]
            #self.draw_graphs_in_grid([x[1] for x in Graph_pair_with_ace], ace_list,iteration, current_level)
            sorted_graph_pair_with_ace = sorted(Graph_pair_with_ace, key=lambda x: x[0], reverse=True)
            percentile_90 = sorted_graph_pair_with_ace[Number][0]
            with open(self.output_file, "a") as f:
                log_and_print(f"ace_list: max: {max(ace_list)}, min: {min(ace_list)}, 90th Percentile of ACE values: {percentile_90}, it's the {Number}th of sorted ace_values", f)
            last_level = current_level
            current_level = percentile_90
            #-----------------------
            bins = 30
            plt.figure(figsize=(10, 6))
            plt.hist(ace_list, bins=bins, color='blue', edgecolor='black', alpha=0.7)
            plt.axvline(x=current_level, color='red', linestyle='--', linewidth=2, label=f'Current_level: {current_level}')
            for target_value in self.X[idx_target:]:
                plt.axvline(x=target_value, color='green', linestyle='--', linewidth=2, label=f'Target ACE: {target_value}')
            plt.axvline(x=last_level, color='yellow', linestyle='--', linewidth=2, label=f'Last_level: {last_level}')
            plt.title('Distribution of ACE Values (Adaptive)')
            plt.xlabel('ACE Value')
            plt.ylabel('Frequency')
            plt.legend()
            filename = f'{self.save_dir}/adaptive_ace_values_{suffix}(level={current_level}).png'
            plt.savefig(filename)
            plt.close()
            #-----------------------
            proportion = sum(1 for item in sorted_graph_pair_with_ace if item[0] >= current_level) / len(sorted_graph_pair_with_ace)
            if proportion == 0:
                return 0
            new_G_samples = [item[1] for item in sorted_graph_pair_with_ace if item[0] >= current_level]
            new_W_samples = [item[3] for item in sorted_graph_pair_with_ace if item[0] >= current_level]
            with open(self.output_file, "a") as f:
                log_and_print(f"90th Percentile {current_level} proportion: {proportion}", f)
            S += np.log(proportion)
            new_samples = list(zip(new_G_samples, new_W_samples))
            while len(new_samples) < n:
                new_samples.append(random.choice(new_samples))
            G_samples, W_samples = zip(*new_samples)
            G_samples = list(G_samples)
            W_samples = list(W_samples)
            level_list.append(current_level)
            probability_list.append(S)
            print(current_level, np.exp(S))
        return level_list, probability_list

if __name__ == "__main__":
    train_size = 1000
    test_size = 1000
    # structure_kernel = "PARNI"
    # joint_kernel = False
    # [("Structure_MCMC", False), ("Structure_MCMC", True), ("PARNI", False), ("PARNI", True)]
    # for structure_kernel, joint_kernel in [("PARNI", False)]:

    for structure_kernel, joint_kernel in [ ("PARNI", False)]:   
        for d in [4]:
            results = []
            sub_dir = f"{structure_kernel}_joint/d={d}(3)" if joint_kernel else f"{structure_kernel}/d={d}(3)"
            base_dir = "PARNI-Structure_MCMC_test/" + sub_dir
            n = 200
            mcmc_iterations = 2000
            levels = [0,1] 
            average_probability_array = []
            for run_num in range(1,6):
                result = []
                for num in range(1,2):
                    for size in [1000]:
                        save_dir = base_dir+f"/case_{num}/run_{run_num}"
                        seed = random.randint(0, 10000)
                        print("seed:",seed)
                        load_dir = f"test_data/d={d}/case{num}"
                        G = np.load(f"{load_dir}/G_{d}Nodes_train_size_1000.npy")
                        B = np.load(f"{load_dir}/B_{d}Nodes_train_size_1000.npy")
                        X_train = np.load(f"{load_dir}/train_{d}Nodes_train_size_1000.npy")
                        X_test = np.load(f"{load_dir}/test_{d}Nodes_train_size_1000.npy")
                        print(X_train.shape)
                        target = np.load(f"test_data/d={d}/target(multi).npy")
                        target_value_list = np.load(f"test_data/d={d}/target_value(multi).npy")
                        # target = np.load(f"test_data/d={d}/target.npy")
                        # target_value_list = np.load(f"test_data/d={d}/target_value.npy")
                        target_value_list = target_value_list[num-1]
                        target = target[num-1]
                        i = int(target[0])
                        j = int(target[1])
                        rng = np.random.default_rng(seed)
                        bge_model = BGe(d=d, alpha_u=10)
                        true_effects = np.linalg.inv(np.eye(d) - B)
                        ce = pairwise_linear_ce_no_params(np.copy([G]), X_train, bge_model, params_per_graph=50, avg=True, return_B=False)
                        print("approx: ", ce)
                        print("true effects: ", true_effects)
                        print(B)
                        print(target_value_list)
                        tmp = Path(save_dir)
                        print(true_effects[i, j])
                        tmp.mkdir(parents=True, exist_ok=True)
                        output_file = os.path.join(tmp, "output_results.txt")
                        multilevel_model = Multilevel(
                            bge_model, X_train, target_value_list, i, j, tmp, output_file, 
                            max_outer_iter=10, 
                            rng=rng, 
                            structure_kernel = structure_kernel,
                            joint_kernel = joint_kernel
                        )

                        # levels, log_probs = multilevel_model.calculate_level_probability(n, mcmc_iterations)
                        # lin_probs = np.exp(log_probs)       
                        # result.append(lin_probs.tolist())  
                        # os.makedirs(save_dir, exist_ok=True)
                        # with open(output_file, "a") as f:
                        #     log_and_print(f"case {num}", f)
                        #     log_and_print(f"True effects: {true_effects[i, j]}", f)
                        #     log_and_print(f"approx effect: {ce[i, j]}", f)
                        #     log_and_print(f"B matrix: {B}", f)
                        #     for target_value, probability in zip(levels, log_probs):
                        #         log_and_print(f"------------------", f)
                        #         log_and_print(f"Target value: {target_value}", f)
                        #         log_and_print(f"P(ACE > {target_value})={np.exp(probability)}, e^{probability}", f)

                        # from adjustText import adjust_text
                        # plt.figure(figsize=(8, 6))
                        # plt.plot(levels, np.log10(lin_probs), marker='o', linestyle='-', color='b', label='P(ACE > X)')
                        # texts = []
                        # for x, y in zip(levels, np.log10(lin_probs)):
                        #     text = plt.text(x, y, f"{y:.2e}", fontsize=6)
                        #     texts.append(text)
                        # adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))
                        # plt.xlabel("Target Value")
                        # plt.ylabel("Probability")
                        # plt.title("Probability vs. Target Value (log Scale)")
                        # plt.legend()
                        # plt.grid()
                        # plt.savefig(save_dir + f"/probability_vs_target_value(log).png", dpi=300, bbox_inches='tight')
                        # plt.show()
                        # plt.close()

                        # plt.figure(figsize=(8, 6))
                        # plt.plot(levels, lin_probs, marker='o', linestyle='-', color='b', label='P(ACE > X)')
                        # texts = []
                        # for x, y in zip(levels, lin_probs):
                        #     text = plt.text(x, y, f"{y:.2e}", fontsize=6)
                        #     texts.append(text)
                        # adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))
                        # plt.xlabel("Target Value")
                        # plt.ylabel("Probability")
                        # plt.title("Probability vs. Target Value (Linear Scale)")
                        # plt.legend()
                        # plt.grid()
                        # plt.savefig(save_dir + f"/probability_vs_target_value.png", dpi=300, bbox_inches='tight')
                        # plt.show()
                        # plt.close()

                        probability_list = multilevel_model.calculate_probability(n, mcmc_iterations)
                        with open(output_file, "a") as f:
                            log_and_print(f"case {num}, run {run_num}", f)
                            log_and_print(f"True effects: {true_effects[i, j]}", f)
                            log_and_print(f"approx effect: {ce[i, j]}", f)
                            log_and_print(f"B matrix: {B}", f)
                            for target_value, probability in zip(target_value_list, probability_list):
                                log_and_print(f"------------------", f)
                                log_and_print(f"Target value: {target_value}", f)
                                log_and_print(f"P(ACE > {target_value})={np.exp(probability)}, e^{probability}", f)
                        print(target_value_list)
                        print(probability_list)
                        result.append(np.exp(probability_list).tolist())
                        from adjustText import adjust_text
                        plt.figure(figsize=(8, 6))
                        plt.plot(target_value_list, probability_list, marker='o', linestyle='-', color='b', label='P(ACE > X)')
                        texts = []
                        for x, y in zip(target_value_list, probability_list):
                            text = plt.text(x, y, f"{y:.2e}", fontsize=6)
                            texts.append(text)
                        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))
                        plt.xlabel("Target Value")
                        plt.ylabel("Probability")
                        plt.title("Probability vs. Target Value (log Scale)")
                        plt.legend()
                        plt.grid()
                        plt.savefig(save_dir + f"/probability_vs_target_value(log).png", dpi=300, bbox_inches='tight')
                        plt.show()
                        plt.close()

                        probability_list = [np.exp(x) for x in probability_list]
                        plt.figure(figsize=(8, 6))
                        plt.plot(target_value_list, probability_list, marker='o', linestyle='-', color='b', label='P(ACE > X)')
                        texts = []
                        for x, y in zip(target_value_list, probability_list):
                            text = plt.text(x, y, f"{y:.2e}", fontsize=6)
                            texts.append(text)
                        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))
                        plt.xlabel("Target Value")
                        plt.ylabel("Probability")
                        plt.title("Probability vs. Target Value (Linear Scale)")
                        plt.legend()
                        plt.grid()
                        plt.savefig(save_dir + f"/probability_vs_target_value.png", dpi=300, bbox_inches='tight')
                        plt.show()
                        plt.close()
                results.append(result)
            # Save the results to a JSON file for later analysis
            results_path = os.path.join(base_dir, "all_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f)
            print(f"Results saved to {results_path}")
