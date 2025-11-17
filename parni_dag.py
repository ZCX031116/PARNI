# -*- coding: utf-8 -*-
# Faithful Python port of PARNI-DAG (from algorithms/*.R)
# Data orientation matches R: X.shape == (p, n)  (rows = variables, cols = samples)

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from trust.utils.bge import BGe

class BGEAdapter:
    """
    适配你上传的 bge.py::BGe
    约定：
      - 你的 BGe.mll(G, X) 计算整图的 log marginal likelihood
      - 你的 BGe._mll_per_variable(i, G[:, i], R, N) 计算单节点/父集的 log marginal likelihood
      - X 形状为 (N, d)；我们的内部数据是 (p, n)，因此需转置
    """
    def __init__(self, bge_obj, X_p_n: np.ndarray):
        assert X_p_n.ndim == 2
        self.p = X_p_n.shape[0]
        self.X_n_d = X_p_n.T             # (n, p) for BGe
        self.N = self.X_n_d.shape[0]
        self.bge = bge_obj
        # 预计算 R，仅一次
        self.R = self.bge.calc_R(self.X_n_d)

    def dag_llh(self, G: np.ndarray) -> float:
        # 整体对数边际似然
        return float(self.bge.mll(G.astype(int), self.X_n_d))

    def local_llh(self, j: int, G: np.ndarray) -> float:
        # 单节点（列 j）局部对数边际似然
        parents_vec = G[:, j].astype(int)
        return float(self.bge._mll_per_variable(j, parents_vec, self.R, self.N))

# ---------- BGE-score versions of log_llh & update ----------
def log_llh_BGE(LA: LAState, hyper_par: HyperPar) -> LAState:
    adapter: BGEAdapter = hyper_par.tables  # 我们把 adapter 暂存在 HyperPar.tables 字段里
    G = LA.curr
    p = hyper_par.p
    A = np.zeros(p, dtype=float)
    for j in range(p):
        A[j] = adapter.local_llh(j, G)
    LA.A = A
    LA.log_det_sigma = None
    LA.llh = float(np.sum(A))
    LA.p_gam = int(G.sum())
    return LA

def log_llh_BGE_update_table(changes: np.ndarray, LA_old: LAState, LA: LAState, hyper_par: HyperPar) -> LAState:
    adapter: BGEAdapter = hyper_par.tables
    G = LA.curr
    A = (LA_old.A.copy() if LA_old.A is not None else np.zeros(hyper_par.p, dtype=float))
    cols = np.unique(np.array(changes, dtype=int))
    # 允许既有 0-based 又有 R 风格 1-based
    cols = np.array([c-1 if c>=1 and c<=hyper_par.p else c for c in cols], dtype=int)
    for j in cols:
        A[j] = adapter.local_llh(j, G)
    LA.A = A
    LA.llh = float(np.sum(A))
    LA.p_gam = int(G.sum())
    return LA


# -----------------------------
# utils: logit_e / inv_logit_e (algorithms/logit_e.R)
# -----------------------------
def logit_e(x: np.ndarray, eps: float) -> np.ndarray:
    x = x.copy()
    x[x > 2*(1-eps)] = 1 - 2*eps
    x[x < 2*eps]     = 2*eps
    return np.log(x - eps) - np.log(1 - x - eps)

def inv_logit_e(y: np.ndarray, eps: float) -> np.ndarray:
    ey = np.exp(-y)
    return (eps*ey - eps + 1) / (ey + 1)

# -----------------------------
# is.DAG_adjmat (algorithms/is.DAG_adjmat.R)
# -----------------------------
def is_DAG_adjmat(W: np.ndarray) -> bool:
    W = (W != 0).astype(int)
    num_precedent = W.sum(axis=0)
    d = W.shape[0]
    while d > 1:
        if np.all(num_precedent > 0):
            return False
        leaf = np.where(num_precedent == 0)[0]
        if leaf.size == 0:
            return False
        mask = np.ones(W.shape[0], dtype=bool)
        mask[leaf] = False
        W = W[mask][:, mask]
        d = W.shape[0]
        if d > 1:
            num_precedent = W.sum(axis=0)
    return True

# -----------------------------
# sample_ind_DAG (algorithms/sample_ind_DAG.R)
# -----------------------------
def sample_ind_DAG(whe_sam: bool, probs: np.ndarray, samples: Optional[np.ndarray]=None, log: bool=False):
    d = probs.shape[0]
    if whe_sam:
        # indices (1-based in R) flattened by column-major; here we keep 0-based flat indices
        draws = (np.random.rand(d, d) < probs).astype(int)
        samples = np.where(draws.ravel(order='F') == 1)[0]
    if log:
        if samples is None or samples.size == 0:
            prob = 0.0
        else:
            # product over selected entries only; R 用 probs[samples]
            flat = probs.ravel(order='F')
            prob = float(np.sum(np.log(flat[samples]+1e-300)))
    else:
        if samples is None or samples.size == 0:
            prob = 1.0
        else:
            flat = probs.ravel(order='F')
            prob = float(np.prod(flat[samples]))
    return {"prob": prob, "sample": samples}

# -----------------------------
# helper: H_to_permi_pars, convert_to_binary, model_encoding (algorithms/other_functions.R + marPIPs_*.R)
# -----------------------------
def H_to_permi_pars(H: np.ndarray) -> List[np.ndarray]:
    p = H.shape[0]
    permi = []
    for j in range(p):
        permi.append(np.where(H[:, j] == 1)[0])
    return permi

def convert_to_binary(n: int, k: Optional[List[int]] = None) -> List[int]:
    # R 的递归写法：convert_to_binary(n, k=c())
    if k is None: k = []
    if n > 1:
        k = convert_to_binary(int(n/2), k)
    return list(k) + [n % 2]

def model_encoding(bitvec: List[int]) -> int:
    # 与 R 一致：把二进制向量视作二进制整数 + 1（作为 1..2^m 的索引）
    out = 0
    for b in bitvec:
        out = (out << 1) | int(b)
    return out + 1

# -----------------------------
# Scores: log_llh_DAG & incremental update (algorithms/log_llh_DAG.R, log_llh_DAG_update_table.R)
# -----------------------------
@dataclass
class HyperPar:
    X: np.ndarray               # shape (p, n)
    g: float
    h: float | Tuple[float,float]   # scalar h (independent Bernoulli) or (alpha,beta)
    p: int
    n: int
    max_p: int
    permi_pars: List[np.ndarray]
    XtX: np.ndarray
    log_llh: Callable           # function(LA, hyper_par) -> LA
    log_llh_update: Callable    # function(changes, LA_old, LA, hyper_par) -> LA
    log_m_prior: Callable       # function(p_gam, h, max_p) -> float
    use_bge: bool = False
    tables: Optional[List[Dict[int, float]]] = None  # 每个 j 的 mask->log_llh（可选）

@dataclass
class LAState:
    curr: np.ndarray            # adjacency matrix (p x p), 0/1
    p_gam: int
    llh: float = 0.0
    lmp: float = 0.0
    log_post: float = 0.0
    A: Optional[np.ndarray] = None              # per-node contribution
    log_det_sigma: Optional[np.ndarray] = None  # per-node log|L|

def _local_llh_from_parents(j: int, parents: np.ndarray, XtX: np.ndarray, n: int, g: float) -> Tuple[float,float]:
    # R: 对 j 的局部打分，返回 (A, log_sqrt_det_Sigma)
    if parents.size == 0:
        xjtxj = XtX[j, j]
        A = - n * np.log(xjtxj/2.0) / 2.0
        return A, 0.0
    V = XtX[np.ix_(parents, parents)].astype(float)
    x_paj_xj_t = XtX[np.ix_(parents, [j])].reshape(-1, 1).astype(float)
    if parents.size == 1:
        V = V + 1.0/g
        L = np.sqrt(V)  # scalar
        log_sqrt_det = float(np.log(L))
        inv_V = 1.0 / V
        quad = float((x_paj_xj_t.T @ inv_V @ x_paj_xj_t).squeeze())
    else:
        V = V.copy()
        V[np.diag_indices_from(V)] += 1.0/g
        L = np.linalg.cholesky(V)
        log_sqrt_det = float(np.sum(np.log(np.diag(L))))
        # 求 (x^T V^{-1} x) 可用解三角方程
        z = np.linalg.solve(L, x_paj_xj_t)
        quad = float((z.T @ z).squeeze())
    xjtxj = XtX[j, j]
    A = - n * np.log((xjtxj - quad)/2.0) / 2.0
    return A, log_sqrt_det

def log_llh_DAG(LA: LAState, hyper_par: HyperPar) -> LAState:
    G = LA.curr
    p = hyper_par.p
    XtX, n, g = hyper_par.XtX, hyper_par.n, hyper_par.g
    A = np.zeros(p, dtype=float)
    log_sqrt = np.zeros(p, dtype=float)
    for j in range(p):
        parents = np.where(G[:, j] == 1)[0]
        a, ls = _local_llh_from_parents(j, parents, XtX, n, g)
        A[j], log_sqrt[j] = a, ls
    LA.llh = - np.sum(log_sqrt) + np.sum(A)
    LA.A = A
    LA.log_det_sigma = log_sqrt
    LA.p_gam = int(G.sum())
    return LA

def log_llh_DAG_update_table(changes: np.ndarray, LA_old: LAState, LA: LAState, hyper_par: HyperPar) -> LAState:
    # 只重算受影响列（与 R 名称一致的增量接口）
    G = LA.curr
    XtX, n, g = hyper_par.XtX, hyper_par.n, hyper_par.g
    A = LA_old.A.copy()
    log_sqrt = LA_old.log_det_sigma.copy()
    for j in np.unique(changes):
        j = int(j) - 1 if j >= 1 else int(j)  # R 是 1-based；这里容错
        parents = np.where(G[:, j] == 1)[0]
        a, ls = _local_llh_from_parents(j, parents, XtX, n, g)
        A[j], log_sqrt[j] = a, ls
    LA.A = A
    LA.log_det_sigma = log_sqrt
    LA.llh = - float(np.sum(log_sqrt)) + float(np.sum(A))
    LA.p_gam = int(G.sum())
    return LA

# -----------------------------
# Compute_LA_DAG (algorithms/Compute_LA_DAG.R)
# -----------------------------
def compute_LA_DAG(gamma: np.ndarray, hyper_par: HyperPar) -> LAState:
    LA = LAState(curr=gamma.copy().astype(int), p_gam=int(gamma.sum()))
    LA = hyper_par.log_llh(LA, hyper_par)
    log_m_prior = hyper_par.log_m_prior(LA.p_gam, hyper_par.h, hyper_par.max_p)
    LA.lmp = log_m_prior
    LA.log_post = LA.llh + log_m_prior
    return LA

# -----------------------------
# marPIPs_DAG_H: enumerate all subsets (algorithms/marPIPs_DAG_H.R)
# -----------------------------
def marPIPs_DAG_H(hyper_par: HyperPar, kappa: float=0.0) -> Tuple[np.ndarray, List[Dict[int,float]]]:
    # 返回 (PIPs, tables)；tables[j] 是 mask(int)->log_llh
    p, n, g, h = hyper_par.p, hyper_par.n, hyper_par.g, hyper_par.h
    permi = hyper_par.permi_pars
    XtX = hyper_par.XtX
    tables: List[Dict[int,float]] = []
    PIPs_all = np.zeros((p, p), dtype=float)

    def log_m_prior_local(k: int, hval, max_p) -> float:
        # 这里对单节点父集大小使用独立伯努利边先验的等价：k*log(h) + (|Pa_set|-k)*log(1-h)
        if isinstance(hval, (tuple, list)):
            # Beta-Binomial 版本（与 PARNI.R 相同参数化）
            alpha, beta = hval
            # 单节点内的 size-先验（用于枚举时的不完全对称情形，这里取独立近似）
            return 0.0  # 与 R 保持：在 marPIPs_bge_H 分支里 often 设 0；对 DAG_H 我用 0 更稳健
        else:
            # 独立伯努利近似：只影响归一化，不改相对比值
            return k * np.log(hval / (1.0 - hval))  # 常数项略去

    for j in range(p):
        Pa_j = np.array(permi[j], dtype=int)
        pj = len(Pa_j)
        table_j: Dict[int, float] = {}
        # 枚举所有子集（2^pj）
        # mask 从 0..(2^pj-1)；mask 的二进制位对应是否选该候选父
        log_weights = []
        masks = []
        for mask in range(1<<pj):
            sel = np.nonzero([(mask>>k)&1 for k in range(pj)])[0]
            parents = Pa_j[sel] if sel.size>0 else np.array([], dtype=int)
            a, ls = _local_llh_from_parents(j, parents, XtX, n, g)
            log_llh = -ls + a
            lmp = log_m_prior_local(len(sel), h, pj)
            lw = log_llh + lmp
            table_j[mask] = log_llh
            log_weights.append(lw)
            masks.append(mask)
        # 归一化得到 PIP
        log_weights = np.array(log_weights)
        mmax = np.max(log_weights)
        w = np.exp(log_weights - mmax)
        Z = w.sum()
        probs = w / Z
        # PIP for each candidate parent
        PIP_j = np.zeros(p, dtype=float)
        for idx, mask in enumerate(masks):
            if mask == 0: 
                continue
            bits = [(mask>>k)&1 for k in range(pj)]
            for k, b in enumerate(bits):
                if b == 1:
                    PIP_j[Pa_j[k]] += probs[idx]
        PIPs_all[:, j] = PIP_j
        tables.append(table_j)

    # kappa 平移（marPIPs_*_H.R 的 “PIPs <- PIPs*(1-2*kappa)+kappa”，随后做方向归一化）
    PIPs = PIPs_all*(1-2*kappa) + kappa
    # 有向互斥归一化：P(i->j) = odds_ij / (odds_ij + odds_ji + 1)
    odds = PIPs / np.maximum(1e-12, 1-PIPs)
    Pdir = odds / (odds + odds.T + 1.0 + 1e-12)
    return Pdir, tables

def marPIPs_DAG_H_bge(hyper_par: HyperPar, kappa: float=0.0) -> Tuple[np.ndarray, List[dict]]:
    adapter: BGEAdapter = hyper_par.tables
    p, h = hyper_par.p, hyper_par.h
    permi = hyper_par.permi_pars
    PIPs_all = np.zeros((p,p), float)
    tables_out: List[dict] = []

    def lmp_single(k: int, hval) -> float:
        # 与 marPIPs_*_H.R 保持“只做相对权重”的简单先验；Beta-Binomial 可按需要加上
        if isinstance(hval, (tuple, list)):
            return 0.0
        return k * np.log(hval / (1.0 - hval))

    for j in range(p):
        cand = np.array(permi[j], dtype=int)
        pj = len(cand)
        table_j = {}
        logw = []
        masks = []
        # 基于 BGe 的局部对数似然，枚举 2^pj 个父集
        # 为了调 adapter.local_llh，需要临时构造 G 的该列父集状态
        base_col = np.zeros(p, dtype=int)
        for mask in range(1<<pj):
            col = base_col.copy()
            if mask:
                sel_bits = [(mask>>k)&1 for k in range(pj)]
                col[cand[np.where(sel_bits)[0]]] = 1
            # 只构造一列 G 即可（其它列不参与局部值）
            G_tmp = np.zeros((p,p), dtype=int); G_tmp[:, j] = col
            llh_loc = adapter.local_llh(j, G_tmp)
            table_j[mask] = llh_loc
            logw.append(llh_loc + lmp_single(int(col.sum()), h))
            masks.append(mask)
        # 归一化并累计 PIP
        logw = np.asarray(logw)
        m = np.max(logw); w = np.exp(logw - m); Z = w.sum()
        probs = w / Z
        PIP_j = np.zeros(p, float)
        for idx, mask in enumerate(masks):
            if mask == 0: continue
            bits = [(mask>>k)&1 for k in range(pj)]
            for k, b in enumerate(bits):
                if b: PIP_j[cand[k]] += probs[idx]
        PIPs_all[:, j] = PIP_j
        tables_out.append(table_j)

    # kappa 平移 & 有向互斥归一化
    PIPs = PIPs_all*(1-2*kappa) + kappa
    odds = PIPs / np.maximum(1e-12, 1-PIPs)
    Pdir = odds / (odds + odds.T + 1.0 + 1e-12)
    return Pdir, tables_out

# -----------------------------
# update_LA_DAG (翻译自 algorithms/update_LA_DAG_sawp.R)
# -----------------------------
def _get_moves() -> np.ndarray:
    # 4 个组合： (0,0),(1,0),(0,1),(1,1)
    return np.array([[0,0],[1,0],[0,1],[1,1]], dtype=int)

def _get_omega_vec(omega: float, change_idx: int, M: np.ndarray) -> np.ndarray:
    # 与 R 同名：基于被选动作与其它动作的汉明距离调节
    # M: (4,2)
    base = M[change_idx]
    dist = np.sum(np.abs(M - base), axis=1)
    return np.power(omega, dist) * np.power(1-omega, 2-dist)

# -------- LOG-domain robust update_LA_DAG (drop-in replacement) --------
def _logsumexp(logw: np.ndarray) -> float:
    m = np.max(logw)
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(logw - m))))

def _safe_log_odds(p: float, eps: float = 1e-12) -> float:
    p = float(np.clip(p, eps, 1 - eps))
    return float(np.log(p) - np.log(1 - p))

def update_LA_DAG(LA: LAState, k: np.ndarray, hyper_par: HyperPar,
                  bal_fun: Callable[[float], float],
                  PIPs: np.ndarray, thinning_rate: float, omega: float):
    """
    LOG-domain, overflow-safe version.
    与原函数同签名/同返回；g(x)=min(1,x) 在 log 域实现为 log_g(L)=min(0,L)。
    """
    temp = LA.curr.copy()
    LA_temp = LA
    log_post_temp = LA.log_post
    max_p, p, h = hyper_par.max_p, hyper_par.p, hyper_par.h
    log_llh_update = hyper_par.log_llh_update
    log_m_prior    = hyper_par.log_m_prior

    d = p
    if k is None or len(k) == 0:
        return dict(LA_prop=LA, JD=0, acc_rate=0.0, thinned_k_size=0,
                    prob_prop=0.0, rev_prob_prop=0.0,
                    prod_bal_con=0.0, rev_prod_bal_con=0.0)

    # 将一维列优先索引转换成 (i,j) 对
    ij = np.array([(idx % d, idx // d) for idx in k], dtype=int)

    # 按“反向”成对；没有反向就单独成组
    ij_pairs = []
    used = set()
    for (i, j) in ij:
        if (i, j) in used:
            continue
        rev_idx = np.where((ij[:, 0] == j) & (ij[:, 1] == i))[0]
        if rev_idx.size > 0:
            used.add((i, j)); used.add((j, i))
            ij_pairs.append((i * d + j, j * d + i))
        else:
            used.add((i, j))
            ij_pairs.append((i * d + j, np.inf))
    grouped = np.array(ij_pairs, dtype=object)

    # thinning
    if grouped.size == 0:
        return dict(LA_prop=LA, JD=0, acc_rate=0.0, thinned_k_size=0,
                    prob_prop=0.0, rev_prob_prop=0.0,
                    prod_bal_con=0.0, rev_prod_bal_con=0.0)
    mask_keep = (np.random.rand(grouped.shape[0]) < np.clip(thinning_rate, 0.0, 1.0))
    grouped = grouped[mask_keep]
    thinned_k_size = grouped.shape[0]
    if thinned_k_size == 0:
        return dict(LA_prop=LA, JD=0, acc_rate=0.0, thinned_k_size=0,
                    prob_prop=0.0, rev_prob_prop=0.0,
                    prod_bal_con=0.0, rev_prod_bal_con=0.0)

    M = _get_moves()  # [[0,0],[1,0],[0,1],[1,1]]
    JD = 0
    # 下面均为“对数域累积”
    prob_prop_log = 0.0
    rev_prob_prop_log = 0.0
    prod_bal_con_log = 0.0
    rev_prod_bal_con_log = 0.0

    ln_omega      = np.log(np.clip(omega, 1e-12, 1 - 1e-12))
    ln_1m_omega   = np.log(np.clip(1 - omega, 1e-12, 1.0))

    for idx in range(thinned_k_size):
        kj, kj_swap = grouped[idx]
        kj = int(kj)
        has_swap = not np.isinf(kj_swap)
        if has_swap:
            kj_swap = int(kj_swap)

        i, j = (kj % d), (kj // d)
        temp_kj = int(temp[i, j])
        if has_swap:
            i2, j2 = (kj_swap % d), (kj_swap // d)
            temp_kj_swap = int(temp[i2, j2])

        # 针对 4 种 move 计算 log 权重
        LA_cands: List[Optional[LAState]] = [None]*4
        L_move   = np.full(4, -np.inf, dtype=float)  # L = Δlogpost + log_ok
        flips_v  = np.zeros(4, dtype=int)

        for m_idx in range(4):
            di, dj = M[m_idx]
            G_prop = temp.copy()
            if di != 0:
                G_prop[i, j] = 1 - temp_kj
            if has_swap and dj != 0:
                G_prop[i2, j2] = 1 - temp_kj_swap

            # 避免 2-cycle（i<->j 同时为 1）
            if (G_prop[i, j] == 1 and G_prop[j, i] == 1) or \
               (has_swap and G_prop[i2, j2] == 1 and G_prop[j2, i2] == 1):
                continue
            if not is_DAG_adjmat(G_prop):
                continue

            # 增量更新（只重算涉及的列）
            LA_prop = LAState(curr=G_prop, p_gam=int(G_prop.sum()))
            if has_swap:
                changes = np.unique(np.array([j+1, j2+1], dtype=int))
            else:
                changes = np.array([j+1], dtype=int)
            LA_prop = log_llh_update(changes, LA_temp, LA_prop, hyper_par)
            lmp_prop = log_m_prior(LA_prop.p_gam, h, max_p)
            LA_prop.lmp = lmp_prop
            LA_prop.log_post = LA_prop.llh + lmp_prop
            LA_cands[m_idx] = LA_prop

            # log_ok：仅对被 flip 的边累加
            log_ok = 0.0
            if di != 0:
                log_ok += (2*temp_kj - 1) * _safe_log_odds(PIPs[i, j])
            if has_swap and dj != 0:
                log_ok += (2*temp_kj_swap - 1) * _safe_log_odds(PIPs[i2, j2])

            L_move[m_idx] = float(LA_prop.log_post - log_post_temp + log_ok)
            flips_v[m_idx] = di + (dj if has_swap else 0)

        # 对数域的 move 权重：log_w = flips*ln(ω) + (2-flips)*ln(1-ω) + log_g
        # log_g(L)=min(0,L)
        log_w = flips_v * ln_omega + (2 - flips_v) * ln_1m_omega + np.minimum(0.0, L_move)

        # keep 的额外权重（与原实现保持一致）：(1-ω)*g(1) → log = ln(1-ω)
        keep_log_w = ln_1m_omega

        # 规范化常数（含 keep）
        bal_const_log = _logsumexp(np.append(log_w, keep_log_w))

        # 从 5 个分支（四个 move + keep）抽样
        # 计算标准化概率
        probs = np.exp(log_w - bal_const_log)
        prob_keep = float(np.exp(keep_log_w - bal_const_log))
        # 采样
        u = np.random.rand()
        cum = np.cumsum(probs)
        if u < cum[0]:
            chosen = 0
        elif u < cum[1]:
            chosen = 1
        elif u < cum[2]:
            chosen = 2
        elif u < cum[3]:
            chosen = 3
        else:
            chosen = -1  # keep

        # 反向常数
        if chosen >= 0:
            LA_temp = LA_cands[chosen]
            temp = LA_temp.curr
            log_post_temp = LA_temp.log_post
            JD += int(M[chosen].sum())

            # 反向用 ω-向量（与 R 一致）
            omega_vec = _get_omega_vec(omega, chosen, M)  # 长度 4
            log_omega_vec = np.log(np.clip(omega_vec, 1e-12, 1.0))

            # L 相对 chosen 的差：L_m - L_chosen
            L_rel = L_move - L_move[chosen]
            rev_log_w = np.minimum(0.0, L_rel) + log_omega_vec
            rev_bal_const_log = _logsumexp(np.append(rev_log_w, ln_1m_omega))

            # 累计对数
            prob_prop_log      += float(log_w[chosen] - bal_const_log)
            rev_prob_prop_log  += float(log_omega_vec[chosen] - rev_bal_const_log)
            prod_bal_con_log   += bal_const_log
            rev_prod_bal_con_log += rev_bal_const_log
        else:
            # 选择 keep
            prob_prop_log      += float(keep_log_w - bal_const_log)
            rev_prob_prop_log  += float(keep_log_w - bal_const_log)
            prod_bal_con_log   += bal_const_log
            rev_prod_bal_const = bal_const_log
            rev_prod_bal_con_log += rev_prod_bal_const

    return dict(LA_prop=LA_temp, JD=JD, acc_rate=1.0, thinned_k_size=thinned_k_size,
                prob_prop=prob_prop_log, rev_prob_prop=rev_prob_prop_log,
                prod_bal_con=prod_bal_con_log, rev_prod_bal_con=rev_prod_bal_con_log)

# -----------------------------
# PARNI 主程序 (algorithms/PARNI.R)
# -----------------------------
@dataclass
class AlgPar:
    N: int
    Nb: int
    n_chain: int
    store_chains: bool
    verbose: bool
    eval_f: Optional[Callable[[np.ndarray], float]] = None
    omega_adap: bool = True
    omega_init: float = 0.5
    omega_par: Tuple[float, float, float] = (0.25, 0.5, 0.75)  # PT 时可用
    eps: float = 1e-6
    kappa: float = 0.0
    bal_fun: Callable[[float], float] = lambda x: min(1.0, float(x))
    PIPs_update: Optional[bool] = False
    p_mar_PIPs: float = 0.0
    use_logit_e: bool = True
    H: Optional[np.ndarray] = None
    gamma_init: Optional[np.ndarray] = None
    random_gamma_init: bool = False
    thinning_target: float = 0.5  # 用于自适应 omega 的目标评估次数缩放（简化）

# ======= Lightweight single-step adapter for MLS integration (with detailed comments) =======

def _log_m_prior_beta_binom(p_gam: int, hval, mp: int) -> float:
    """
    计算“图大小”的先验 log 概（只需到加常数项即可，用于 MH 比值足够）。
    - p_gam: 当前图的有向边条数 |E|
    - hval : 
        * 若为标量 h ∈ (0,1)，表示每条(有向)边独立 Bernoulli(h) —— 常见稀疏先验
        * 若为二元组 (alpha, beta)，表示对 |E| 施加 Beta-Binomial 先验（对总边数做控制）
    - mp   : 最大可用有向边数 = p*(p-1)

    说明：
    - 该函数只返回对 MH 接受率有影响的部分（log 概到加常数），不求归一化常数；
      因为 MH 接受率中用到的是“新/旧”的差值（比值的 log），常数会相消。
    """
    if isinstance(hval, (tuple, list)):
        # Beta-Binomial 形式：log C + log Beta(a + |E|, b + mp - |E|) - log Beta(a, b)
        # 其中组合数常数项可视作常数（在 MH 比值中相消），这里保留 Beta 部分的对数。
        a, b = hval
        return float(
            np.log(np.math.gamma(p_gam + a))
            + np.log(np.math.gamma(mp - p_gam + b))
            - np.log(np.math.gamma(mp + a + b))
        )
    else:
        # Bernoulli(h) 独立边先验：log p(γ) = |E| log h + (mp - |E|) log (1 - h)
        h = float(hval)
        # 数值稳定：裁剪 h，避免 log(0)
        h = min(max(h, 1e-12), 1 - 1e-12)
        return float(p_gam * np.log(h) + (mp - p_gam) * np.log(1 - h))
# ---- parni_dag.py 里新增：按论文 3.2 从 H 近似 PIP（BGe 或 g-prior 路线）----

def _local_log_prior_size(s: int, m: int, hval):
    """父集大小 s 的先验 log 概；m = 最大父数 = p-1。支持 Bernoulli(h) 和 Beta-Binomial(a,b)"""
    if isinstance(hval, (tuple, list)):
        a, b = hval
        # log Beta(a+s, b+(m-s)) - log Beta(a, b)
        return float(
            np.log(np.math.gamma(a + s)) +
            np.log(np.math.gamma(b + (m - s))) -
            np.log(np.math.gamma(a + b + m))  # 与常数只差个 logC(m,s)，在相对权重里可忽略
        )
    else:
        h = float(hval); h = min(max(h, 1e-12), 1-1e-12)
        return float(s*np.log(h) + (m - s)*np.log(1 - h))

def _iter_subsets(cand_idx, max_enum=None):
    """枚举 cand_idx 的所有子集；max_enum 超过则截断（按长度从小到大），用于防炸。"""
    cand_idx = list(cand_idx)
    m = len(cand_idx)
    if max_enum is None or (2**m) <= max_enum:
        from itertools import combinations
        for k in range(m+1):
            for comb in combinations(cand_idx, k):
                yield np.array(comb, dtype=int)
    else:
        # 近似：优先小规模子集，再随机补足
        from itertools import combinations
        count = 0; budget = int(max_enum)
        for k in range(min(m, 3)+1):  # 小子集全枚举
            for comb in combinations(cand_idx, k):
                yield np.array(comb, dtype=int); count += 1
                if count >= budget: return
        rng = np.random.default_rng(0)
        while count < budget:
            k = rng.integers(low=0, high=min(m, 6)+1)
            S = rng.choice(cand_idx, size=k, replace=False)
            yield np.sort(S)
            count += 1

def paper_marPIPs_from_H(hp, H: np.ndarray, kappa: float = 0.0,
                         extend_one: bool = True,
                         max_enum_parents: int = 8,
                         max_enum_sets: int = 4096):
    """
    论文 3.2：用骨架 H（每列 j 的许可父集合）近似有向边的 PIP（方向互斥归一），作为式(3) 的热启动。
    - extend_one: 是否做一次性的 h_j^+ 扩展（从 H 列外额外+1 父作比较）
    - max_enum_parents: cand 父集合 |h_j| 的上限；太大则先用权重截断
    - max_enum_sets: 每个 j 的父集枚举最多考虑多少个子集（防指数爆炸）
    返回：PIP_oriented (p,p)，对角线为 0
    """
   
    p = hp.p
    P_raw = np.zeros((p, p), float)  # 列 j 的“边缘 PIP”（未做方向互斥归一）
    # 准备局部似然句柄
    use_bge = getattr(hp, "use_bge", False)
    local_llh = None
    if use_bge and hp.tables is not None and hasattr(hp.tables, "local_llh"):
        # 这里 hp.p 就是变量个数 p
        def local_llh(j, parents):
            Gtmp = np.zeros((hp.p, hp.p), dtype=int)
            if len(parents) > 0:
                Gtmp[np.array(parents, dtype=int), j] = 1  # 只填第 j 列的父
            return hp.tables.local_llh(j, Gtmp)
    else:
        # g-prior 路线维持你现在的 _local_llh_from_parents
        def local_llh(j, parents):
            return _local_llh_from_parents(j, np.array(parents, dtype=int),
                                        hp.XtX, hp.n, hp.g)

    # 列遍历
    for j in range(p):
        cand = np.where(H[:, j] == 1)[0]
        cand = cand[cand != j]
        if cand.size == 0:
            continue
        # 若候选过多，先按相关权重截断（用 XtX 的绝对相关做近似）
        if cand.size > max_enum_parents:
            # 绝对相关近似：|corr(i, j)| ~ |XtX[i,j]|
            wj = np.abs(hp.XtX[cand, j])
            take = np.argsort(-wj)[:max_enum_parents]
            cand = cand[take]

        # 构造 h_j 与 h_j^+ 的候选父集列表
        subsets = list(_iter_subsets(cand, max_enum=max_enum_sets))
        if extend_one:
            outside = np.setdiff1d(np.arange(p), np.append(cand, j))
            # 只取外部前若干个强相关变量，避免爆炸
            if outside.size > 0:
                wout = np.abs(hp.XtX[outside, j])
                keep_o = outside[np.argsort(-wout)[:min(10, outside.size)]]
                ext_sets = []
                for S in subsets:
                    for o in keep_o:
                        ext_sets.append(np.sort(np.append(S, o)))
                if ext_sets:
                    arr_tuples = list(dict.fromkeys(tuple(int(x) for x in s) for s in ext_sets))
                    subsets += [np.array(t, dtype=int) for t in arr_tuples]
                    if len(subsets) > max_enum_sets:
                        subsets = subsets[:max_enum_sets]

        # 计算每个父集的未归一化权重：exp(local_llh + prior_size)
        m = p - 1
        Ws = []
        for S in subsets:
            ll = local_llh(j, S)
            lp = _local_log_prior_size(len(S), m, hp.h)
            Ws.append(ll + lp)
        Ws = np.array(Ws, float)
        # 归一化为父集后验
        Ws -= Ws.max()  # 防溢出
        w = np.exp(Ws)
        Z = w.sum()
        if not np.isfinite(Z) or Z <= 0:
            continue
        post = w / Z
        # 汇总成边的边缘概率
        for S, prob in zip(subsets, post):
            P_raw[S, j] += prob  # 对包含 i 的父集累加

    # 方向互斥归一：禁止 i<->j 同时为 1，用胜算归一得到面向概率
    P = P_raw.copy()
    eps = 1e-9
    odds = np.clip(P / np.clip(1 - P, eps, None), eps, 1/eps)
    for i in range(p):
        for j in range(i+1, p):
            oij, oji = odds[i, j], odds[j, i]
            denom = 1.0 + oij + oji
            pij = oij / denom
            pji = oji / denom
            P[i, j], P[j, i] = pij, pji
    np.fill_diagonal(P, 0.0)

    # κ 收缩：P <- (1-κ)P + κ*0.5（让极端值回中）
    if kappa and kappa > 0:
        P = (1 - kappa) * P + kappa * 0.5
        np.fill_diagonal(P, 0.0)
    return P

def parni_prepare_context(
    X_p_n: np.ndarray,
    h,
    bge_obj=None,
    H: np.ndarray | None = None,
    kappa: float = 0.0,
    omega: float = 0.5,
    pips_mode: str = "uniform",   # "uniform"：PIP=0.5 全平；后续可切 "bge" 以骨架+枚举近似 PIP
):
    """
    构建一次性的 PARNI 上下文（hp, PIPs, A, D, omega 等）。不跑采样。

    参数
    ----
    X_p_n : (p, n) 的数据矩阵（注意：与很多库的 (n, d) 相反，这里“特征在行，样本在列”）
    h     : 图先验超参。可为标量（Bernoulli(h)）或 (alpha, beta)（Beta-Binomial）
    bge_obj : 若提供，则使用 BGe 评分路线；否则走 g-prior 解析路线
    H     : 先验骨架（0/1 矩阵，禁止自环）；若为 None，则默认“完全骨架”（除对角线）
    kappa : PIP 收缩强度（靠近 0.5）；仅在 pips_mode="bge" 时参与 marPIPs 的收缩
    omega : thinning 参数，控制每步评估的子邻域数量（参见论文 §3.4）
    pips_mode : 
        - "uniform"：设置所有边的 PIP=0.5（最稳妥的无偏热启动）
        - "bge"    ：基于 H 的父集枚举 + BGe 得分近似 PIP（大图指数复杂，要谨慎开启）

    返回
    ----
    ctx : dict，包含
        - hp: HyperPar（封装数据、先验与评分函数）
        - PIPs: (p, p) 近似后验边概率（方向化后）
        - A, D: 由 PIP 派生的“加边/删边倾向”矩阵（见下述式(3)映射）
        - omega: thinning 率
        - bal_fun: 子步的平衡函数 g(x)，论文推荐 Hastings 选择 g(x)=min(1,x)

    关键说明
    --------
    1) HyperPar 配置了两种评分路线：
       - BGe：log_llh_BGE / log_llh_BGE_update_table（与权重步一致性最关键）
       - g-prior：log_llh_DAG / log_llh_DAG_update_table
       确保你权重步（HMC/RW）使用的条件后验与这里的边际似然来自“同一家族”，
       以保证联合 π(G,B|D) 的不变性。

    2) PIP -> (A, D) -> η 与式(3)的对应（论文 §3.1）：
       我们从 PIP 构造两张“倾向表”：
         A_ij = min(PIP_ij / (1 - PIP_ij), 1)     # 当前无边时，把它纳入邻域的倾向（鼓励加边）
         D_ij = min((1 - PIP_ij) / PIP_ij, 1)     # 当前有边时，把它纳入邻域的倾向（鼓励删边）
       在具体抽样时，针对当前位置 (i,j)：
         η_ij = (1 - γ_ij) * A_ij + γ_ij * D_ij
       然后按照式(3)的规则基于 η_ij 采样 k_ij（见 parni_step_one 中的调用）。
       这与把 η 直接设为 1-PIP 的写法“单调等价”，效果相同，数值上更稳。
    """
    p, n = X_p_n.shape
    XtX = X_p_n @ X_p_n.T  # 预计算 X X^T，供局部回归闭式项使用

    # 若未提供骨架 H，默认允许所有非自环边（完全骨架）
    if H is None:
        H = np.ones((p, p), dtype=int) - np.eye(p, dtype=int)
    # 骨架 -> 每个节点的“可选父集合”列表（减少父集枚举空间）
    permi = H_to_permi_pars(H)

    # 选择评分路线并构建 HyperPar
    if bge_obj is not None:
        # BGe 路线：适配器把 X 转为 (n, d) 并准备充要统计量
        adapter = BGEAdapter(bge_obj, X_p_n)
        hp = HyperPar(
            X=X_p_n, g=float(n),  h=h, p=p, n=n, max_p=p*(p-1),
            permi_pars=permi, XtX=XtX,
            log_llh=log_llh_BGE, log_llh_update=log_llh_BGE_update_table,
            log_m_prior=lambda pg, hh, mp: _log_m_prior_beta_binom(pg, hh, mp),
            use_bge=True, tables=adapter
        )
    else:
        # g-prior 路线：使用闭式节点可分解边际似然
        hp = HyperPar(
            X=X_p_n, g=float(n), h=h, p=p, n=n, max_p=p*(p-1),
            permi_pars=permi, XtX=XtX,
            log_llh=log_llh_DAG, log_llh_update=log_llh_DAG_update_table,
            log_m_prior=lambda pg, hh, mp: _log_m_prior_beta_binom(pg, hh, mp),
            use_bge=False, tables=None
        )

    # —— PIPs 热启动 —— 
    if (pips_mode in ("bge", "bge_paper")) and (H is not None) and (bge_obj is not None):
        # 论文 §3.2：无环放松父集枚举 + (+1扩展) + 方向互斥归一化
        PIPs = paper_marPIPs_from_H(
            hp, H, kappa=kappa,
            extend_one=True,
            max_enum_parents=8,
            max_enum_sets=4096
        )
    elif (pips_mode == "bge_legacy") and (H is not None) and (bge_obj is not None):
        # 如果你想保留旧实现，就放在这里
        PIPs, _ = marPIPs_DAG_H_bge(hp, kappa=kappa)
    else:
        # uniform 或者缺 H / 缺 bge_obj 的回退
        PIPs = np.full((hp.p, hp.p), 0.5, float)
        np.fill_diagonal(PIPs, 0.0)


    A = np.minimum(PIPs / np.maximum(1e-12, 1 - PIPs), 1.0); np.fill_diagonal(A, 0.0)
    D = np.minimum((1 - PIPs) / np.maximum(1e-12, PIPs), 1.0); np.fill_diagonal(D, 0.0)
    ctx = dict(hp=hp, PIPs=PIPs, A=A, D=D, omega=float(omega),
               bal_fun=(lambda x: min(1.0, float(x))))
    return ctx


def parni_make_LA_from_G(G: np.ndarray, ctx) -> "LAState":
    """
    从给定结构 G（0/1 邻接矩阵，无自环）构建/刷新 LAState 评分缓存。

    作用
    ----
    - 调用 compute_LA_DAG(...) 计算整图的：
        * llh : log p(D | G)（边际似然，节点可分解后累加）
        * lmp : log p(G)     （先验；含“DAG 指示函数”的约束思想）
        * log_post = llh + lmp
      以及用于增量更新的逐列缓存（后续子步只重算受影响列）。
    - LAState 将在 parni_step_one(...) 里传递与更新，是“当前图的记分账本”。
    """
    return compute_LA_DAG(G.astype(int), ctx["hp"])


def parni_step_one(LA: "LAState", ctx, rng: np.random.Generator | None = None,
                   proposal_only: bool = False):
    """
    执行“一次 PARNI 结构步”。

    模式
    ----
    - proposal_only=False（默认）：【原行为】单步 MH，计算 log α 并按 exp(log α) 落地/拒绝；
    - proposal_only=True ：【提案模式】只生成候选，不做接受/拒绝；返回 LA_prop、结构核 log q 的前/后向、
      以及边集差分 A/R/K，供联合核在外层做一次 MH。

    返回
    ----
    - 若 proposal_only=False：与原先一致 -> (LA_new, accepted: bool, info: dict)
    - 若 proposal_only=True ：(LA, False, info: dict)，其中 info 含：
        * "proposal_only": True
        * "LA_prop": LA_prop
        * "log_qG_fwd": float    # log q_G(G->G')
        * "log_qG_rev": float    # log q_G(G'->G)
        * "A", "R", "K": set[(u,v)]
        * "k_size": int
        * 以及 "log_post_curr", "log_post_prop"
    """
    if rng is None:
        rng = np.random.default_rng()

    hp    = ctx["hp"]
    A, D  = ctx["A"], ctx["D"]
    PIPs  = ctx["PIPs"]
    
    gfun  = ctx["bal_fun"]

    # —— (3) 由 A/D 与当前 γ 生成 η，并采样邻域指示 k —— 
    # η_ij = (1 - γ_ij)*A_ij + γ_ij*D_ij
    eta = (1 - LA.curr) * A + LA.curr * D

    # 抽样 k；log=True 让内部同时返回对数概率，便于反向核计算
    neigh = sample_ind_DAG(True, eta, None, log=True)
    k = neigh["sample"]
    # 容错：若本步没有可动位置（极少见），直接“不作为”
    if k is None or (isinstance(k, np.ndarray) and k.size == 0):
        return LA, False, {"reason": "empty_k"}

    # —— (5) 在子邻域里做“知情”子步：先过滤非法 DAG，再按 g(x)=min(1,x) 归一化 —— 
    omega = float(ctx["omega"])
    upd = update_LA_DAG(
        LA, k, hp, gfun, PIPs,
        thinning_rate=omega,  # 控制本步评估的子邻域数量（§3.4）
        omega=omega             # 子提议里“保持不变”的基线概率权重，可按需要调参
    )
    LA_prop: LAState = upd["LA_prop"]

    log_qG_fwd = float(upd["prob_prop"] + upd["prod_bal_con"])
    log_qG_rev = float(upd["rev_prob_prop"] + upd["rev_prod_bal_con"])

    if proposal_only:
        return {
            "proposal_only": True,
            "LA_prop": LA_prop,
            "log_qG_fwd": log_qG_fwd,
            "log_qG_rev": log_qG_rev,
            "k_size": int(upd.get("thinned_k_size", 0)),
            "log_post_curr": float(LA.log_post),
            "log_post_prop": float(LA_prop.log_post),
        }

    # —— (6) 整步接受率：目标后验差 + 前/后向提议核差 + 归一化常数差 —— 
    log_alpha = (LA_prop.log_post - LA.log_post) \
              + (upd["rev_prob_prop"] - upd["prob_prop"]) \
              + (upd["rev_prod_bal_con"] - upd["prod_bal_con"])

    accept = (np.log(rng.random()) < log_alpha)
    return (LA_prop if accept else LA), bool(accept), {
        "log_alpha": float(log_alpha),
        "k_size": int(upd.get("thinned_k_size", 0)),
        "accepted": bool(accept),
    }


# def update_LA_DAG(LA: LAState, k: np.ndarray, hyper_par: HyperPar,
#                   bal_fun: Callable[[float], float],
#                   PIPs: np.ndarray, thinning_rate: float, omega: float):
#     """
#     忠实实现要点：
#     - 把 k 中可构成“反向”对的两个条目成对分组；其余按单边处理
#     - 对每个分组构造 4 种 move (flip/keep 的笛卡尔积)
#     - 用 log_llh_DAG_update_table 增量计算候选
#     - 用 g(x)=min(1,x) 作为 Hastings 平衡函数
#     - 记录前向/反向的 q 概率因子（含 omega 稀疏化、PIP 邻域几率校正）
#     """
#     temp = LA.curr.copy()
#     LA_temp = LA  # work copy
#     log_post_temp = LA.log_post
#     max_p, p, h = hyper_par.max_p, hyper_par.p, hyper_par.h
#     log_llh_update = hyper_par.log_llh_update
#     log_m_prior    = hyper_par.log_m_prior

#     # 将 k（一维索引，列优先 0-based）转成成对（反向）分组
#     # R 里用 swap_idx 把 i->j 与 j->i 对应起来；这里直接算：
#     d = p
#     ij = np.array([(idx % d, idx // d) for idx in k], dtype=int)  # (row=i, col=j)
#     # 只保留 i<j 的一对，第二列给相反方向；若没有则填 inf
#     # 建哈希：i->j 的反向在 where(j->i)
#     ij_pairs = []
#     used = set()
#     for t,(i,j) in enumerate(ij):
#         if (i,j) in used: 
#             continue
#         rev = np.where((ij[:,0]==j) & (ij[:,1]==i))[0]
#         if rev.size>0:
#             used.add((i,j)); used.add((j,i))
#             ij_pairs.append((i*d+j, j*d+i))  # 存平面索引
#         else:
#             used.add((i,j))
#             ij_pairs.append((i*d+j, np.inf))
#     grouped = np.array(ij_pairs, dtype=object)
#     # 稀疏化（thinning）
#     if grouped.size == 0:
#         return dict(LA_prop=LA, JD=0, acc_rate=0.0, thinned_k_size=0,
#                     prob_prop=0.0, rev_prob_prop=0.0,
#                     prod_bal_con=0.0, rev_prod_bal_con=0.0)
#     mask_keep = (np.random.rand(grouped.shape[0]) < max(0.0, min(1.0, thinning_rate)))
#     grouped = grouped[mask_keep]
#     thinned_k_size = grouped.shape[0]
#     if thinned_k_size == 0:
#         return dict(LA_prop=LA, JD=0, acc_rate=0.0, thinned_k_size=0,
#                     prob_prop=0.0, rev_prob_prop=0.0,
#                     prod_bal_con=0.0, rev_prod_bal_con=0.0)

#     M = _get_moves()
#     JD = 0
#     prob_prop = 0.0
#     rev_prob_prop = 0.0
#     prod_bal_con = 0.0
#     rev_prod_bal_con = 0.0

#     for idx in range(thinned_k_size):
#         kj, kj_swap = grouped[idx]
#         kj = int(kj)
#         has_swap = not np.isinf(kj_swap)
#         if has_swap:
#             kj_swap = int(kj_swap)

#         # 当前两条边的状态（0/1）
#         d = p
#         i, j = (kj % d), (kj // d)
#         temp_kj = temp[i, j]
#         if has_swap:
#             i2, j2 = (kj_swap % d), (kj_swap // d)
#             temp_kj_swap = temp[i2, j2]

#         # 生成 4 种 move
#         LA_temps = []
#         prob_change = np.zeros(4, dtype=float)
#         prob_change_k_ratio = np.ones(4, dtype=float)
#         odd_k_change = np.ones(4, dtype=float)

#         # 先算“保持不变”的基准（i=0 -> (0,0)）
#         # 其对数后验就是 log_post_temp；作为 ratio 的分母
#         base_log_post = log_post_temp

#         for m_idx in range(4):
#             di, dj = M[m_idx]
#             # 拟议图
#             G_prop = temp.copy()
#             # flip 第一条
#             if di != 0:
#                 G_prop[i, j] = 1 - temp_kj
#             # flip 第二条（若存在）
#             if has_swap and dj != 0:
#                 if G_prop[j, i] == 1 and di!=0:
#                     # 避免 2-cycle；R 也限制同时为 1 的情况
#                     pass
#                 G_prop[i2, j2] = 1 - temp_kj_swap
#             # 若两条都为 1 则拒绝（避免 2-cycle）
#             if G_prop[i, j] + (G_prop[i2, j2] if has_swap else 0) >= 2:
#                 prob_change[m_idx] = 0.0
#                 LA_temps.append(None)
#                 continue
#             # DAG 约束
#             if not is_DAG_adjmat(G_prop):
#                 prob_change[m_idx] = 0.0
#                 LA_temps.append(None)
#                 continue
#             # 增量更新（只重算受影响列 j / j2）
#             LA_prop = LAState(curr=G_prop, p_gam=int(G_prop.sum()))
#             changes = np.array([j+1], dtype=int)  # +1 模拟 R 索引
#             if has_swap:
#                 changes = np.unique(np.array([j+1, j2+1], dtype=int))
#             # 前面已有：log_llh_update = hyper_par.log_llh_update
#             LA_prop = log_llh_update(changes, LA_temp, LA_prop, hyper_par)

#             lmp_prop = hyper_par.log_m_prior(LA_prop.p_gam, hyper_par.h, hyper_par.max_p)
#             LA_prop.lmp = lmp_prop
#             LA_prop.log_post = LA_prop.llh + lmp_prop
#             LA_temps.append(LA_prop)

#             # 概率比（目标密度）
#             ratio = np.exp(LA_prop.log_post - base_log_post)
#             # 邻域采样几率校正：依赖边的 PIP（与 R 相同结构）
#             mar_eff = []
#             mar_eff.append(PIPs[i, j])
#             if has_swap:
#                 mar_eff.append(PIPs[i2, j2])
#             # (mar/(1-mar))^(2*curr-1)*move_bit
#             # 近似实现：把当前位从 temp_k? -> G_prop_k?，只对翻转的位计入
#             ok = 1.0
#             if di != 0:
#                 ok *= (mar_eff[0] / max(1e-12, 1-mar_eff[0]))**(2*temp_kj - 1)
#             if has_swap and dj != 0:
#                 ok *= (mar_eff[1] / max(1e-12, 1-mar_eff[1]))**(2*temp_kj_swap - 1)
#             odd_k_change[m_idx] = ok
#             prob_change_k_ratio[m_idx] = ratio * ok

#             # 最终每个 move 的权重：omega^|flip_bits| * (1-omega)^(2-|flip_bits|) * g( ratio*odd )
#             flips = di + (dj if has_swap else 0)
#             prob_change[m_idx] = (omega**flips) * ((1-omega)**(2-flips)) * bal_fun(prob_change_k_ratio[m_idx])

#         # 归一化 + “保持不变”的概率（对应 (0,0)）
#         keep_weight = (1-omega) * bal_fun(1.0)
#         bal_const = float(np.sum(prob_change) + keep_weight)
#         probs = prob_change / max(1e-300, bal_const)
#         prob_keep = keep_weight / max(1e-300, bal_const)

#         # 采样一个动作（含“保持不变”）
#         u = np.random.rand()
#         cum = np.cumsum(probs)
#         if u < cum[0]:
#             chosen = 0
#         elif u < cum[1]:
#             chosen = 1
#         elif u < cum[2]:
#             chosen = 2
#         elif u < cum[3]:
#             chosen = 3
#         else:
#             chosen = -1  # keep

#         # 反向归一化常数（用选中的动作反推）
#         if chosen >= 0:
#             # 选中变更
#             LA_temp = LA_temps[chosen]
#             temp = LA_temp.curr
#             log_post_temp = LA_temp.log_post
#             JD += int((M[chosen] if has_swap else np.array([M[chosen,0],0])).sum())
#             # 反向概率常数
#             omega_vec = _get_omega_vec(omega, chosen, M)
#             rev_keep = (1-omega) * bal_fun(1.0)
#             ratios = prob_change_k_ratio / max(1e-300, prob_change_k_ratio[chosen])
#             bal_elems = np.array([bal_fun(float(r)) for r in ratios])
#             rev_weights = bal_elems * omega_vec

#             # rev_weights = bal_fun(prob_change_k_ratio / max(1e-300, prob_change_k_ratio[chosen])) * omega_vec
#             rev_bal_const = float(np.sum(rev_weights) + rev_keep)

#             prob_prop += np.log(max(1e-300, probs[chosen]))
#             rev_prob_prop += np.log(max(1e-300, bal_fun(1.0) * omega_vec[chosen] / max(1e-300, rev_bal_const)))
#         else:
#             # 选择保持不变
#             rev_bal_const = bal_const
#             prob_prop += np.log(max(1e-300, prob_keep))
#             rev_prob_prop += np.log(max(1e-300, prob_keep))

#         prod_bal_con     += np.log(max(1e-300, bal_const))
#         rev_prod_bal_con += np.log(max(1e-300, rev_bal_const))

#     return dict(LA_prop=LA_temp, JD=JD,
#                 acc_rate=1.0,  # 累积式子步骤内的 MH 接受由外层处理
#                 thinned_k_size=thinned_k_size,
#                 prob_prop=prob_prop, rev_prob_prop=rev_prob_prop,
#                 prod_bal_con=prod_bal_con, rev_prod_bal_con=rev_prod_bal_con)

# def PARNI(alg_par: AlgPar, hyper_raw: Dict) -> Dict:
#     # ---- 初始化 hyper_par ----
#     X: np.ndarray = hyper_raw["X"]     # (p, n)
#     g: float      = hyper_raw.get("g", 10.0)  # BGe 时用不到，留默认即可
#     if "h" not in hyper_raw:
#         raise ValueError("hyper_raw['h'] 缺失：请提供边先验 h（标量）或 (alpha, beta) 元组。")
#     h            = hyper_raw["h"]     # scalar or (alpha,beta)
#     p            = X.shape[0]
#     n            = X.shape[1]
#     max_p        = p*(p-1)
#     XtX          = X @ X.T

#     H = alg_par.H
#     if H is None:
#         H = np.ones((p,p), dtype=int) - np.eye(p, dtype=int)

#     permi = H_to_permi_pars(H)

#     bge_obj = hyper_raw.get("bge", None)
#     if bge_obj is not None:
#         adapter = BGEAdapter(bge_obj, X)  # 注意 X 仍是 (p, n)；适配器内部会转置
#         hp = HyperPar(
#             X=X, g=g, h=h, p=p, n=n, max_p=p*(p-1), permi_pars=permi,
#             XtX=XtX, log_llh=log_llh_BGE, log_llh_update=log_llh_BGE_update_table,
#             log_m_prior=(lambda p_gam, hval, mp: (p_gam*np.log(hval) + (mp - p_gam)*np.log(1-hval)) if not isinstance(hval,(tuple,list))
#                          else float(np.log(np.math.gamma(p_gam+hval[0])) + np.log(np.math.gamma(mp-p_gam+hval[1])) - np.log(np.math.gamma(mp+hval[0]+hval[1])))),
#             use_bge=True, tables=adapter  # <-- tables 字段挂载 adapter
#         )
#         approx_PIPs, _ = marPIPs_DAG_H_bge(hp, kappa=alg_par.kappa)
#     else:
#         # ---- 原 g-prior 路线（保持不变） ----
#         hp = HyperPar(
#             X=X, g=g, h=h, p=p, n=n, max_p=p*(p-1), permi_pars=permi,
#             XtX=XtX, log_llh=log_llh_DAG, log_llh_update=log_llh_DAG_update_table,
#             log_m_prior=(lambda p_gam, hval, mp: (p_gam*np.log(hval) + (mp - p_gam)*np.log(1-hval)) if not isinstance(hval,(tuple,list))
#                          else float(np.log(np.math.gamma(p_gam+hval[0])) + np.log(np.math.gamma(mp-p_gam+hval[1])) - np.log(np.math.gamma(mp+hval[0]+hval[1])))),
#             use_bge=False, tables=None
#         )
#         approx_PIPs, _ = marPIPs_DAG_H(hp, kappa=alg_par.kappa)

#     # 近似 PIPs（warm-start）
#     PIPs = approx_PIPs.copy()

#     # 计算 A 与 D
#     A = np.minimum(PIPs / np.maximum(1e-12, 1-PIPs), 1.0)
#     np.fill_diagonal(A, 0.0)
#     D = np.minimum((1-PIPs) / np.maximum(1e-12, PIPs), 1.0)
#     np.fill_diagonal(D, 0.0)

#     # ---- 多链初始化 ----
#     chains: List[List[np.ndarray]] = [[] for _ in range(alg_par.n_chain)]
#     LAs:    List[LAState] = []
#     log_posts = np.full((alg_par.N+1, alg_par.n_chain), np.nan, dtype=float)

#     for i in range(alg_par.n_chain):
#         if alg_par.gamma_init is None:
#             if alg_par.random_gamma_init:
#                 # 随机 DAG
#                 ok = False
#                 while not ok:
#                     G0 = (np.random.rand(p,p) < (0.5 if not isinstance(h,(tuple,list)) else 0.5)).astype(int)
#                     np.fill_diagonal(G0, 0)
#                     ok = is_DAG_adjmat(G0)
#                 curr = G0
#             else:
#                 curr = np.zeros((p,p), dtype=int)
#         else:
#             curr = alg_par.gamma_init.copy().astype(int)
#         LA0 = compute_LA_DAG(curr, hp)
#         LAs.append(LA0)
#         log_posts[0, i] = LA0.log_post
#         if alg_par.store_chains:
#             chains[i].append(curr.copy())

#     omegas = np.full(alg_par.N+1, alg_par.omega_init, dtype=float)
#     omega = alg_par.omega_init

#     # ---- 迭代 ----
#     for it in range(1, alg_par.N+1):
#         # 简化的 Robbins–Monro：把“每步被评估的子邻域数”向目标调
#         target_evals = alg_par.thinning_target * p
#         avg_eval_this_iter = 0.0

#         for ic in range(alg_par.n_chain):
#             LA = LAs[ic]
#             curr = LA.curr
#             # 组装 η 并抽 k
#             eta = (1-curr)*A + curr*D
#             neigh = sample_ind_DAG(True, eta, None, log=True)
#             k = neigh["sample"]  # 一维列优先索引
#             if k is None or k.size == 0:
#                 # nothing to do; 记录，继续
#                 log_posts[it, ic] = LA.log_post
#                 if alg_par.store_chains:
#                     chains[ic].append(LA.curr.copy())
#                 continue

#             # 调用逐步更新（包含前/反向 q 概率的积）
#             upd = update_LA_DAG(LA, k, hp, alg_par.bal_fun, PIPs,
#                                 thinning_rate=omega, omega=0.5)

#             LA_prop: LAState = upd["LA_prop"]
#             # 总 MH 接受
#             log_alpha = (LA_prop.log_post - LA.log_post) \
#                         + (upd["rev_prob_prop"] - upd["prob_prop"]) \
#                         + (upd["rev_prod_bal_con"] - upd["prod_bal_con"])
#             if np.log(np.random.rand()) < log_alpha:
#                 LAs[ic] = LA_prop
#             # 记录
#             log_posts[it, ic] = LAs[ic].log_post
#             if alg_par.store_chains:
#                 chains[ic].append(LAs[ic].curr.copy())

#             # 统计评估次数（粗略估计：变更组数）
#             avg_eval_this_iter += max(1, upd["thinned_k_size"])

#         avg_eval_this_iter /= max(1, alg_par.n_chain)
#         # 自适应 omega（在 logit_ε 空间）
#         if alg_par.omega_adap:
#             z = float(logit_e(np.array([omega]), alg_par.eps))
#             psi = it**(-0.7)
#             z_new = z - psi * (avg_eval_this_iter - target_evals)
#             omega = float(inv_logit_e(np.array([z_new]), alg_par.eps))
#             omega = min(0.99, max(0.01, omega))
#             omegas[it] = omega

#     return dict(
#         chains=chains if alg_par.store_chains else None,
#         final_LAs=LAs,
#         log_posts=log_posts,
#         omegas=omegas,
#         PIPs=PIPs,
#         A=A, D=D
#     )
