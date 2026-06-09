import pandas as pd
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import os
from sklearn.cluster import SpectralClustering
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, recall_score

MISSING_CODE = -1
REF_CODE = 0
ALT_CODE = 1
OTHER_CODE = 2
VALID_CODES = (REF_CODE, ALT_CODE, OTHER_CODE)
LABEL_TO_CODE = {'ref': REF_CODE, 'alt': ALT_CODE, 'other': OTHER_CODE}


###################################---------------some functions----------------################################

def coerce_r_codes(df_r):
    r = df_r.to_numpy() if isinstance(df_r, pd.DataFrame) else np.asarray(df_r)
    if np.issubdtype(r.dtype, np.integer):
        return r.astype(np.int8, copy=False)
    out = np.full(r.shape, MISSING_CODE, dtype=np.int8)
    out[r == REF_CODE] = REF_CODE
    out[r == ALT_CODE] = ALT_CODE
    out[r == OTHER_CODE] = OTHER_CODE
    out[r == 'ref'] = REF_CODE
    out[r == 'alt'] = ALT_CODE
    out[r == 'other'] = OTHER_CODE
    return out


def coerce_pi_array(df_pi):
    pi = df_pi.to_numpy() if isinstance(df_pi, pd.DataFrame) else np.asarray(df_pi)
    return pi.astype(float, copy=False)


def prepare_em_inputs(df_r, df_pi, noise_probs=None):
    r_array = coerce_r_codes(df_r)
    pi_array = coerce_pi_array(df_pi)
    valid_mask = r_array != MISSING_CODE
    snv_observed_indices = [np.flatnonzero(valid_mask[:, j]) for j in range(valid_mask.shape[1])]
    if noise_probs is None:
        noise_probs = p_noise(r_array, pi_array)
    else:
        noise_probs = np.asarray(noise_probs, dtype=float)
    noise_probs = noise_probs.copy()
    noise_probs[~valid_mask] = 1.0
    return {
        'r_array': r_array,
        'pi_array': pi_array,
        'valid_mask': valid_mask,
        'snv_observed_indices': snv_observed_indices,
        'noise_probs': noise_probs,
    }

def concurrence_to_haplo_init(df_r, seed=None):
    n_snvs = df_r.shape[1]
    # Convert to binary matrix: alt = 1, ref/other/NA = 0
    binary_alt = (coerce_r_codes(df_r) == ALT_CODE).astype(np.int32, copy=False)
    concurrence_matrix = binary_alt.T @ binary_alt  # [n_snvs x n_snvs]
    # Run Spectral Clustering
    clustering = SpectralClustering(n_clusters=2, affinity='precomputed',
                                    assign_labels='kmeans', random_state=seed)
    labels = clustering.fit_predict(concurrence_matrix)
    # For each SNV, compute mean connection to its own cluster
    scores = np.zeros(n_snvs)
    for i in range(n_snvs):
        cluster_i = labels[i]
        in_cluster = np.where(labels == cluster_i)[0]
        if len(in_cluster) > 1:
            scores[i] = (concurrence_matrix[i, in_cluster].sum() -  concurrence_matrix[i, i] )/ (len(in_cluster) - 1)
        else:
            scores[i] = 0
    # Compute global median score to filter noise
    median_score = np.median(scores)
    keep = scores >= median_score
    # Assign h_A values: 0.9 for cluster 0, 0.1 for cluster 1, 0.5 (±ε) for noise
    epsilon = 1e-3
    rng = np.random.default_rng(seed=seed)
    ha0 = 0.5 + rng.uniform(-epsilon, epsilon, size=n_snvs)
    for idx in np.where(keep)[0]:
        ha0[idx] = 0.9 if labels[idx] == 0 else 0.1
    return ha0



def binary_metrics(pred_probs, true_labels, threshold=0.5):
    auc = roc_auc_score(true_labels, pred_probs)
    preds = (np.array(pred_probs) >= threshold).astype(int)
    sensitivity = recall_score(true_labels, preds, pos_label=1)
    specificity = recall_score(true_labels, preds, pos_label=0)
    return auc, sensitivity, specificity


def compute_P_het(df_r, df_pi, priors=(0.4, 0.2, 0.4), coverage_factor = 1) -> list:
    n_gene_reads = df_r.shape[0]
    eps = 1e-300
    log_prior = np.log(np.maximum(priors, eps))
    r_array = coerce_r_codes(df_r)
    pi_array = coerce_pi_array(df_pi)
    valid_mask = r_array != MISSING_CODE
    ref_mask = valid_mask & (r_array == REF_CODE)
    alt_mask = valid_mask & (r_array == ALT_CODE)
    other_mask = valid_mask & (r_array == OTHER_CODE)

    logL = np.zeros((3, r_array.shape[1]), dtype=float)

    if np.any(ref_mask):
        ref_pi = pi_array[ref_mask]
        logL[0] += np.log(np.maximum(np.where(ref_mask, 1.0 - pi_array, 1.0), eps)).sum(axis=0)
        logL[1] += np.log(np.maximum(np.where(ref_mask, 0.5 * (1.0 - pi_array) + 0.5 * (pi_array / 3.0), 1.0), eps)).sum(axis=0)
        logL[2] += np.log(np.maximum(np.where(ref_mask, pi_array / 3.0, 1.0), eps)).sum(axis=0)

    if np.any(alt_mask):
        logL[0] += np.log(np.maximum(np.where(alt_mask, pi_array / 3.0, 1.0), eps)).sum(axis=0)
        logL[1] += np.log(np.maximum(np.where(alt_mask, 0.5 * (1.0 - pi_array) + 0.5 * (pi_array / 3.0), 1.0), eps)).sum(axis=0)
        logL[2] += np.log(np.maximum(np.where(alt_mask, 1.0 - pi_array, 1.0), eps)).sum(axis=0)

    if np.any(other_mask):
        other_term = np.log(np.maximum(np.where(other_mask, pi_array / 3.0, 1.0), eps)).sum(axis=0)
        logL += other_term[None, :]

    site_depths = valid_mask.sum(axis=0)
    if n_gene_reads > 0:
        coverage_fracs = site_depths / float(n_gene_reads)
    else:
        coverage_fracs = np.zeros(r_array.shape[1], dtype=float)
    coverage_shrinkage = np.minimum(1.0, np.maximum(0.0001, coverage_fracs ** coverage_factor))

    log_post = log_prior[:, None] + coverage_shrinkage[None, :] * logL
    log_total = np.logaddexp.reduce(log_post, axis=0)
    p_het = np.exp(log_post[1] - log_total)
    p_het = p_het.astype(float, copy=False)

    zero_depth = site_depths == 0
    if np.any(zero_depth):
        p_het[zero_depth] = float(priors[1] / sum(priors))

    return p_het.tolist()


###################################---------------em algorithm----------------################################
# p(r_ij|Ii = 1/0, pi, h_j)
def rij_given_I(rij, hj_A, pi_ij, Ii):
    hj_B = 1 - hj_A
    if rij == REF_CODE or rij == 'ref':
        prob = Ii*(hj_A * pi_ij/3 + hj_B * (1-pi_ij)) + (1-Ii)*(hj_B * pi_ij/3 + hj_A * (1-pi_ij))
    elif rij == ALT_CODE or rij == 'alt':
        prob = Ii * (hj_A * (1-pi_ij) + hj_B * pi_ij/3) + (1-Ii)*(hj_B * (1-pi_ij) + hj_A * pi_ij/3)
    elif rij == OTHER_CODE or rij == 'other':
        prob = 2 * pi_ij/3
    else:
        raise ValueError("Invalid rij")
    return prob

def p_noise(df_r, df_pi):
    r = coerce_r_codes(df_r)
    pi = coerce_pi_array(df_pi)
    is_ref = r == REF_CODE
    is_alt = r == ALT_CODE
    is_other = r == OTHER_CODE
    noise_probs = np.full(r.shape, np.nan, dtype=float)
    noise_probs[is_ref] = 1 - pi[is_ref]
    noise_probs[is_alt] = pi[is_alt] / 3
    noise_probs[is_other] = (2 * pi[is_other]) / 3
    return noise_probs


def emission_probs(r_array, pi_array, h_A, valid_mask):
    """Vectorized emission P(r|I=1) and P(r|I=0) for all (read, SNV) pairs."""
    h = np.asarray(h_A, dtype=float)[None, :]
    hb = 1.0 - h
    pi = np.asarray(pi_array, dtype=float)

    emit_I1 = np.ones_like(pi, dtype=float)
    emit_I0 = np.ones_like(pi, dtype=float)

    ref = valid_mask & (r_array == REF_CODE)
    alt = valid_mask & (r_array == ALT_CODE)
    other = valid_mask & (r_array == OTHER_CODE)

    emit_I1[ref] = (h * (pi / 3.0) + hb * (1.0 - pi))[ref]
    emit_I0[ref] = (hb * (pi / 3.0) + h * (1.0 - pi))[ref]

    emit_I1[alt] = (h * (1.0 - pi) + hb * (pi / 3.0))[alt]
    emit_I0[alt] = (hb * (1.0 - pi) + h * (pi / 3.0))[alt]

    emit_I1[other] = (2.0 * pi / 3.0)[other]
    emit_I0[other] = (2.0 * pi / 3.0)[other]

    return emit_I1, emit_I0



def e_step(df_r, df_pi, alpha, h_A, h_m, noise_probs=None,
           r_array=None, pi_array=None, valid_mask=None):
    if r_array is None or pi_array is None or valid_mask is None:
        prepared = prepare_em_inputs(df_r, df_pi, noise_probs=noise_probs)
        r_array = prepared['r_array']
        pi_array = prepared['pi_array']
        valid_mask = prepared['valid_mask']
        noise_probs = prepared['noise_probs']
    elif noise_probs is None:
        prepared = prepare_em_inputs(df_r, df_pi)
        noise_probs = prepared['noise_probs']
    N, M = r_array.shape
    h_A = np.asarray(h_A, dtype=float)
    h_m = np.asarray(h_m, dtype=float)
    tiny = 1e-300

    log_alpha = np.log(alpha) if alpha > 0 else -np.inf
    log_1malpha = np.log(1 - alpha) if alpha < 1 else -np.inf
    _hm_safe = np.clip(h_m, 1e-300, 1 - 1e-300)
    log_hm = np.where(h_m > 0, np.log(_hm_safe), -np.inf)
    log_1mhm = np.where(h_m < 1, np.log(1 - _hm_safe), -np.inf)

    w = h_m[None, :]
    emit_I1, emit_I0 = emission_probs(r_array, pi_array, h_A, valid_mask)

    prob_I1 = w * (0.5 * emit_I1) + (1.0 - w) * noise_probs
    prob_I0 = w * (0.5 * emit_I0) + (1.0 - w) * noise_probs

    log_prob_I1 = np.zeros((N, M), dtype=float)
    log_prob_I1[valid_mask] = np.log(np.maximum(prob_I1[valid_mask], tiny))
    log_prob_I0 = np.zeros((N, M), dtype=float)
    log_prob_I0[valid_mask] = np.log(np.maximum(prob_I0[valid_mask], tiny))

    logLi_I1 = log_prob_I1.sum(axis=1)
    logLi_I0 = log_prob_I0.sum(axis=1)

    log_post_I1 = log_alpha + logLi_I1
    log_post_I0 = log_1malpha + logLi_I0
    log_norm_I = np.logaddexp(log_post_I1, log_post_I0)
    hat_I = np.exp(log_post_I1 - log_norm_I)

    emit_Ihat = hat_I[:, None] * emit_I1 + (1.0 - hat_I[:, None]) * emit_I0
    prob_Z1 = 0.5 * emit_Ihat
    prob_Z0 = noise_probs

    log_prob_Z1 = np.zeros((N, M), dtype=float)
    log_prob_Z1[valid_mask] = np.log(np.maximum(prob_Z1[valid_mask], tiny))
    log_prob_Z0 = np.zeros((N, M), dtype=float)
    log_prob_Z0[valid_mask] = np.log(np.maximum(prob_Z0[valid_mask], tiny))

    logLj_Z1 = log_prob_Z1.sum(axis=0)
    logLj_Z0 = log_prob_Z0.sum(axis=0)

    log_post_Z1 = log_hm + logLj_Z1
    log_post_Z0 = log_1mhm + logLj_Z0
    log_norm_Z = np.logaddexp(log_post_Z1, log_post_Z0)
    hat_Z = np.exp(log_post_Z1 - log_norm_Z)
    return hat_I, hat_Z

#（19）
def Qj_objective(hj_A, j, df_r, df_pi, hat_I, hat_Z, noise_probs=None,
                 r_array=None, pi_array=None, snv_observed_indices=None):
    if hj_A < 0 or hj_A > 1:
        return np.inf
    if r_array is None or pi_array is None or snv_observed_indices is None:
        prepared = prepare_em_inputs(df_r, df_pi, noise_probs=noise_probs)
        r_array = prepared['r_array']
        pi_array = prepared['pi_array']
        snv_observed_indices = prepared['snv_observed_indices']
        noise_probs = prepared['noise_probs']
    hat_Zj = hat_Z[j]
    hat_I = np.asarray(hat_I)
    rows = snv_observed_indices[j]
    if len(rows) == 0:
        return 0.0

    rj = r_array[rows, j]
    pij = pi_array[rows, j]
    Ij = hat_I[rows]
    noisej = noise_probs[rows, j]

    tiny = 1e-300
    hb = 1.0 - hj_A

    emit1 = np.empty(len(rows), dtype=float)
    emit0 = np.empty(len(rows), dtype=float)

    ref = rj == REF_CODE
    alt = rj == ALT_CODE
    other = rj == OTHER_CODE

    emit1[ref] = hj_A * (pij[ref] / 3.0) + hb * (1.0 - pij[ref])
    emit0[ref] = hb * (pij[ref] / 3.0) + hj_A * (1.0 - pij[ref])
    emit1[alt] = hj_A * (1.0 - pij[alt]) + hb * (pij[alt] / 3.0)
    emit0[alt] = hb * (1.0 - pij[alt]) + hj_A * (pij[alt] / 3.0)
    emit1[other] = 2.0 * pij[other] / 3.0
    emit0[other] = 2.0 * pij[other] / 3.0

    emit = Ij * emit1 + (1.0 - Ij) * emit0
    prob = hat_Zj * (0.5 * emit) + (1.0 - hat_Zj) * noisej
    return -np.sum(np.log(np.maximum(prob, tiny)))

#（19）, (20)
def Q_total_objective(h_A, df_r, df_pi, hat_I, hat_Z, noise_probs=None,
                      r_array=None, pi_array=None, valid_mask=None):
    if r_array is None or pi_array is None or valid_mask is None:
        prepared = prepare_em_inputs(df_r, df_pi, noise_probs=noise_probs)
        r_array = prepared['r_array']
        pi_array = prepared['pi_array']
        valid_mask = prepared['valid_mask']
        noise_probs = prepared['noise_probs']
    elif noise_probs is None:
        prepared = prepare_em_inputs(df_r, df_pi)
        noise_probs = prepared['noise_probs']

    tiny = 1e-300
    hat_I = np.asarray(hat_I, dtype=float)
    hat_Z = np.asarray(hat_Z, dtype=float)

    emit_I1, emit_I0 = emission_probs(r_array, pi_array, h_A, valid_mask)
    emit_Ihat = hat_I[:, None] * emit_I1 + (1.0 - hat_I[:, None]) * emit_I0
    prob = hat_Z[None, :] * (0.5 * emit_Ihat) + (1.0 - hat_Z[None, :]) * noise_probs

    log_p = np.zeros_like(prob, dtype=float)
    log_p[valid_mask] = np.log(np.maximum(prob[valid_mask], tiny))
    total = -log_p.sum()
    if not np.isfinite(total):
        return np.inf
    return total



def _classifier_gap_filter(h_m, gap_tau=0.10, min_keep=1):
    """Gap-only filter for classifier-informed h_m values.
    If a clear gap exists, cut there. Otherwise keep ALL (no tier 3 fallback).
    Used when classifier + iterative pruning already cleaned the candidate set."""
    h = np.asarray(h_m, dtype=float)
    m = len(h)
    if m == 0:
        return np.zeros(0, dtype=bool)
    if m <= min_keep:
        return np.ones(m, dtype=bool)
    order = np.argsort(-h)
    hs = h[order]
    gaps = hs[:-1] - hs[1:]
    k = int(np.argmax(gaps))
    if gaps[k] >= gap_tau:
        thr = 0.5 * (hs[k] + hs[k + 1])
        keep = h >= thr
        if keep.sum() < min_keep:
            keep[:] = False
            keep[order[:min_keep]] = True
        return keep
    # No clear gap: keep everything (iterative pruning already ensured quality)
    return np.ones(m, dtype=bool)


def adaptive_keep_mask(h_m, min_abs=0.90, gap_tau=0.10, fallback_q=0.50,
                       min_keep=1, max_keep=None):
    """
    h_m: 1D array-like of marker probabilities
    min_abs: absolute threshold to keep obviously strong sites
    gap_tau: minimum largest-gap size to trust elbow
    fallback_q: fallback quantile if no strong sites and no clear elbow (e.g., 0.5 = median)
    min_keep: keep at least this many sites
    max_keep: cap how many to keep (None = no cap)
    """
    h = np.asarray(h_m, dtype=float)
    m = len(h)
    if m == 0:
        return np.zeros(0, dtype=bool)
    # 1) Keep any very confident sites
    keep = h >= min_abs
    if keep.any():
        # optional cap
        if max_keep is not None and keep.sum() > max_keep:
            # keep top max_keep among those ≥ min_abs
            idx = np.argsort(-h[keep])[:max_keep]
            mask = np.zeros_like(keep)
            mask[np.where(keep)[0][idx]] = True
            keep = mask
        # ensure min_keep
        if keep.sum() < min_keep:
            top_idx = np.argsort(-h)[:min_keep]
            keep[:] = False
            keep[top_idx] = True
        return keep
    # 2) No very-strong sites: try elbow (largest gap) on descending scores
    order = np.argsort(-h)            # indices for descending sort
    hs = h[order]                     # sorted scores
    if m >= 2:
        gaps = hs[:-1] - hs[1:]
        k = int(np.argmax(gaps))      # position of largest gap
        if gaps[k] >= gap_tau:
            thr = 0.5 * (hs[k] + hs[k+1])  # cut midway between the two scores
            keep = h >= thr
            # enforce caps
            if max_keep is not None and keep.sum() > max_keep:
                keep[:] = False
                keep[order[:max_keep]] = True
            if keep.sum() < min_keep:
                keep[:] = False
                keep[order[:min_keep]] = True
            return keep
    # 3) Fallback: percentile (e.g., keep top 50%)
    thr = np.quantile(h, fallback_q)
    keep = h >= thr
    if max_keep is not None and keep.sum() > max_keep:
        keep[:] = False
        keep[order[:max_keep]] = True
    if keep.sum() < min_keep:
        keep[:] = False
        keep[order[:min_keep]] = True
    return keep

def run_em(df_r, df_pi, max_iter=50, tol=1e-3, verbose=True, seed = None,
           heterozygous_priors = (0.4, 0.2, 0.4), heterozygous_coverage_factor = 1, h_m_filter = False,
           filter_reads = True, results = None, h_m_init = None, gap_tau = 0.10):
    N0, M0 = df_r.shape
    # Initialize
    alpha = 0.5 #estimated alleleA pct
    if results is None:
        h_A_full = np.array(concurrence_to_haplo_init(df_r, seed)) if M0 >1 else np.ones([1])
    else:
        h_A_full = results['h_A']
    if results is not None:
        h_m_full = results['h_m']
    elif h_m_init is not None:
        h_m_full = np.asarray(h_m_init, dtype=float).reshape(-1)
        if h_m_full.shape[0] != M0:
            raise ValueError(f"h_m_init length {h_m_full.shape[0]} does not match number of SNVs {M0}.")
        h_m_full = np.clip(h_m_full, 1e-6, 1 - 1e-6)
    else:
        h_m_full = np.array(compute_P_het(df_r, df_pi, priors=heterozygous_priors, coverage_factor = heterozygous_coverage_factor))
    if results is not None:
        keep_mask = results['kept_mask']
    elif (M0 > 5) and h_m_filter:
        if h_m_init is not None:
            # Classifier-derived h_m: gap-only filter (pruning already cleaned)
            keep_mask = _classifier_gap_filter(h_m_full, gap_tau=gap_tau, min_keep=4)
        else:
            # compute_P_het h_m: full adaptive filter with fallback
            keep_mask = adaptive_keep_mask(h_m_full, min_abs=0.90, gap_tau=gap_tau,
                                           fallback_q=0.50, min_keep=4)
    else:
        keep_mask = np.ones(M0, dtype=bool)
    df_r_col_reduced = df_r.loc[:, keep_mask]
    df_pi_col_reduced = df_pi.loc[:, keep_mask]
    r_codes_reduced = coerce_r_codes(df_r_col_reduced)
    if filter_reads:
        reads_keep_mask = (r_codes_reduced != MISSING_CODE).any(axis=1)
    else:
        reads_keep_mask = np.ones(N0, dtype=bool)
    if reads_keep_mask.sum() == 0:
        return {
            "alpha": np.nan,
            "h_A": np.full(M0, 0.5, dtype=float),
            "h_m": np.zeros(M0, dtype=float),
            "hat_I": np.full(N0, np.nan, dtype=float),
            "hat_I_binary": np.full(N0, -1, dtype=int),
            "hat_Z_binary": np.zeros(M0, dtype=int),
            "kept_mask": np.ones(M0, dtype=bool),
            "reads_keep_mask": reads_keep_mask,
            "iteration": 0}
    df_r_reduced = df_r_col_reduced.loc[reads_keep_mask]
    df_pi_reduced = df_pi_col_reduced.loc[reads_keep_mask]
    h_m = h_m_full[keep_mask]
    h_A = h_A_full[keep_mask]
    N, M = df_r_reduced.shape
    prepared = prepare_em_inputs(df_r_reduced, df_pi_reduced)
    noise_probs = prepared['noise_probs']
    prev_Q = float('inf')
    iteration = 0
    for iteration in range(max_iter):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")
        # --- E-step ---
        hat_I, hat_Z = e_step(
            df_r_reduced, df_pi_reduced, alpha, h_A, h_m, noise_probs,
            r_array=prepared['r_array'], pi_array=prepared['pi_array'],
            valid_mask=prepared['valid_mask'])
        hat_Z = hat_Z if results is None else results['hat_Z_binary'][keep_mask]
        # --- M-step ---
        # Update alpha
        alpha_new = np.mean(hat_I)
        # update h^marker
        h_m_new = np.array(hat_Z) if results is None else results['h_m'][keep_mask]
        # Update h_A (one-by-one optimization for each SNV j)
        if results is None:
            h_A_new = np.zeros_like(h_A)
            for j in range(M):
                res = minimize_scalar(Qj_objective, bounds=(0.0, 1.0), method='bounded',
                                      args=(j, df_r_reduced, df_pi_reduced, hat_I, hat_Z, noise_probs,
                                            prepared['r_array'], prepared['pi_array'], prepared['snv_observed_indices']))
                h_A_new[j] = res.x if res.success else h_A[j]
        else:
            h_A_new = results['h_A'][keep_mask]
        Q_total = Q_total_objective(
            h_A_new, df_r_reduced, df_pi_reduced, hat_I, hat_Z, noise_probs,
            prepared['r_array'], prepared['pi_array'], prepared['valid_mask'])
        #convergence check
        delta_param = max(abs(alpha - alpha_new),
            np.max(np.abs(h_m - h_m_new)),np.max(np.abs(h_A - h_A_new)))
        delta_q = abs(Q_total - prev_Q) if prev_Q is not None else float("inf")
        if verbose:
            print(f"Max parameter change: {float(delta_param):.6f}")
            print(f"Q objective change:   {float(delta_q):.6f}")
            print(f"Total Q:              {-float(Q_total):.6f}")
        alpha, h_m, h_A, prev_Q = alpha_new, h_m_new, h_A_new, Q_total
        if delta_param < tol or delta_q < tol:
            if verbose:
                print("Convergence reached.")
            break
    h_m_full = np.zeros(M0, dtype=float)  # 0.0 for masked
    h_A_full = np.ones(M0, dtype=float) * 0.5  # 0.5 for masked
    h_m_full[keep_mask] = h_m
    h_A_full[keep_mask] = h_A
    hat_Z_full_binary = np.zeros(M0, dtype=int)
    hat_Z_full_binary[keep_mask] = (h_m > 0.5).astype(int)
    hat_I_full = np.full(N0, np.mean(hat_I), dtype=float)
    hat_I_full[reads_keep_mask] = np.array(hat_I, dtype=float)
    hat_I_full_binary = (hat_I_full > 0.5).astype(int)
    return {"alpha": min(np.mean(hat_I), 1 - np.mean(hat_I)),
            "h_A": h_A_full,"h_m": h_m_full,"hat_I": hat_I_full,
            "hat_I_binary": hat_I_full_binary, "hat_Z_binary": hat_Z_full_binary,
            "kept_mask": keep_mask,"reads_keep_mask": reads_keep_mask,
            'iteration': iteration + 1}




###################################---------------simulation----------------################################
def simulate_rij_truth(Ii, j, hapA_vars, hapB_vars,
                       hapA_vars_somatic, hapB_vars_somatic,
                       somatic_vaf=[0.05,0.5], rng = None):
    pct = rng.uniform(somatic_vaf[0], somatic_vaf[1])
    if j in hapA_vars:
        return 'alt' if Ii == 1 else 'ref'
    elif j in hapB_vars:
        return 'alt' if Ii == 0 else 'ref'
    elif j in hapA_vars_somatic:
        out = rng.choice(['ref','alt'], p =[1-pct, pct])
        return out if Ii == 1 else 'ref'
    elif j in hapB_vars_somatic:
        out = rng.choice(['ref','alt'], p =[1-pct, pct])
        return out if Ii == 0 else 'ref'
    else:
        return 'ref'  # noise variants

def mutate_rij_piij(rij, pi_ij, gamma, rng):
    if rng.random() < gamma:
        return np.nan, np.nan
    if rng.random() < pi_ij:
        if rij =='ref':
            rij = rng.choice(['ref', 'alt', 'other', 'other'], p=[1 - pi_ij, pi_ij / 3, pi_ij / 3, pi_ij / 3])
        elif rij == 'alt':
            rij = rng.choice(['alt', 'ref', 'other', 'other'], p=[1 - pi_ij, pi_ij / 3, pi_ij / 3, pi_ij / 3])
    return rij, pi_ij

def simulate_r_pi(true_I, n_snvs, n_reads, gamma, hapA_vars, hapB_vars,
                  hapA_vars_somatic, hapB_vars_somatic, seed = 42,
                  somatic_vaf=[0.05,0.5]):
    rng = np.random.default_rng(seed)
    pi_data0 = np.random.uniform(0.01, 0.05, size=(n_reads, n_snvs)) #sequence error
    r_data, pi_data = [], []
    for i, Ii in enumerate(true_I):
        ri, pi_i = [], []
        for j in range(n_snvs):
            rij = simulate_rij_truth(Ii, j, hapA_vars, hapB_vars, hapA_vars_somatic, hapB_vars_somatic,
                                     somatic_vaf=somatic_vaf, rng = rng)
            pi_ij = pi_data0[i, j]
            rij, pi_ij = mutate_rij_piij(rij, pi_ij, gamma, rng = rng)
            ri.append(rij)
            pi_i.append(pi_ij)
        r_data.append(ri)
        pi_data.append(pi_i)
    df_pi = pd.DataFrame(pi_data, dtype = 'object')
    df_r = pd.DataFrame(r_data, dtype = 'object')
    all_indices = list(range(n_snvs))
    noise_vars = list(set(all_indices) - set(hapA_vars) - set(hapB_vars) - set(hapA_vars_somatic) -set(hapB_vars_somatic))
    # mandatory some alt for noises
    for j in noise_vars:
        nonmissing_indices = df_r[df_r[j].notna()].index.tolist()
        if len(nonmissing_indices)<2:
            continue
        pct = np.random.uniform(0.01, 0.05)
        n_to_assign = max(1, int(len(nonmissing_indices) * pct))
        chosen = np.random.choice(nonmissing_indices, size=n_to_assign, replace=False)
        for idx in chosen:
            df_r.at[idx, j] = 'alt'
    return df_r, df_pi


def run_simulation_and_evaluate(gamma, n_reads, n_snvs, allele_pct, hap_ratio, max_iter=100,
                                 verbose=False, seed = 42, tol=1e-5, somatic_vaf=[0.05,0.5],
                                clip = True):
    #gamma: average rate of read-variant non-coverage
    #allele_pct: percentage of reads belong to hapA
    #hap_ratio: percentage of variants belong to hapA and hapB, hapA_somatic, hapB_somatic respectively
    assert sum(hap_ratio) <= 1.0, "hap_ratio must sum to ≤ 1.0"
    # --- Step 1: assign true read haplotypes
    n_A = int(n_reads * allele_pct)
    n_B = n_reads - n_A
    true_I = np.array([1] * n_A + [0] * n_B)
    # --- Step 2: assign haplotype-informative variants
    n_hapA = int(n_snvs * hap_ratio[0])
    n_hapB = int(n_snvs * hap_ratio[1])
    n_hapA_somatic = int(n_snvs * hap_ratio[2])
    n_hapB_somatic = int(n_snvs * hap_ratio[3])
    hapA_vars = list(range(n_hapA))
    hapB_vars = list(range(n_hapA, n_hapA + n_hapB))
    hapA_vars_somatic = list(range(n_hapA + n_hapB, n_hapA + n_hapB+n_hapA_somatic))
    hapB_vars_somatic = list(range(n_hapA + n_hapB + n_hapA_somatic, n_hapA + n_hapB + n_hapA_somatic + n_hapB_somatic))
    germline_vars = hapA_vars + hapB_vars
    somatic_vars = hapA_vars_somatic + hapB_vars_somatic
    informative_vars = germline_vars + somatic_vars
    # remaining SNVs are uninformative (noise)
    # --- Step 3: simulate reads and error probabilities
    df_r, df_pi = simulate_r_pi(true_I, n_snvs, n_reads, gamma, hapA_vars, hapB_vars, hapA_vars_somatic, hapB_vars_somatic, seed, somatic_vaf)
    # --- Step 4: run EM
    results = run_em(df_r, df_pi, max_iter=max_iter, tol=tol,verbose=verbose, seed=seed, clip = clip)
    alpha = results["alpha"]
    h_A = np.array(results["h_A"])
    h_m = np.array(results["h_m"])
    hat_I = np.array(results["hat_I"])
    hat_I_binary = np.array(results["hat_I_binary"])
    hat_Z_binary = np.array(results["hat_Z_binary"])
    # --- Step 5: evaluation
    alpha_diff = min(abs(alpha - allele_pct), abs(1 - alpha - allele_pct))
    true_h_A = np.zeros(len(informative_vars))
    true_h_A[hapA_vars] = 1.0  # hapA = 1, hapB = 0 ---germline
    true_h_A[hapA_vars_somatic] = 1.0 # somatic hapA
    pred_h_A = h_A[informative_vars]
    pred_h_A_flipped = 1 - pred_h_A
    auc1, sens1, spec1 = binary_metrics(pred_h_A, true_h_A)
    auc2, sens2, spec2 = binary_metrics(pred_h_A_flipped, true_h_A)
    h_A_auc, h_A_sens, h_A_spec = (auc1, sens1, spec1) if auc1 > auc2 else (auc2, sens2, spec2)
    pred_h_A_binary = (pred_h_A > 0.5).astype(int)
    pred_h_A_binary_flipped = 1 - pred_h_A_binary
    acc1 = np.mean(pred_h_A_binary == true_h_A)
    acc2 = np.mean(pred_h_A_binary_flipped == true_h_A)
    h_A_acc = max(acc1, acc2)
    # Germline h_A
    h_A_auc_germline, h_A_sens_germline, h_A_spec_germline, h_A_acc_germline = None, None, None, None
    if len(germline_vars) > 0:
        pred_h_A_germline = h_A[germline_vars]
        true_h_A_germline = np.zeros(len(germline_vars))
        true_h_A_germline[:len(hapA_vars)] = 1
        pred_h_A_germline_flipped = 1 - pred_h_A_germline
        auc1, sens1, spec1 = binary_metrics(pred_h_A_germline, true_h_A_germline)
        auc2, sens2, spec2 = binary_metrics(pred_h_A_germline_flipped, true_h_A_germline)
        h_A_auc_germline, h_A_sens_germline, h_A_spec_germline = (auc1, sens1, spec1) if auc1 > auc2 else (
        auc2, sens2, spec2)
        pred_h_A_germline_binary = (pred_h_A_germline > 0.5).astype(int)
        pred_h_A_germline_binary_flipped = 1 - pred_h_A_germline_binary
        acc1_g = np.mean(pred_h_A_germline_binary == true_h_A_germline)
        acc2_g = np.mean(pred_h_A_germline_binary_flipped == true_h_A_germline)
        h_A_acc_germline = max(acc1_g, acc2_g)
    # Somatic h_A
    h_A_auc_somatic, h_A_sens_somatic, h_A_spec_somatic, h_A_acc_somatic = None, None, None, None
    if len(somatic_vars) > 0:
        pred_h_A_somatic = h_A[somatic_vars]
        true_h_A_somatic = np.zeros(len(somatic_vars))
        true_h_A_somatic[:len(hapA_vars_somatic)] = 1
        pred_h_A_somatic_flipped = 1 - pred_h_A_somatic
        auc1, sens1, spec1 = binary_metrics(pred_h_A_somatic, true_h_A_somatic)
        auc2, sens2, spec2 = binary_metrics(pred_h_A_somatic_flipped, true_h_A_somatic)
        h_A_auc_somatic, h_A_sens_somatic, h_A_spec_somatic = (auc1, sens1, spec1) if auc1 > auc2 else (auc2, sens2, spec2)
        pred_h_A_somatic_binary = (pred_h_A_somatic > 0.5).astype(int)
        pred_h_A_somatic_binary_flipped = 1 - pred_h_A_somatic_binary
        acc1_s = np.mean(pred_h_A_somatic_binary == true_h_A_somatic)
        acc2_s = np.mean(pred_h_A_somatic_binary_flipped == true_h_A_somatic)
        h_A_acc_somatic = max(acc1_s, acc2_s)
    true_h_marker = np.zeros(n_snvs)
    true_h_marker[informative_vars] = 1
    pred_h_marker = h_m
    Z_auc, Z_sens, Z_spec = binary_metrics(pred_h_marker, true_h_marker)
    Z_acc = max(np.mean(hat_Z_binary == true_h_marker), np.mean(1 - hat_Z_binary == true_h_marker))
    hat_I = np.array(hat_I)
    auc1, sens1, spec1 = binary_metrics(hat_I, true_I)
    auc2, sens2, spec2 = binary_metrics(1 - hat_I, true_I)
    I_auc, I_sens, I_spec = (auc1, sens1, spec1) if auc1 > auc2 else (auc2, sens2, spec2)
    I_acc = max(np.mean(hat_I_binary == true_I), np.mean(1 - hat_I_binary == true_I))
    return {'alpha': min(alpha, 1-alpha), 'alpha_diff': alpha_diff,
        'h_A_auc': h_A_auc,'h_A_sens': h_A_sens,'h_A_spec': h_A_spec,'h_A_acc': h_A_acc,
        'h_A_auc_germline': h_A_auc_germline, 'h_A_sens_germline': h_A_sens_germline,
        'h_A_spec_germline': h_A_spec_germline,'h_A_acc_germline': h_A_acc_germline,
        'h_A_auc_somatic': h_A_auc_somatic, 'h_A_sens_somatic': h_A_sens_somatic,
        'h_A_spec_somatic': h_A_spec_somatic, 'h_A_acc_somatic': h_A_acc_somatic,
        'Z_auc': Z_auc,'Z_acc': Z_acc,'Z_sens': Z_sens,'Z_spec': Z_spec,
        'I_auc': I_auc,'I_acc': I_acc,'I_sens': I_sens,'I_spec': I_spec}




def grid_search_simulation(gamma_list, n_reads_list, n_snvs_list,
                           allele_pct_list, hap_ratio_list,somatic_vaf_list, seed=42,  tol=1e-3, clip = True,
                           max_iter=30, verbose=False, output_folder = None):
    def safe_run(gamma, n_reads, n_snvs, allele_pct, hap_ratio, somatic_vaf_value,
                 max_iter, verbose, seed, clip):
        try:
            res = run_simulation_and_evaluate(
                gamma=gamma,n_reads=n_reads,n_snvs=n_snvs,
                allele_pct=allele_pct,
                hap_ratio=hap_ratio,max_iter=max_iter,
                verbose=verbose,seed=seed, tol = tol,
                somatic_vaf = [somatic_vaf_value, somatic_vaf_value],
                clip = clip)
            return {
                'gamma': gamma,'n_reads': n_reads,'n_snvs': n_snvs,
                'n_informative_snvs_g':int((hap_ratio[0]) * n_snvs) + int((hap_ratio[1]) * n_snvs),
                'n_informative_snvs_s': int((hap_ratio[2]) * n_snvs) + int((hap_ratio[3]) * n_snvs),
                'allele_pct': allele_pct,
                'hapA_ratio': hap_ratio[0],'hapB_ratio': hap_ratio[1],
                'hapA_ratio_somatic': hap_ratio[2],'hapB_ratio_somatic': hap_ratio[3],
                'somatic_vaf_low': somatic_vaf_value, 'somatic_vaf_high': somatic_vaf_value,
                'alpha': res["alpha"], 'alpha_diff': res["alpha_diff"],
                'h_A_auc': res["h_A_auc"], 'h_A_sens': res["h_A_sens"], 'h_A_spec': res["h_A_spec"],'h_A_acc': res["h_A_acc"],
                'h_A_auc_germline': res["h_A_auc_germline"], 'h_A_sens_germline': res["h_A_sens_germline"],
                'h_A_spec_germline': res["h_A_spec_germline"],'h_A_acc_germline': res["h_A_acc_germline"],
                'h_A_auc_somatic': res["h_A_auc_somatic"], 'h_A_sens_somatic': res["h_A_sens_somatic"],
                'h_A_spec_somatic': res["h_A_spec_somatic"],'h_A_acc_somatic': res["h_A_acc_somatic"],
                'Z_acc': res["Z_acc"], 'Z_sens': res["Z_sens"], 'Z_spec': res["Z_spec"],
                'I_acc': res["I_acc"], 'I_sens': res["I_sens"], 'I_spec': res["I_spec"],
                'error': None}
        except Exception as e:
            return {
                'gamma': gamma,'n_reads': n_reads,'n_snvs': n_snvs,
                'n_informative_snvs_g': (hap_ratio[0] + hap_ratio[1]) * n_snvs,
                'n_informative_snvs_s': (hap_ratio[2] + hap_ratio[3]) * n_snvs,
                'allele_pct': allele_pct,
                'hapA_ratio': hap_ratio[0],'hapB_ratio': hap_ratio[1],
                'hapA_ratio_somatic': hap_ratio[2], 'hapB_ratio_somatic': hap_ratio[3],
                'alpha':None, 'alpha_diff': None,
                'h_A_auc': None, 'h_A_sens': None, 'h_A_spec': None, 'h_A_acc': None,
                'h_A_auc_germline': None, 'h_A_sens_germline': None, 'h_A_spec_germline': None, 'h_A_acc_germline': None,
                'h_A_auc_somatic': None, 'h_A_sens_somatic': None, 'h_A_spec_somatic': None, 'h_A_acc_somatic': None,
                'Z_acc': None, 'Z_sens': None, 'Z_spec': None,
                'I_acc': None, 'I_sens': None, 'I_spec': None,
                'error': str(e)}
    #param_grid = list(product(gamma_list, n_reads_list, n_snvs_list, allele_pct_list, hap_ratio_list, somatic_vaf_list))
    param_grid = [
        (gamma, n_reads, n_snvs, allele_pct, hap_ratio, somatic_vaf)
        for gamma in gamma_list
        for n_reads in n_reads_list
        for n_snvs in n_snvs_list
        for allele_pct in allele_pct_list
        for hap_ratio in hap_ratio_list
        for somatic_vaf in ([somatic_vaf_list[0]] if hap_ratio[2] + hap_ratio[3] == 0 else somatic_vaf_list)]

    results = Parallel(n_jobs=-1)(delayed(safe_run)(g, r, s, a, h, v,
                                                    max_iter, verbose, seed, clip) for g, r, s, a, h, v in param_grid)
    df = pd.DataFrame(results)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_folder,
                                   f"simulation_results_{timestamp}_{seed}.csv")
        df.to_csv(output_file, index=False)
    return df


