import pandas as pd
import numpy as np
from scipy.special import betaln as _betaln
from scipy.special import logsumexp as _logsumexp
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

def linkage_agreement(df_r, min_shared=3, statistic='max'):
    r = coerce_r_codes(df_r)
    alt = (r == ALT_CODE).astype(np.int32)
    ref = (r == REF_CODE).astype(np.int32)
    W = (alt.T @ alt + ref.T @ ref - alt.T @ ref - ref.T @ alt).astype(float)
    np.fill_diagonal(W, 0.0)
    valid = (alt + ref).astype(np.int32)
    n_shared = (valid.T @ valid).astype(float)
    np.fill_diagonal(n_shared, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        s_pair = np.where(n_shared > 0, np.abs(W) / n_shared, 0.0)
    if statistic not in ('max', 'mean'):
        raise ValueError(f"statistic must be 'max' or 'mean', got {statistic!r}")
    eligible = n_shared >= min_shared
    n_partner = eligible.sum(axis=1)
    if statistic == 'max':
        s_marker = np.where(eligible, s_pair, 0.0).max(axis=1) if s_pair.size \
            else np.zeros(s_pair.shape[0])
    else:
        tot = np.where(eligible, n_shared, 0.0).sum(axis=1)
        num = np.where(eligible, np.abs(W), 0.0).sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            s_marker = np.where(tot > 0, num / np.maximum(tot, 1e-12), 0.0)
    return s_pair, n_shared, np.clip(s_marker, 0.0, 1.0), n_partner


def linkage_loglr(df_r, df_pi=None, min_shared=3, eps_floor=0.001,
                  eps_cap=0.25):
    from scipy.stats import binom
    r = coerce_r_codes(df_r)
    alt = (r == ALT_CODE).astype(np.int32)
    ref = (r == REF_CODE).astype(np.int32)
    W = (alt.T @ alt + ref.T @ ref - alt.T @ ref - ref.T @ alt).astype(float)
    valid = (alt + ref).astype(np.int32)
    n = (valid.T @ valid).astype(float)
    np.fill_diagonal(W, 0.0)
    np.fill_diagonal(n, 0.0)
    k = (W + n) / 2.0

    if df_pi is None:
        eps = np.full(n.shape, 0.02)
    else:
        pi = np.asarray(df_pi, dtype=float)
        pi = np.where(np.isfinite(pi), pi, 0.0)
        mean_pi = (valid * pi).sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
        eps = mean_pi[:, None] + mean_pi[None, :]
    eps = np.clip(eps, eps_floor, eps_cap)

    with np.errstate(divide='ignore', invalid='ignore'):
        l1 = np.logaddexp(np.log(0.5) + binom.logpmf(k, n, 1 - eps),
                          np.log(0.5) + binom.logpmf(k, n, eps))
        l0 = binom.logpmf(k, n, 0.5)
        loglr = l1 - l0
    eligible = (n >= min_shared) & np.isfinite(loglr)
    loglr = np.where(eligible, loglr, -np.inf)
    np.fill_diagonal(loglr, -np.inf)
    best = loglr.max(axis=1)
    best = np.where(np.isfinite(best), best, 0.0)
    n_partner = (n >= min_shared).sum(axis=1)
    return np.where(eligible, loglr, 0.0), n, best, n_partner


def linkage_n_partners(df_r, min_shared=3):
    r = coerce_r_codes(df_r)
    valid = ((r == ALT_CODE) | (r == REF_CODE)).astype(np.int32)
    n = (valid.T @ valid).astype(float)
    np.fill_diagonal(n, 0.0)
    return (n >= min_shared).sum(axis=1)


def signed_haplo_init(df_r, seed=None, link_min_agreement=0.0,
                      link_min_shared=3):
    r = coerce_r_codes(df_r)


    alt = (r == ALT_CODE)
    ref = (r == REF_CODE)
    s_ = (alt.astype(np.float64) - ref.astype(np.float64))
    W = s_.T @ s_
    np.fill_diagonal(W, 0.0)
    if link_min_agreement > 0.0:


        valid = (alt | ref).astype(np.float64)
        n_shared = valid.T @ valid
        np.fill_diagonal(n_shared, 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            agree = np.where(n_shared > 0, np.abs(W) / n_shared, 0.0)
        W = np.where((n_shared >= link_min_shared)
                     & (agree >= link_min_agreement), W, 0.0)
    M = W.shape[0]
    ha0 = np.full(M, 0.5, dtype=float)
    weak = np.zeros(M, dtype=bool)
    from scipy.sparse.csgraph import connected_components
    _n, labels = connected_components(W != 0.0, directed=False)
    for c in np.unique(labels):
        m = labels == c
        if m.sum() == 1:
            weak[m] = True
            continue
        _vals, vecs = np.linalg.eigh(W[np.ix_(m, m)])
        v = vecs[:, -1]
        sub = np.where(v >= 0, 0.9, 0.1).astype(float)
        tie = np.abs(v) < 1e-8
        sub[tie] = 0.5
        ha0[m] = sub
        weak[np.flatnonzero(m)[tie]] = True
    if weak.any():
        rng = np.random.default_rng(seed)
        ha0[weak] = 0.5 + rng.uniform(-1e-3, 1e-3, size=int(weak.sum()))
    return ha0


def concurrence_to_haplo_init(df_r, seed=None):


    n_snvs = df_r.shape[1]

    binary_alt = (coerce_r_codes(df_r) == ALT_CODE).astype(np.int32, copy=False)
    concurrence_matrix = binary_alt.T @ binary_alt

    clustering = SpectralClustering(n_clusters=2, affinity='precomputed',
                                    assign_labels='kmeans', random_state=seed)
    labels = clustering.fit_predict(concurrence_matrix)

    scores = np.zeros(n_snvs)
    for i in range(n_snvs):
        cluster_i = labels[i]
        in_cluster = np.where(labels == cluster_i)[0]
        if len(in_cluster) > 1:
            scores[i] = (concurrence_matrix[i, in_cluster].sum() -  concurrence_matrix[i, i] )/ (len(in_cluster) - 1)
        else:
            scores[i] = 0

    median_score = np.median(scores)
    keep = scores >= median_score

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


def compute_P_het(df_r, df_pi, priors=(0.4, 0.2, 0.4), coverage_factor=1,
                  het_beta=None, shrink_denominator='gene_reads') -> list:
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
        logL[0] += np.log(np.maximum(np.where(ref_mask, 1.0 - pi_array, 1.0), eps)).sum(axis=0)
        logL[2] += np.log(np.maximum(np.where(ref_mask, pi_array / 3.0, 1.0), eps)).sum(axis=0)
        if het_beta is None:
            logL[1] += np.log(np.maximum(np.where(ref_mask, 0.5 * (1.0 - pi_array) + 0.5 * (pi_array / 3.0), 1.0), eps)).sum(axis=0)

    if np.any(alt_mask):
        logL[0] += np.log(np.maximum(np.where(alt_mask, pi_array / 3.0, 1.0), eps)).sum(axis=0)
        logL[2] += np.log(np.maximum(np.where(alt_mask, 1.0 - pi_array, 1.0), eps)).sum(axis=0)
        if het_beta is None:
            logL[1] += np.log(np.maximum(np.where(alt_mask, 0.5 * (1.0 - pi_array) + 0.5 * (pi_array / 3.0), 1.0), eps)).sum(axis=0)

    if np.any(other_mask):
        other_term = np.log(np.maximum(np.where(other_mask, pi_array / 3.0, 1.0), eps)).sum(axis=0)
        logL += other_term[None, :]

    site_depths = valid_mask.sum(axis=0)
    if shrink_denominator == 'gene_reads':
        base = float(n_gene_reads)
        coverage_fracs = (site_depths / base if base > 0
                          else np.zeros(r_array.shape[1], dtype=float))
    elif shrink_denominator == 'deepest_site':
        base = float(site_depths.max()) if site_depths.size else 0.0
        coverage_fracs = (site_depths / base if base > 0
                          else np.zeros(r_array.shape[1], dtype=float))
    else:
        raise ValueError(f"shrink_denominator must be 'gene_reads' or "
                         f"'deepest_site', got {shrink_denominator!r}")
    coverage_shrinkage = np.minimum(1.0, np.maximum(0.0001, coverage_fracs ** coverage_factor))

    if het_beta is not None:


        a_, b_ = float(het_beta[0]), float(het_beta[1])
        x, w = np.polynomial.legendre.leggauss(48)
        pgrid, wgrid = 0.5 * (x + 1), 0.5 * w
        n_alt = alt_mask.sum(axis=0).astype(float)
        n_ref = ref_mask.sum(axis=0).astype(float)
        lp = coverage_shrinkage[None, :] * (
            n_alt[None, :] * np.log(pgrid)[:, None]
            + n_ref[None, :] * np.log(1.0 - pgrid)[:, None])
        lprior = ((a_ - 1) * np.log(pgrid) + (b_ - 1) * np.log(1.0 - pgrid)
                  - _betaln(a_, b_) + np.log(wgrid))[:, None]
        logL_het_marginal = _logsumexp(lp + lprior, axis=0)


        log_post = log_prior[:, None] + coverage_shrinkage[None, :] * logL
        log_post[1] = log_prior[1] + logL_het_marginal
    else:
        log_post = log_prior[:, None] + coverage_shrinkage[None, :] * logL
    log_total = np.logaddexp.reduce(log_post, axis=0)
    p_het = np.exp(log_post[1] - log_total)
    p_het = p_het.astype(float, copy=False)

    zero_depth = site_depths == 0
    if np.any(zero_depth):
        p_het[zero_depth] = float(priors[1] / sum(priors))

    return p_het.tolist()


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

    emit_I1, emit_I0, logLi_I1, logLi_I0 = _read_loglik_by_hap(
        r_array, pi_array, h_A, h_m, noise_probs, valid_mask)

    log_post_I1 = log_alpha + logLi_I1
    log_post_I0 = log_1malpha + logLi_I0
    log_norm_I = np.logaddexp(log_post_I1, log_post_I0)
    hat_I = np.exp(log_post_I1 - log_norm_I)

    emit_Ihat = hat_I[:, None] * emit_I1 + (1.0 - hat_I[:, None]) * emit_I0
    prob_Z1 = emit_Ihat
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

def _read_loglik_by_hap(r_array, pi_array, h_A, h_m, noise_probs, valid_mask):
    N, M = r_array.shape
    tiny = 1e-300
    w = h_m[None, :]
    emit_I1, emit_I0 = emission_probs(r_array, pi_array, h_A, valid_mask)

    prob_I1 = w * emit_I1 + (1.0 - w) * noise_probs
    prob_I0 = w * emit_I0 + (1.0 - w) * noise_probs

    log_prob_I1 = np.zeros((N, M), dtype=float)
    log_prob_I1[valid_mask] = np.log(np.maximum(prob_I1[valid_mask], tiny))
    log_prob_I0 = np.zeros((N, M), dtype=float)
    log_prob_I0[valid_mask] = np.log(np.maximum(prob_I0[valid_mask], tiny))

    return emit_I1, emit_I0, log_prob_I1.sum(axis=1), log_prob_I0.sum(axis=1)


def assign_reads_frozen_markers(df_r, df_pi, results, max_iter=50, tol=1e-3,
                                filter_reads=True):
    N0, M0 = df_r.shape
    keep_mask = np.asarray(results['kept_mask'], dtype=bool)
    h_A_full = np.asarray(results['h_A'], dtype=float)
    h_m_full = np.asarray(results['h_m'], dtype=float)
    df_r_col = df_r.loc[:, keep_mask]
    df_pi_col = df_pi.loc[:, keep_mask]
    if filter_reads:
        reads_keep_mask = (coerce_r_codes(df_r_col) != MISSING_CODE).any(axis=1)
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
            "h_A_init_used": h_A_full.copy(), "h_m_init_used": h_m_full.copy(),
            "iteration": 0}
    df_r_red = df_r_col.loc[reads_keep_mask]
    df_pi_red = df_pi_col.loc[reads_keep_mask]
    prepared = prepare_em_inputs(df_r_red, df_pi_red)
    h_A = h_A_full[keep_mask]
    h_m = h_m_full[keep_mask]

    _, _, logLi_I1, logLi_I0 = _read_loglik_by_hap(
        prepared['r_array'], prepared['pi_array'], h_A, h_m,
        prepared['noise_probs'], prepared['valid_mask'])

    alpha = 0.5
    iteration = 0
    for iteration in range(max_iter):
        log_alpha = np.log(alpha) if alpha > 0 else -np.inf
        log_1malpha = np.log(1 - alpha) if alpha < 1 else -np.inf
        log_post_I1 = log_alpha + logLi_I1
        log_post_I0 = log_1malpha + logLi_I0
        hat_I = np.exp(log_post_I1 - np.logaddexp(log_post_I1, log_post_I0))
        alpha_new = float(np.mean(hat_I))
        delta = abs(alpha - alpha_new)
        alpha = alpha_new
        if delta < tol:
            break
    h_m_out = np.zeros(M0, dtype=float)
    h_A_out = np.ones(M0, dtype=float) * 0.5
    h_m_out[keep_mask] = h_m
    h_A_out[keep_mask] = h_A
    hat_Z_full_binary = np.zeros(M0, dtype=int)
    hat_Z_full_binary[keep_mask] = (h_m > 0.5).astype(int)
    hat_I_full = np.full(N0, np.mean(hat_I), dtype=float)
    hat_I_full[reads_keep_mask] = np.asarray(hat_I, dtype=float)
    return {"alpha": min(np.mean(hat_I), 1 - np.mean(hat_I)),
            "h_A": h_A_out, "h_m": h_m_out, "hat_I": hat_I_full,
            "hat_I_binary": (hat_I_full > 0.5).astype(int),
            "hat_Z_binary": hat_Z_full_binary,
            "kept_mask": keep_mask, "reads_keep_mask": reads_keep_mask,
            "h_A_init_used": h_A_full.copy(), "h_m_init_used": h_m_full.copy(),
            'iteration': iteration + 1}


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
    prob = hat_Zj * emit + (1.0 - hat_Zj) * noisej
    return -np.sum(np.log(np.maximum(prob, tiny)))

def m_step_h_A_vectorized(r_array, pi_array, valid_mask, hat_I, hat_Z,
                          noise_probs, h_A_prev, grad_tol=1e-10,
                          max_newton=60):
    I = np.asarray(hat_I, dtype=float)[:, None]
    Z = np.asarray(hat_Z, dtype=float)[None, :]
    pi = pi_array
    c1 = 4.0 * pi / 3.0 - 1.0
    u = np.zeros_like(pi, dtype=float)
    v = np.zeros_like(pi, dtype=float)
    ref = valid_mask & (r_array == REF_CODE)
    alt = valid_mask & (r_array == ALT_CODE)
    other = valid_mask & (r_array == OTHER_CODE)
    u[ref] = (I * (1.0 - pi) + (1.0 - I) * (pi / 3.0))[ref]
    v[ref] = (c1 * (2.0 * I - 1.0))[ref]
    u[alt] = (I * (pi / 3.0) + (1.0 - I) * (1.0 - pi))[alt]
    v[alt] = (c1 * (1.0 - 2.0 * I))[alt]
    u[other] = (2.0 * pi / 3.0)[other]

    A = np.where(valid_mask, Z * u + (1.0 - Z) * noise_probs, 1.0)
    B = np.where(valid_mask, Z * v, 0.0)

    tiny = 1e-300
    M = A.shape[1]

    def fprime(h_row, cols=slice(None)):

        denom = A[:, cols] + B[:, cols] * h_row[None, :]
        ratio = B[:, cols] / np.maximum(denom, tiny)
        return -ratio.sum(axis=0), (ratio * ratio).sum(axis=0)

    h_new = np.clip(np.asarray(h_A_prev, dtype=float).copy(), 0.0, 1.0)


    with np.errstate(invalid='ignore'):
        col_ok = (np.isfinite(A).all(axis=0) & np.isfinite(B).all(axis=0)
                  & (A >= 0.0).all(axis=0) & ((A + B) >= 0.0).all(axis=0))

    flat = (B == 0.0).all(axis=0)
    fp0, _ = fprime(np.zeros(M))
    fp1, _ = fprime(np.ones(M))

    solvable = col_ok & ~flat
    at_zero = solvable & (fp0 >= 0.0)
    at_one = solvable & (fp1 <= 0.0)
    h_new[at_zero] = 0.0
    h_new[at_one & ~at_zero] = 1.0
    interior = solvable & ~(at_zero | at_one)
    cols = np.flatnonzero(interior)
    if cols.size:
        lo = np.zeros(cols.size)
        hi = np.ones(cols.size)
        h = np.clip(h_new[cols], 1e-6, 1 - 1e-6)
        converged = np.zeros(cols.size, dtype=bool)
        for _ in range(max_newton):
            fp, fpp = fprime(h, cols)
            newly = np.abs(fp) <= grad_tol
            converged |= newly
            if converged.all():
                break

            lo = np.where(fp < 0.0, h, lo)
            hi = np.where(fp > 0.0, h, hi)
            step = fp / np.maximum(fpp, tiny)
            h_next = h - step


            bad = ~np.isfinite(h_next) | (h_next <= lo) | (h_next >= hi)
            h_next = np.where(bad, 0.5 * (lo + hi), h_next)
            h = np.where(converged, h, h_next)

        converged |= (hi - lo) < 1e-12
        h_new[cols] = h
        unconverged = cols[~converged]
    else:
        unconverged = np.zeros(0, dtype=int)
    bad_cols = np.flatnonzero(~col_ok)
    if bad_cols.size:
        unconverged = np.union1d(unconverged, bad_cols)
    return h_new, unconverged


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
    prob = hat_Z[None, :] * emit_Ihat + (1.0 - hat_Z[None, :]) * noise_probs

    log_p = np.zeros_like(prob, dtype=float)
    log_p[valid_mask] = np.log(np.maximum(prob[valid_mask], tiny))
    total = -log_p.sum()
    if not np.isfinite(total):
        return np.inf
    return total


def _classifier_gap_filter(h_m, gap_tau=0.10, min_keep=1):
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

    return np.ones(m, dtype=bool)


def adaptive_keep_mask(h_m, min_abs=0.90, gap_tau=0.10, fallback_q=0.50,
                       min_keep=1, max_keep=None):
    h = np.asarray(h_m, dtype=float)
    m = len(h)
    if m == 0:
        return np.zeros(0, dtype=bool)

    keep = h >= min_abs
    if keep.any():

        if max_keep is not None and keep.sum() > max_keep:

            idx = np.argsort(-h[keep])[:max_keep]
            mask = np.zeros_like(keep)
            mask[np.where(keep)[0][idx]] = True
            keep = mask

        if keep.sum() < min_keep:
            top_idx = np.argsort(-h)[:min_keep]
            keep[:] = False
            keep[top_idx] = True
        return keep

    order = np.argsort(-h)
    hs = h[order]
    if m >= 2:
        gaps = hs[:-1] - hs[1:]
        k = int(np.argmax(gaps))
        if gaps[k] >= gap_tau:
            thr = 0.5 * (hs[k] + hs[k+1])
            keep = h >= thr

            if max_keep is not None and keep.sum() > max_keep:
                keep[:] = False
                keep[order[:max_keep]] = True
            if keep.sum() < min_keep:
                keep[:] = False
                keep[order[:min_keep]] = True
            return keep

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
           filter_reads = True, results = None, h_m_init = None, gap_tau = 0.10,
           init_method = 'signed', link_min_agreement = 0.0,
           link_min_shared = 3, h_m_init_cleaned = True,
           het_beta = None, shrink_denominator = 'gene_reads'):
    N0, M0 = df_r.shape
    if init_method not in ('signed', 'concurrence'):
        raise ValueError(f"init_method must be 'signed' or 'concurrence', "
                         f"got {init_method!r}")
    if link_min_agreement and init_method != 'signed':
        raise ValueError('link_min_agreement only applies to the signed init; '
                         f'got init_method={init_method!r}')
    init_fn = (signed_haplo_init if init_method == 'signed'
               else concurrence_to_haplo_init)

    alpha = 0.5
    if results is None:
        if M0 > 1:
            h_A_full = np.array(
                init_fn(df_r, seed, link_min_agreement=link_min_agreement,
                        link_min_shared=link_min_shared)
                if init_method == 'signed' else init_fn(df_r, seed))
        else:
            h_A_full = np.ones([1])
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
        h_m_full = np.array(compute_P_het(
            df_r, df_pi, priors=heterozygous_priors,
            coverage_factor=heterozygous_coverage_factor,
            het_beta=het_beta, shrink_denominator=shrink_denominator))
    if results is not None:
        keep_mask = results['kept_mask']
    elif (M0 > 5) and h_m_filter:
        if h_m_init is not None and h_m_init_cleaned:

            keep_mask = _classifier_gap_filter(h_m_full, gap_tau=gap_tau, min_keep=4)
        else:


            keep_mask = adaptive_keep_mask(h_m_full, min_abs=0.90, gap_tau=gap_tau,
                                           fallback_q=0.50, min_keep=4)
    else:
        keep_mask = np.ones(M0, dtype=bool)


    h_A_init_used, h_m_init_used = h_A_full.copy(), h_m_full.copy()
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
            "h_A_init_used": h_A_full.copy(), "h_m_init_used": h_m_full.copy(),
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

        hat_I, hat_Z = e_step(
            df_r_reduced, df_pi_reduced, alpha, h_A, h_m, noise_probs,
            r_array=prepared['r_array'], pi_array=prepared['pi_array'],
            valid_mask=prepared['valid_mask'])
        hat_Z = hat_Z if results is None else results['hat_Z_binary'][keep_mask]


        alpha_new = np.mean(hat_I)

        h_m_new = np.array(hat_Z) if results is None else results['h_m'][keep_mask]


        if results is None:
            h_A_new, _uncov = m_step_h_A_vectorized(
                prepared['r_array'], prepared['pi_array'],
                prepared['valid_mask'], hat_I, hat_Z, noise_probs, h_A)
            for j in _uncov:
                res = minimize_scalar(Qj_objective, bounds=(0.0, 1.0), method='bounded',
                                      args=(j, df_r_reduced, df_pi_reduced, hat_I, hat_Z, noise_probs,
                                            prepared['r_array'], prepared['pi_array'], prepared['snv_observed_indices']))
                h_A_new[j] = res.x if res.success else h_A[j]
        else:
            h_A_new = results['h_A'][keep_mask]
        Q_total = Q_total_objective(
            h_A_new, df_r_reduced, df_pi_reduced, hat_I, hat_Z, noise_probs,
            prepared['r_array'], prepared['pi_array'], prepared['valid_mask'])

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
    h_m_full = np.zeros(M0, dtype=float)
    h_A_full = np.ones(M0, dtype=float) * 0.5
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
            "h_A_init_used": h_A_init_used, "h_m_init_used": h_m_init_used,
            'iteration': iteration + 1}


def guarded_switch_flip(df_r, df_pi, results, positions, k_bridge=2,
                        net_margin=2, ll_eps=1e-6, polish_iters=4):
    from src.statistical_test import observed_loglikelihood
    info = {'flip_applied': False, 'flip_gap_pos': None, 'flip_dll': None}
    kept = results['kept_mask']
    reads_keep = results['reads_keep_mask']
    h_A_full, h_m_full = results['h_A'], results['h_m']
    N0, M0 = df_r.shape
    positions = np.asarray(positions)
    if positions.shape != (M0,):
        raise ValueError(f'positions must have shape ({M0},), '
                         f'got {positions.shape}')
    for name, arr, shape in (('kept_mask', kept, (M0,)),
                             ('reads_keep_mask', reads_keep, (N0,)),
                             ('h_A', h_A_full, (M0,)),
                             ('h_m', h_m_full, (M0,))):
        if np.shape(arr) != shape:
            raise ValueError(f'results[{name!r}] must have shape {shape}, '
                             f'got {np.shape(arr)}')
    if reads_keep.sum() == 0 or not np.isfinite(results['alpha']):
        return results, info


    alpha0 = float(np.mean(np.asarray(results['hat_I'])[reads_keep]))
    df_r_red = df_r.loc[:, kept].loc[reads_keep]
    df_pi_red = df_pi.loc[:, kept].loc[reads_keep]
    h_A = h_A_full[kept].copy()
    h_m = h_m_full[kept].copy()
    pos_red = positions[kept]
    phased = (h_m > 0.5) & (h_A != 0.5)
    idx = np.flatnonzero(phased)
    if len(idx) < 2:
        return results, info
    idx = idx[np.argsort(pos_red[idx])]
    prep = prepare_em_inputs(df_r_red, df_pi_red)
    r = prep['r_array']

    def polish(hA_vec):
        a, hA, hm = alpha0, hA_vec.copy(), h_m.copy()
        with np.errstate(all='ignore'):
            for _ in range(polish_iters):
                hat_I, hat_Z = e_step(df_r_red, df_pi_red, a, hA, hm,
                                      prep['noise_probs'],
                                      r_array=prep['r_array'],
                                      pi_array=prep['pi_array'],
                                      valid_mask=prep['valid_mask'])
                a = float(np.mean(hat_I))
                hm = np.asarray(hat_Z, dtype=float)
                hA, _ = m_step_h_A_vectorized(
                    prep['r_array'], prep['pi_array'], prep['valid_mask'],
                    hat_I, hat_Z, prep['noise_probs'], hA)
            ll = observed_loglikelihood(df_r_red, df_pi_red, a, hA, hm)
        return ll, a, hA, hm, hat_I, hat_Z


    if len(idx) >= np.iinfo(np.int16).max:
        raise ValueError('too many phased markers for int16 vote accumulation')
    rp = r[:, idx]
    C = np.zeros(rp.shape, dtype=np.int16)
    ori = (h_A[idx] > 0.5)
    C[(rp == ALT_CODE) & ori[None, :]] = 1
    C[(rp == ALT_CODE) & ~ori[None, :]] = -1
    C[(rp == REF_CODE) & ~ori[None, :]] = 1
    C[(rp == REF_CODE) & ori[None, :]] = -1
    inf_mask = ((rp == ALT_CODE) | (rp == REF_CODE)).astype(np.int16)
    cum_c = np.cumsum(C, axis=1, dtype=np.int16)
    cum_i = np.cumsum(inf_mask, axis=1, dtype=np.int16)
    del C, inf_mask
    tot_c = cum_c[:, -1:]
    tot_i = cum_i[:, -1:]
    n_gaps = len(idx) - 1
    left_s = cum_c[:, :n_gaps]
    right_s = tot_c - left_s
    bridge = (cum_i[:, :n_gaps] > 0) & ((tot_i - cum_i[:, :n_gaps]) > 0)
    votes = np.sign(left_s) * np.sign(right_s) * bridge
    widths = bridge.sum(axis=0)
    nets = votes.sum(axis=0)
    del rp, cum_c, cum_i, left_s, right_s, bridge, votes
    qualified = np.flatnonzero((widths >= k_bridge) & (-nets >= net_margin))
    if qualified.size == 0:
        return results, info


    ll_cur = polish(h_A)[0]
    best = None
    for g in qualified:
        hA_flip = h_A.copy()
        hA_flip[idx[g + 1:]] = 1.0 - hA_flip[idx[g + 1:]]
        dll = polish(hA_flip)[0] - ll_cur
        if dll > ll_eps and (best is None or dll > best[1]):
            best = (int(g), dll, hA_flip)
    if best is None:
        return results, info
    g, dll, hA_flip = best
    _ll, a, hA, hm, hat_I, hat_Z = polish(hA_flip)

    h_m_new = np.zeros(M0, dtype=float)
    h_A_new = np.ones(M0, dtype=float) * 0.5
    h_m_new[kept] = hm
    h_A_new[kept] = hA
    hat_Z_binary = np.zeros(M0, dtype=int)
    hat_Z_binary[kept] = (hm > 0.5).astype(int)
    hat_I_full = np.full(N0, float(np.mean(hat_I)), dtype=float)
    hat_I_full[reads_keep] = np.asarray(hat_I, dtype=float)
    results = dict(results)
    results.update({
        'alpha': min(float(np.mean(hat_I)), 1 - float(np.mean(hat_I))),
        'h_A': h_A_new, 'h_m': h_m_new, 'hat_I': hat_I_full,
        'hat_I_binary': (hat_I_full > 0.5).astype(int),
        'hat_Z_binary': hat_Z_binary})
    info.update({'flip_applied': True,
                 'flip_gap_pos': int(pos_red[idx[g]]),
                 'flip_dll': float(dll)})
    return results, info


def assign_phase_blocks(df_r, results, positions, min_shared=1,
                        min_agreement=0.0):
    kept = results['kept_mask']
    reads_keep = results['reads_keep_mask']
    h_A_full, h_m_full = results['h_A'], results['h_m']
    N0, M0 = df_r.shape
    positions = np.asarray(positions)
    if positions.shape != (M0,):
        raise ValueError(f'positions must have shape ({M0},), '
                         f'got {positions.shape}')
    out = np.zeros(M0, dtype=int)
    if reads_keep.sum() == 0:
        return out
    h_A = h_A_full[kept]
    h_m = h_m_full[kept]
    phased = (h_m > 0.5) & (h_A != 0.5)
    idx = np.flatnonzero(phased)
    kept_indices = np.flatnonzero(kept)
    if len(idx) == 0:
        return out
    if len(idx) == 1:
        out[kept_indices[idx[0]]] = 1
        return out
    pos_red = positions[kept]
    idx = idx[np.argsort(pos_red[idx])]
    r = coerce_r_codes(df_r.loc[:, kept].loc[reads_keep])
    rp = r[:, idx]
    inf_mask = ((rp == ALT_CODE) | (rp == REF_CODE)).astype(np.int32)
    shared = inf_mask.T @ inf_mask
    if int(min_shared) < 1:
        raise ValueError(f"min_shared must be >= 1, got {min_shared}: zero would "
                         f"link every pair of markers and make the whole gene "
                         f"one block regardless of the data")
    adj = shared >= int(min_shared)
    if float(min_agreement) > 0:


        alt = (rp == ALT_CODE).astype(np.int32)
        ref = (rp == REF_CODE).astype(np.int32)
        concord = alt.T @ alt + ref.T @ ref
        discord = alt.T @ ref + ref.T @ alt
        with np.errstate(invalid='ignore', divide='ignore'):


            frac = np.maximum(concord, discord) / np.maximum(shared, 1)
        adj = adj & (frac >= float(min_agreement))
    np.fill_diagonal(adj, True)
    adj = adj > 0
    from scipy.sparse.csgraph import connected_components
    _n, labels = connected_components(adj, directed=False)

    seen = {}
    block_ids = np.array([seen.setdefault(int(lab), len(seen) + 1)
                          for lab in labels])
    out[kept_indices[idx]] = block_ids
    return out


def assign_read_blocks(df_r, marker_blocks):
    r = coerce_r_codes(df_r)
    mb = np.asarray(marker_blocks)
    out = np.zeros(r.shape[0], dtype=int)
    informative = ((r == ALT_CODE) | (r == REF_CODE)) & (mb > 0)[None, :]
    for b in np.unique(mb[mb > 0]):
        touch = informative[:, mb == b].any(axis=1)
        out = np.where((out == 0) & touch, b, out)
    return out


def block_aware_alpha_bounds(hat_I, reads_phasable, read_blocks):
    s = np.asarray(hat_I, dtype=float)
    rp = np.asarray(reads_phasable).astype(bool)
    rb = np.asarray(read_blocks)
    T = len(s)
    if T == 0:
        return np.nan, np.nan
    free = rp & (rb == 0)
    lo = hi = float(s[free].sum())
    for b in np.unique(rb[rb > 0]):
        in_b = rp & (rb == b)
        sb = float(s[in_b].sum())
        nb = float(in_b.sum())
        if b == 1:
            lo += sb
            hi += sb
        else:
            lo += min(sb, nb - sb)
            hi += max(sb, nb - sb)
    return lo / T, (hi + float((~rp).sum())) / T


def block_orientation_alpha_bounds(hat_I, reads_phasable, read_blocks):
    s = np.asarray(hat_I, dtype=float)
    rp = np.asarray(reads_phasable).astype(bool)
    rb = np.asarray(read_blocks)
    T = int(rp.sum())
    if T == 0:
        return np.nan, np.nan


    anchored = rp & ((rb == 1) | (rb == 0))
    lo = hi = float(s[anchored].sum())
    for b in np.unique(rb[rb > 1]):
        in_b = rp & (rb == b)
        sb = float(s[in_b].sum())
        nb = float(in_b.sum())
        lo += min(sb, nb - sb)
        hi += max(sb, nb - sb)
    return lo / T, hi / T


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
        return 'ref'

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
    pi_data0 = np.random.uniform(0.01, 0.05, size=(n_reads, n_snvs))
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


    assert sum(hap_ratio) <= 1.0, "hap_ratio must sum to ≤ 1.0"

    n_A = int(n_reads * allele_pct)
    n_B = n_reads - n_A
    true_I = np.array([1] * n_A + [0] * n_B)

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


    df_r, df_pi = simulate_r_pi(true_I, n_snvs, n_reads, gamma, hapA_vars, hapB_vars, hapA_vars_somatic, hapB_vars_somatic, seed, somatic_vaf)

    results = run_em(df_r, df_pi, max_iter=max_iter, tol=tol,verbose=verbose, seed=seed, clip = clip)
    alpha = results["alpha"]
    h_A = np.array(results["h_A"])
    h_m = np.array(results["h_m"])
    hat_I = np.array(results["hat_I"])
    hat_I_binary = np.array(results["hat_I_binary"])
    hat_Z_binary = np.array(results["hat_Z_binary"])

    alpha_diff = min(abs(alpha - allele_pct), abs(1 - alpha - allele_pct))
    true_h_A = np.zeros(len(informative_vars))
    true_h_A[hapA_vars] = 1.0
    true_h_A[hapA_vars_somatic] = 1.0
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
