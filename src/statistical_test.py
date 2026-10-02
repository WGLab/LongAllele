import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
from src.inference import *
import numpy as np
from scipy.optimize import minimize_scalar


def observed_loglikelihood(df_r, df_pi, alpha, h_A, h_m, tiny=1e-300, kept_mask = None):
    if kept_mask is None:
        kept_mask = np.ones(df_r.shape[1], dtype=bool)
    h_A = np.asarray(h_A, float)
    h_m = np.asarray(h_m, float)
    df_r_reduced = df_r.loc[:, kept_mask]
    df_pi_reduced = df_pi.loc[:, kept_mask]
    h_A_reduced = h_A[kept_mask]
    h_m_reduced = h_m[kept_mask]
    prepared = prepare_em_inputs(df_r_reduced, df_pi_reduced)
    r = prepared['r_array']
    pi = prepared['pi_array']
    valid_mask = prepared['valid_mask']
    noise_probs = prepared['noise_probs']
    N, M = r.shape
    emit_I1, emit_I0 = emission_probs(r, pi, h_A_reduced, valid_mask)
    prob_I1 = h_m_reduced[None, :] * emit_I1 + (1.0 - h_m_reduced[None, :]) * noise_probs
    prob_I0 = h_m_reduced[None, :] * emit_I0 + (1.0 - h_m_reduced[None, :]) * noise_probs

    log_I1 = np.zeros((N, M), dtype=float)
    log_I1[valid_mask] = np.log(np.maximum(prob_I1[valid_mask], tiny))
    log_I0 = np.zeros((N, M), dtype=float)
    log_I0[valid_mask] = np.log(np.maximum(prob_I0[valid_mask], tiny))

    logLi_I1 = log_I1.sum(axis=1)
    logLi_I0 = log_I0.sum(axis=1)
    la, lb = np.log(np.clip(alpha, tiny, 1.0)), np.log(np.clip(1.0 - alpha, tiny, 1.0))
    m = np.maximum(la + logLi_I1, lb + logLi_I0)
    ll = m + np.log(np.exp(la + logLi_I1 - m) + np.exp(lb + logLi_I0 - m))
    return float(np.sum(ll))


def run_em_fixed_alpha(df_r, df_pi, max_iter=30, tol=1e-3, verbose=True, seed = None, alpha_fixed=0.5,
           heterozygous_priors = (0.4, 0.2, 0.4), heterozygous_coverage_factor = 1, kept_mask = None,
           h_m_init = None, h_A_init = None,
           het_beta = None, shrink_denominator = 'gene_reads'):
    N0, M0 = df_r.shape
    if kept_mask is None:
        kept_mask = np.ones(M0, dtype=bool)

    alpha = alpha_fixed
    if h_A_init is not None:
        h_A_full = np.asarray(h_A_init, dtype=float).reshape(-1)
        if h_A_full.shape[0] != M0:
            raise ValueError(f"h_A_init length {h_A_full.shape[0]} does not match "
                             f"number of SNVs {M0}.")
        if not np.isfinite(h_A_full).all():
            raise ValueError("h_A_init contains non-finite values")
        h_A_full = h_A_full.copy()
    else:
        h_A_full = np.array(concurrence_to_haplo_init(df_r, seed)) if M0 > 1 else np.ones([1])


    if h_m_init is not None:
        _hm = np.asarray(h_m_init, dtype=float).reshape(-1)
        if _hm.shape[0] != M0:
            raise ValueError(f"h_m_init length {_hm.shape[0]} does not match "
                             f"number of SNVs {M0}.")
        if not np.isfinite(_hm).all():
            raise ValueError("h_m_init contains non-finite values; they would "
                             "reach the E-step as a NaN prior and turn every "
                             "downstream number into NaN with no error")


        h_m_full = _hm.copy()
    else:
        h_m_full = np.array(compute_P_het(
            df_r, df_pi, priors=heterozygous_priors,
            coverage_factor=heterozygous_coverage_factor,
            het_beta=het_beta, shrink_denominator=shrink_denominator))
    df_r_reduced = df_r.loc[:, kept_mask]
    df_pi_reduced = df_pi.loc[:, kept_mask]
    h_m = h_m_full[kept_mask]
    h_A = h_A_full[kept_mask]
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


        h_m_new = np.array(hat_Z)


        h_A_new, _uncov = m_step_h_A_vectorized(
            prepared['r_array'], prepared['pi_array'],
            prepared['valid_mask'], hat_I, hat_Z, noise_probs, h_A)
        for j in _uncov:
            res = minimize_scalar(Qj_objective, bounds=(0.0, 1.0), method='bounded',
                                  args=(j, df_r_reduced, df_pi_reduced, hat_I, hat_Z, noise_probs,
                                        prepared['r_array'], prepared['pi_array'], prepared['snv_observed_indices']))
            h_A_new[j] = res.x if res.success else h_A[j]
        Q_total = Q_total_objective(
            h_A_new, df_r_reduced, df_pi_reduced, hat_I, hat_Z, noise_probs,
            prepared['r_array'], prepared['pi_array'], prepared['valid_mask'])

        delta_param = max(np.max(np.abs(h_m - h_m_new)),np.max(np.abs(h_A - h_A_new)))
        delta_q = abs(Q_total - prev_Q) if prev_Q is not None else float("inf")
        if verbose:
            print(f"Max parameter change: {float(delta_param):.6f}")
            print(f"Q objective change:   {float(delta_q):.6f}")
            print(f"Total Q:              {-float(Q_total):.6f}")
        h_m, h_A, prev_Q = h_m_new, h_A_new, Q_total
        if delta_param < tol or delta_q < tol:
            if verbose:
                print("Convergence reached.")
            break
    h_m_full = np.zeros(M0, dtype=float)
    h_A_full = np.ones(M0, dtype=float) * 0.5
    h_m_full[kept_mask] = h_m
    h_A_full[kept_mask] = h_A
    hat_Z_full_binary = np.zeros(M0, dtype=int)
    hat_Z_full_binary[kept_mask] = (h_m > 0.5).astype(int)
    hat_I_binary = (np.array(hat_I) > 0.5).astype(int)
    return {"alpha": min(np.mean(hat_I), 1 - np.mean(hat_I)),
            "h_A": h_A_full,"h_m": h_m_full,"hat_I": hat_I,
            "hat_I_binary": hat_I_binary, "hat_Z_binary": hat_Z_full_binary,


            "reads_keep_mask": np.ones(N0, dtype=bool),
            "kept_mask": kept_mask,'iteration': iteration + 1}
