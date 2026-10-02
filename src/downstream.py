import json
import os
import pickle
import time
import re
import zlib
import pysam
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, binomtest
from scipy.optimize import brentq
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from src.compat import _collapse_legacy_merge_suffixes, resolve_scotch_auxiliary_tsv
from src.utils import canonicalize_read_name as _canonicalize_read_name
from src.utils import path_safe_gene_name as _path_safe_gene_name
from src.utils import Haplotyping as _Hap


def _actv_delta_ase(sum_a, sum_b):
    t = sum_a + sum_b
    return np.nan if t <= 0 else 2.0 * sum_a / t - 1.0


def _actv_delta_astu(dom_a, tot_a, dom_b, tot_b):
    if min(tot_a, tot_b) <= 0:
        return np.nan
    return dom_a / tot_a - dom_b / tot_b


def _actv_range(deltas):
    vals = [v for v in deltas if v == v]
    return np.nan if len(vals) < 2 else float(max(vals) - min(vals))


def _actv_deltas_for_labels(labels_arr, uniq, axis, a, b, dom_a=None,
                            dom_b=None):
    out = []
    for L in uniq:
        m = labels_arr == L
        if axis == 'ase':
            out.append(_actv_delta_ase(float(a[m].sum()), float(b[m].sum())))
        else:
            out.append(_actv_delta_astu(float(dom_a[m].sum()),
                                        float(a[m].sum()),
                                        float(dom_b[m].sum()),
                                        float(b[m].sum())))
    return out


def actv_read_unit_matrices(gene_read_hap, gene_isoform_map):
    d = gene_read_hap[gene_read_hap['reads_phasable'] == 1]


    iso_one = (gene_isoform_map[['Read', 'Isoform']]
               .sort_values(['Read', 'Isoform'], kind='mergesort')
               .drop_duplicates(subset=['Read']))
    d = d.merge(iso_one, on='Read', how='inner')
    if d.empty:
        return None, None
    matA = d.pivot_table(index='Read', columns='Isoform', values='hat_I',
                         aggfunc='sum', fill_value=0.0)
    hib = (d.assign(_hib=1.0 - d['hat_I'])
           .pivot_table(index='Read', columns='Isoform', values='_hib',
                        aggfunc='sum', fill_value=0.0))
    return matA, hib.reindex(index=matA.index, columns=matA.columns,
                             fill_value=0.0)


def actv_gate_flags(gene_snv_df, axis):
    g = gene_snv_df.drop_duplicates(['CellType', 'geneID'])
    g = g[g['CellType'] != 'Bulk']
    if g.empty:
        return {}
    col = 'gene_p_value' if axis == 'ase' else 'isoform_p_value'
    ok = pd.to_numeric(g[col], errors='coerce') <= .05
    return ok.groupby(g['geneID']).any().to_dict()


def actv_gene_axis(matA, matB, label_of, axis, n_perm=300, min_cells=10,
                   seed=42, gene_id='', min_phasable_reads=20,
                   max_attempts=None, unit='cell', expressed_counts=None):
    if axis not in ('ase', 'astu') or unit not in ('cell', 'read'):
        raise ValueError('Invalid ACTV axis or unit')
    if n_perm < 1 or min_cells < 1 or min_phasable_reads < 0:
        raise ValueError('Invalid ACTV permutation count or coverage threshold')
    max_attempts = 10 * n_perm if max_attempts is None else max_attempts
    if max_attempts < n_perm:
        raise ValueError('ACTV max_attempts must be >= n_perm')
    idx = matA.index.union(matB.index)
    A, B = matA.reindex(idx).fillna(0.0), matB.reindex(idx).fillna(0.0)
    A.columns = [c[:-5] if str(c).endswith('_hapA') else c for c in A.columns]
    B.columns = [c[:-5] if str(c).endswith('_hapB') else c for c in B.columns]
    cols = A.columns.union(B.columns)
    A, B = A.reindex(columns=cols).fillna(0.0), B.reindex(columns=cols).fillna(0.0)
    a_tot, b_tot = A.sum(axis=1).to_numpy(float), B.sum(axis=1).to_numpy(float)
    cts = pd.Index(idx).map(label_of)
    use = ((a_tot + b_tot) > 0) & cts.notna()
    a, b = a_tot[use], b_tot[use]
    labels_all = np.asarray(cts[use])
    uniq_all = sorted(pd.unique(labels_all))
    counts = {L: int((labels_all == L).sum()) for L in uniq_all}
    depths = {L: float((a + b)[labels_all == L].sum()) for L in uniq_all}
    cell_counts = counts if expressed_counts is None else expressed_counts
    uniq = [L for L in uniq_all
            if depths[L] >= min_phasable_reads - 1e-8
            and (unit == 'read' or cell_counts.get(L, 0) >= min_cells)]
    out = dict(eligible_cells=False, actv=np.nan, thr95=np.nan, pval=np.nan,
               call=False, test_status='insufficient_contexts',
               n_valid_permutations=0, n_permutation_attempts=0,
               n_invalid_permutations=0,
               qualifying_contexts=json.dumps(uniq),
               n_cells_by_ct=json.dumps(counts, sort_keys=True),
               n_expressing_cells_by_ct=(json.dumps(cell_counts, sort_keys=True)
                                         if unit == 'cell' else None),
               n_phasable_reads_by_ct=json.dumps(depths, sort_keys=True),
               delta_by_ct=None)
    if len(uniq) < 2:
        return out
    out['eligible_cells'] = True
    keep = np.isin(labels_all, uniq)
    labels_arr = labels_all[keep]
    dom_a = dom_b = dom_a_all = dom_b_all = None
    if axis == 'astu':
        if not len(cols):
            out['test_status'] = 'undefined_observed_statistic'
            return out
        pool_pos = np.flatnonzero(use)[keep]
        dom = (A.iloc[pool_pos].sum(axis=0) + B.iloc[pool_pos].sum(axis=0)).idxmax()
        dom_a_all, dom_b_all = A[dom].to_numpy(float)[use], B[dom].to_numpy(float)[use]
        dom_a, dom_b = dom_a_all[keep], dom_b_all[keep]
    deltas_all = _actv_deltas_for_labels(labels_all, uniq_all, axis,
                                         a, b, dom_a_all, dom_b_all)
    out['delta_by_ct'] = json.dumps({L: (round(d, 6) if np.isfinite(d) else None)
                                     for L, d in zip(uniq_all, deltas_all)}, sort_keys=True)
    a, b = a[keep], b[keep]
    deltas = _actv_deltas_for_labels(labels_arr, uniq, axis, a, b, dom_a, dom_b)
    if not np.isfinite(deltas).all():
        out['test_status'] = 'undefined_observed_statistic'
        return out
    real = _actv_range(deltas)
    out['actv'] = real
    rng = np.random.default_rng([int(seed), zlib.crc32(str(gene_id).encode()),
                                 zlib.crc32(axis.encode())])
    null = []
    attempts = 0
    perm = labels_arr.copy()
    while len(null) < n_perm and attempts < max_attempts:
        rng.shuffle(perm)
        attempts += 1
        ds = _actv_deltas_for_labels(perm, uniq, axis, a, b, dom_a, dom_b)

        if np.isfinite(ds).all():
            null.append(_actv_range(ds))
    out.update(n_valid_permutations=len(null), n_permutation_attempts=attempts,
               n_invalid_permutations=attempts-len(null))
    if len(null) != n_perm:
        out['test_status'] = 'permutation_limit_reached'
        return out
    null = np.asarray(null)
    pval = float((1 + np.count_nonzero(null >= real)) / (n_perm + 1))
    out.update(thr95=float(np.quantile(null, .95)), pval=pval,
               call=bool(pval <= .05), test_status='ok')
    return out


_ACTV_LABEL_CACHE = {}


def _actv_label_of(path):
    s = _ACTV_LABEL_CACHE.get(path)
    if s is None:
        with open(path, 'rb') as fh:
            s = pickle.load(fh)
        _ACTV_LABEL_CACHE.clear()
        _ACTV_LABEL_CACHE[path] = s
    return s


def _actv_gene_task(gene_id, gene_name, sample, unit, inputs, gene_rows,
                    params, label_path):
    (n_perm, min_cells, seed, min_phasable_reads, max_attempts,
     min_phasable_frac, min_actv) = params
    mats = expressed_counts = unit_label_of = None


    ct_reads = {}
    if len(gene_rows) and 'n_reads' in gene_rows.columns:
        for r in gene_rows.drop_duplicates('CellType').itertuples(index=False):
            if getattr(r, 'CellType', None) == 'Bulk':
                continue
            nr = pd.to_numeric(getattr(r, 'n_reads', np.nan), errors='coerce')
            npz = pd.to_numeric(getattr(r, 'n_reads_phasable', np.nan), errors='coerce')

            ct_reads[str(r.CellType)] = (float(nr) if np.isfinite(nr) and nr > 0 else None,
                                         float(npz) if np.isfinite(npz) and npz >= 0 else None)
    if inputs is not None:
        if unit == 'read':
            mats, unit_label_of = (inputs['mA'], inputs['mB']), inputs['labels']
        else:
            label_of = _actv_label_of(label_path)
            full_a = pd.read_csv(inputs['fa'], index_col=0)
            full_b = pd.read_csv(inputs['fb'], index_col=0)
            totals = full_a.sum(axis=1).add(full_b.sum(axis=1), fill_value=0)
            expressed_counts = (pd.Series(totals.index[totals > 0].map(label_of))
                                .dropna().value_counts().astype(int).to_dict())
            mats = (pd.read_csv(inputs['pa'], index_col=0),
                    pd.read_csv(inputs['pb'], index_col=0))
            unit_label_of = label_of
    rows = []
    for axis in ('ase', 'astu'):
        res = {'geneID': gene_id, 'axis': axis,
               'actv': np.nan, 'thr95': np.nan, 'pval': np.nan,
               'call': False, 'eligible_cells': False,
               'delta_by_ct': None, 'n_cells_by_ct': None,

               'ase_sig_ct': False, 'astu_sig_ct': False,
               'test_status': 'missing_inputs',
               'n_valid_permutations': 0, 'n_permutation_attempts': 0,
               'n_invalid_permutations': 0, 'qualifying_contexts': '[]',
               'n_permutations': n_perm,
               'n_reads_by_ct': json.dumps({c: v[0] for c, v in ct_reads.items()}, sort_keys=True),
               'min_phasable_frac': np.nan,
               'pass_phasable_frac': False, 'pass_min_actv': False}
        if sample is not None:
            res['Sample'] = sample
        if mats is not None:
            stat = actv_gene_axis(mats[0], mats[1], unit_label_of,
                                  axis, n_perm=n_perm,
                                  min_cells=min_cells,
                                  seed=seed, gene_id=gene_id, unit=unit,
                                  min_phasable_reads=min_phasable_reads,
                                  max_attempts=max_attempts,
                                  expressed_counts=expressed_counts)
            if stat is not None:
                res.update(stat)
                qualifying = json.loads(stat['qualifying_contexts'])
                if stat['eligible_cells']:
                    selected = gene_rows[gene_rows.CellType.isin(qualifying)]


                    res['ase_sig_ct'] = bool(actv_gate_flags(selected, 'ase').get(gene_id, False))
                    res['astu_sig_ct'] = bool(actv_gate_flags(selected, 'astu').get(gene_id, False))


                    fracs = []
                    for c in qualifying:
                        nr, npz = ct_reads.get(str(c), (None, None))
                        fracs.append(npz / nr if (nr is not None and npz is not None and nr > 0)
                                     else np.nan)
                    mf = float(min(fracs)) if fracs and all(f == f for f in fracs) else np.nan
                    res['min_phasable_frac'] = mf
                    res['pass_phasable_frac'] = bool(min_phasable_frac <= 0 or (mf == mf and mf >= min_phasable_frac))
                    av = res.get('actv', np.nan)
                    res['pass_min_actv'] = bool(min_actv <= 0 or (av == av and av >= min_actv))
        rows.append(res)
    return rows


def _run_parallel(n_workers, tasks, fn):
    if n_workers == 0:
        raise ValueError('n_workers=0 is not a worker count (joblib refuses it '
                         'too); use 1 for in-process or -1 for all CPUs')
    if n_workers == 1:
        return [fn(*t) for t in tasks]
    blas_vars = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    saved = {}
    try:
        for v in blas_vars:
            saved[v] = os.environ.get(v)
            os.environ[v] = '1'
        par = Parallel(n_jobs=n_workers, backend='loky', batch_size='auto')
        return par(delayed(fn)(*t) for t in tasks)
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def load_pickle(file):
    if os.path.exists(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
    return None


_OBS_EXON_MIN_OVERLAP_BP = 20
_OBS_EXON_MIN_OVERLAP_FRAC = 0.30
_OBS_JUNCTION_TOLERANCE_BP = 20
_OBS_JUNCTION_FLANK_MIN_BP = 20


def _judge_event_obs(event_type, event, blocks, intron_spans):
    if not blocks:
        return 'unobserved'
    ev_start, ev_end = int(event[0]), int(event[1])
    align_start = blocks[0][0]
    align_end = blocks[-1][1]

    if event_type == 'exon':
        ev_len = max(1, ev_end - ev_start)
        min_overlap = min(_OBS_EXON_MIN_OVERLAP_BP, int(_OBS_EXON_MIN_OVERLAP_FRAC * ev_len))
        min_overlap = max(1, min_overlap)
        total_overlap = 0
        for bs, be in blocks:
            total_overlap += max(0, min(be, ev_end) - max(bs, ev_start))
        if total_overlap >= min_overlap:
            return 'include'


        if align_start >= ev_start or align_end <= ev_end:
            return 'unobserved'
        for ins, ine in intron_spans:
            if ins <= ev_start and ine >= ev_end:
                return 'skip'
        return 'unobserved'

    if event_type == 'junction':
        upstream_cov = 0
        downstream_cov = 0
        for bs, be in blocks:
            upstream_cov += max(0,
                                min(be, ev_start) - max(bs, ev_start - _OBS_JUNCTION_FLANK_MIN_BP))
            downstream_cov += max(0,
                                  min(be, ev_end + _OBS_JUNCTION_FLANK_MIN_BP) - max(bs, ev_end))
        if upstream_cov < _OBS_JUNCTION_FLANK_MIN_BP or downstream_cov < _OBS_JUNCTION_FLANK_MIN_BP:
            return 'unobserved'
        for ins, ine in intron_spans:
            if (abs(ins - ev_start) <= _OBS_JUNCTION_TOLERANCE_BP
                    and abs(ine - ev_end) <= _OBS_JUNCTION_TOLERANCE_BP):
                return 'include'
        return 'skip'

    return 'unobserved'


def _process_gene_events(gene_id, g, gsi, min_reads, gene_event_cache=None,
                         variant_dir=None):
    if gene_event_cache is None:
        if gene_id not in gsi:
            return []
        geneInfo, exon_positions, exon_isoform_dict = gsi[gene_id]
        iso_exon_map, iso_junction_map = Downstream._build_isoform_event_maps(
            exon_positions, exon_isoform_dict)
        all_exons = sorted({e for s in iso_exon_map.values() for e in s
                            if e[1] - e[0] >= 5})
        all_junctions = sorted({j for s in iso_junction_map.values() for j in s
                                if j[1] != j[0]})
        iso_exon_event_indices = _build_event_indices(all_exons, iso_exon_map)
        iso_junction_event_indices = _build_event_indices(all_junctions, iso_junction_map)
    else:
        geneInfo = gene_event_cache['geneInfo']
        iso_exon_map = gene_event_cache['iso_exon_map']
        iso_junction_map = gene_event_cache['iso_junction_map']
        all_exons = gene_event_cache.get('all_exons')
        all_junctions = gene_event_cache.get('all_junctions')
        iso_exon_event_indices = gene_event_cache.get('iso_exon_event_indices')
        iso_junction_event_indices = gene_event_cache.get('iso_junction_event_indices')

        if all_exons is None:
            all_exons = sorted({e for s in iso_exon_map.values() for e in s
                                if e[1] - e[0] >= 5})
        if all_junctions is None:
            all_junctions = sorted({j for s in iso_junction_map.values() for j in s
                                    if j[1] != j[0]})
        if iso_exon_event_indices is None:
            iso_exon_event_indices = _build_event_indices(all_exons, iso_exon_map)
        if iso_junction_event_indices is None:
            iso_junction_event_indices = _build_event_indices(all_junctions, iso_junction_map)

    if not iso_exon_map or g.empty:
        return []


    iso_weights = g.groupby('Isoform', sort=False)[['hat_I', 'hat_I_B']].sum()
    if iso_weights.empty:
        return []

    iso_names = iso_weights.index.to_numpy()
    hat_I_by_iso = iso_weights['hat_I'].to_numpy(dtype=float)
    hat_I_B_by_iso = iso_weights['hat_I_B'].to_numpy(dtype=float)
    total_A = float(hat_I_by_iso.sum())
    total_B = float(hat_I_B_by_iso.sum())


    read_obs_data = _gather_read_obs_data(g, variant_dir=variant_dir, gene_id=gene_id)

    rows = []
    for event_type, events, iso_event_indices in [
        ('exon', all_exons, iso_exon_event_indices),
        ('junction', all_junctions, iso_junction_event_indices),
    ]:
        if not events:
            continue

        n_ev = len(events)
        hapA_pres = np.zeros(n_ev, dtype=float)
        hapB_pres = np.zeros(n_ev, dtype=float)

        for iso, a_weight, b_weight in zip(iso_names, hat_I_by_iso, hat_I_B_by_iso):
            idx = iso_event_indices.get(iso)
            if idx is None or len(idx) == 0:
                continue
            hapA_pres[idx] += a_weight
            hapB_pres[idx] += b_weight

        hapA_abs = total_A - hapA_pres
        hapB_abs = total_B - hapB_pres

        obs_hapA_inc = obs_hapA_skip = obs_hapA_unobs = None
        obs_hapB_inc = obs_hapB_skip = obs_hapB_unobs = None
        if read_obs_data is not None:
            obs_hapA_inc, obs_hapA_skip, obs_hapA_unobs, \
                obs_hapB_inc, obs_hapB_skip, obs_hapB_unobs = _aggregate_obs_per_event(
                    read_obs_data, event_type, events)

        for e_idx, event in enumerate(events):
            table = np.array([
                [hapA_pres[e_idx], hapA_abs[e_idx]],
                [hapB_pres[e_idx], hapB_abs[e_idx]]
            ])
            if table.sum() < min_reads or table.min() < 1:
                continue
            try:
                chi2_stat, p_val, _, _ = chi2_contingency(table, correction=False)
            except Exception:
                continue

            obs_extra = _obs_columns_for_event(
                e_idx, obs_hapA_inc, obs_hapA_skip, obs_hapA_unobs,
                obs_hapB_inc, obs_hapB_skip, obs_hapB_unobs,
                min_reads=min_reads, read_obs_available=(read_obs_data is not None),
            )

            row = {
                'geneID': gene_id,
                'geneName': geneInfo['geneName'],
                'geneChr': geneInfo['geneChr'],
                'event_type': event_type,
                'event_start': int(event[0]),
                'event_end': int(event[1]),
                'hapA_present': round(hapA_pres[e_idx], 2),
                'hapA_absent': round(hapA_abs[e_idx], 2),
                'hapB_present': round(hapB_pres[e_idx], 2),
                'hapB_absent': round(hapB_abs[e_idx], 2),
                'chi2': round(float(chi2_stat), 4),
                'p_value': float(p_val),
            }
            row.update(obs_extra)
            rows.append(row)
    return rows


def _build_event_indices(events, iso_event_map):
    event_to_idx = {event: idx for idx, event in enumerate(events)}
    result = {}
    for iso, iso_events in iso_event_map.items():
        if not iso_events:
            continue
        idx = [event_to_idx[e] for e in sorted(iso_events) if e in event_to_idx]
        if idx:
            result[iso] = np.array(idx, dtype=np.int32)
    return result


def _process_gene_events_joined(gene_id, read_hap_gene, scotch_gene_isoform,
                                gsi, min_reads, gene_event_cache=None,
                                variant_dir=None):
    merged = read_hap_gene.join(scotch_gene_isoform, on='Read', how='inner')
    if merged.empty:
        return []
    return _process_gene_events(
        gene_id, merged, gsi, min_reads, gene_event_cache=gene_event_cache,
        variant_dir=variant_dir,
    )


def _gather_read_obs_data(g, variant_dir, gene_id):
    if not variant_dir:
        return None
    pkl_path = os.path.join(variant_dir, f'{gene_id}_read_blocks.pkl')
    if not os.path.isfile(pkl_path):
        return None
    read_weights = g.groupby('Read', sort=False)[['hat_I', 'hat_I_B']].first()
    if read_weights.empty:
        return None
    try:
        read_blocks = load_pickle(pkl_path)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    if not isinstance(read_blocks, dict):
        return None
    out = {}
    for rn, row in read_weights.iterrows():
        hat_I = float(row.hat_I)
        hat_I_B = float(row.hat_I_B)
        entry = read_blocks.get(rn)
        if entry is None:
            out[rn] = (hat_I, hat_I_B, None, None)
            continue
        try:
            blocks, intron_spans = entry
        except (TypeError, ValueError):


            out[rn] = (hat_I, hat_I_B, None, None)
            continue
        out[rn] = (hat_I, hat_I_B, blocks, intron_spans)
    return out


def _aggregate_obs_per_event(read_obs_data, event_type, events):
    n_ev = len(events)
    hapA_inc = np.zeros(n_ev, dtype=float)
    hapA_skip = np.zeros(n_ev, dtype=float)
    hapA_unobs = np.zeros(n_ev, dtype=float)
    hapB_inc = np.zeros(n_ev, dtype=float)
    hapB_skip = np.zeros(n_ev, dtype=float)
    hapB_unobs = np.zeros(n_ev, dtype=float)
    for hat_I, hat_I_B, blocks, intron_spans in read_obs_data.values():
        if blocks is None:
            hapA_unobs += hat_I
            hapB_unobs += hat_I_B
            continue
        for e_idx, event in enumerate(events):
            status = _judge_event_obs(event_type, event, blocks, intron_spans)
            if status == 'include':
                hapA_inc[e_idx] += hat_I
                hapB_inc[e_idx] += hat_I_B
            elif status == 'skip':
                hapA_skip[e_idx] += hat_I
                hapB_skip[e_idx] += hat_I_B
            else:
                hapA_unobs[e_idx] += hat_I
                hapB_unobs[e_idx] += hat_I_B
    return hapA_inc, hapA_skip, hapA_unobs, hapB_inc, hapB_skip, hapB_unobs


def _obs_columns_for_event(e_idx, hapA_inc, hapA_skip, hapA_unobs,
                           hapB_inc, hapB_skip, hapB_unobs,
                           min_reads, read_obs_available):
    if not read_obs_available:
        return {
            'obs_hapA_include': None, 'obs_hapA_skip': None, 'obs_hapA_unobserved': None,
            'obs_hapB_include': None, 'obs_hapB_skip': None, 'obs_hapB_unobserved': None,
            'obs_chi2': None, 'obs_p_value': None, 'obs_test_type': 'no_bam',
        }
    a_inc = float(hapA_inc[e_idx]); a_skip = float(hapA_skip[e_idx]); a_unobs = float(hapA_unobs[e_idx])
    b_inc = float(hapB_inc[e_idx]); b_skip = float(hapB_skip[e_idx]); b_unobs = float(hapB_unobs[e_idx])
    table = np.array([[a_inc, a_skip], [b_inc, b_skip]])
    chi2_val = p_val = None
    test_type = 'insufficient_data'


    margins_ok = not ((table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any())
    if table.sum() >= min_reads and margins_ok:
        try:
            chi2_stat, p_v, _, _ = chi2_contingency(table, correction=False)
            chi2_val = round(float(chi2_stat), 4)
            p_val = float(p_v)
            test_type = 'chi2_hap_event'
        except Exception:
            pass
    return {
        'obs_hapA_include': round(a_inc, 2),
        'obs_hapA_skip': round(a_skip, 2),
        'obs_hapA_unobserved': round(a_unobs, 2),
        'obs_hapB_include': round(b_inc, 2),
        'obs_hapB_skip': round(b_skip, 2),
        'obs_hapB_unobserved': round(b_unobs, 2),
        'obs_chi2': chi2_val,
        'obs_p_value': p_val,
        'obs_test_type': test_type,
    }


class Downstream:

    def __init__(self, output_folder, scotch_target, bam_path=None,
                 ref_pickle_path=None, sample_name_parse=None,
                 prefix='LongAllele', sample_names=None,
                 cell_type_df_path=None, n_workers=1, logger=None,
                 astu_sig_only=False, astu_sig_from_bulk=False,
                 astu_sig_threshold=0.05, n_jobs=1, job_index=0,
                 job_array_by_sample=False, gene_subset=None,
                 conf_nonphasable_astu=1.0, conf_nonphasable=None,
                 ase_call_margin=0.095):
        self.output_folder = output_folder
        self.scotch_target = ([scotch_target] if isinstance(scotch_target, str)
                              else list(scotch_target))
        self.bam_paths = self._normalize_optional_list(bam_path, len(self.scotch_target))
        self.bam_path = self.bam_paths[0] if self.bam_paths else None
        self.sample_name_parse = sample_name_parse
        self.prefix = prefix
        self.n_workers = n_workers
        self.logger = logger
        self.astu_sig_only = astu_sig_only
        self.astu_sig_from_bulk = astu_sig_from_bulk
        self.astu_sig_threshold = astu_sig_threshold


        if conf_nonphasable is not None:
            conf_nonphasable_astu = conf_nonphasable
        self.conf_nonphasable_astu = conf_nonphasable_astu


        self.ase_call_margin = float(ase_call_margin)
        self.n_jobs = n_jobs
        self.job_index = job_index
        self.job_array_by_sample = job_array_by_sample

        if self.astu_sig_only and self.astu_sig_from_bulk:
            msg = 'Both astu_sig_only and astu_sig_from_bulk set; using bulk-level filtering.'
            if self.logger:
                self.logger.warning(msg)
            else:
                print(f'WARNING: {msg}')

        n_samples = len(self.scotch_target)
        self.sample_names = self._normalize_sample_names(sample_names, self.scotch_target)


        if cell_type_df_path is not None:
            paths = ([cell_type_df_path] if isinstance(cell_type_df_path, str)
                     else list(cell_type_df_path))
            if len(paths) == 1:
                paths = paths * n_samples
            elif len(paths) != n_samples:
                raise ValueError('cell_type_df_path must contain one entry per sample.')
            self.cell_type_df_list = [pd.read_csv(p) for p in paths]
        else:
            self.cell_type_df_list = None
        self.cell_type_df = None
        self.ref_pickle_path = ref_pickle_path
        self.gene_subset = set(gene_subset) if gene_subset is not None else None
        self.gsi = None
        self.meta = None
        self.gsi_path = None
        self.scotch_gtf_path = None
        self._sample_gtf_junction_index = None
        self._sample_gtf_junction_index_path = None

        self.sample_configs = self._build_sample_configs()
        self._set_sample_context(0)
        self._bulk_shrinkage_k = None


    def run_all(self, event_min_reads=10,
                snv_event_distance=50,
                event_mode='all_events',
                fdr_events_value=0.05,
                actv=False, actv_permutations=300, actv_min_cells=10,
                actv_seed=42, actv_unit='cell', actv_min_phasable_reads=20,
                actv_max_attempts=None, actv_min_phasable_frac=0.6,
                actv_min_actv=0.3):
        if self.job_array_by_sample:
            if self.job_index < 0 or self.job_index >= len(self.sample_configs):
                raise ValueError(
                    f'job_index {self.job_index} out of range for '
                    f'{len(self.sample_configs)} samples.')
            sample_iter = [(self.job_index, self.sample_configs[self.job_index])]
        else:
            sample_iter = list(enumerate(self.sample_configs))

        for sample_idx, sample_cfg in sample_iter:
            self._set_sample_context(sample_idx)
            os.makedirs(self.downstream_output, exist_ok=True)
            self._log(f'Processing sample: {sample_cfg["sample_name"]}')
            has_raw_alleles = bool(self.bam_path) or self._has_variant_site_read_pkls()
            self._sample_site_reads_cache = {}
            self._gene_event_map_cache = {}
            self._resolved_bam_path_cache = {}
            self._isoform_agg_cache = {}
            self._scotch_by_gene_cache = {}

            if self.gene_subset is not None:
                self._log(
                    f'Applying gene subset filter to step 5: '
                    f'{len(self.gene_subset)} genes requested.'
                )


            summary_df = None
            summary_by_cell_type = None
            if os.path.exists(self.summary_statistics_path):
                summary_df = pd.read_csv(self.summary_statistics_path)
                summary_by_cell_type = {
                    cell_type: grp.copy()
                    for cell_type, grp in summary_df.groupby('CellType')
                }
            self._bulk_shrinkage_k = self._compute_bulk_shrinkage_k(summary_by_cell_type)
            if self.gene_subset is not None and summary_df is not None and 'geneID' in summary_df.columns:
                summary_df = summary_df[
                    summary_df['geneID'].isin(self.gene_subset)
                ].copy()
                summary_by_cell_type = {
                    cell_type: grp.copy()
                    for cell_type, grp in summary_df.groupby('CellType')
                }

            bulk_sig_gene_ids = None
            if self.astu_sig_from_bulk:
                bulk_sig_gene_ids = self._get_significant_astu_gene_ids(
                    cell_type='Bulk', summary_df=summary_df)

            read_hap_df = None
            if os.path.exists(self.read_hap_map_path):
                read_hap_df = pd.read_csv(self.read_hap_map_path)
                if self.gene_subset is not None and 'geneID' in read_hap_df.columns:
                    read_hap_df = read_hap_df[
                        read_hap_df['geneID'].isin(self.gene_subset)
                    ].copy()


            self._log('Task 1: Refining SNV calls...')
            snv_df = self._refine_snv_calls()
            if self.gene_subset is not None and 'geneID' in snv_df.columns:
                snv_df = snv_df[snv_df['geneID'].isin(self.gene_subset)].copy()


            cell_types = self._discover_cell_types(summary_df=summary_df)


            scotch_read_cell = None
            scotch_isoform_df = None
            if os.path.exists(self.scotch_tsv_path):
                scotch_read_cell, scotch_isoform_df = self._load_scotch_tables()
                if self.gene_subset is not None:
                    if scotch_read_cell is not None and not scotch_read_cell.empty:
                        scotch_read_cell = scotch_read_cell[
                            scotch_read_cell['geneID'].isin(self.gene_subset)
                        ].copy()
                    if scotch_isoform_df is not None and not scotch_isoform_df.empty:
                        scotch_isoform_df = scotch_isoform_df[
                            scotch_isoform_df['geneID'].isin(self.gene_subset)
                        ].copy()
                if scotch_isoform_df is not None and not scotch_isoform_df.empty:
                    self._scotch_by_gene_cache = {
                        gene_id: sub[['Read', 'Isoform']].set_index('Read')
                        for gene_id, sub in scotch_isoform_df.groupby('geneID', sort=False)
                    }

            gene_snv_frames = []
            event_snv_frames = []

            for ct in cell_types:
                self._log(f'--- Cell type: {ct} ---')
                ct_summary = None
                if summary_by_cell_type is not None:
                    ct_summary = summary_by_cell_type.get(ct, summary_df.iloc[0:0].copy())


                self._log('  Tasks 2/3: Computing ASE / aStu effect sizes...')
                _t0 = time.perf_counter()
                effect_df = self._compute_effect_sizes(
                    snv_df, cell_type=ct,
                    summary_df=summary_df, ct_summary=ct_summary)
                self._log(f'  Tasks 2/3 done in {time.perf_counter() - _t0:.1f}s')
                gene_snv_df = self._assemble_gene_snv_output(
                    effect_df,
                    sample_name=sample_cfg['sample_name'],
                    cell_type=ct
                )
                gene_snv_frames.append(gene_snv_df)


                self._log('  Task 4a: Haplotype–event associations...')
                _t0 = time.perf_counter()
                hap_event_df = self._haplotype_event_associations(
                    min_reads=event_min_reads, cell_type=ct,
                    scotch_read_cell=scotch_read_cell,
                    read_hap_df=read_hap_df,
                    scotch_isoform_df=scotch_isoform_df,
                    summary_df=summary_df, ct_summary=ct_summary,
                    bulk_sig_gene_ids=bulk_sig_gene_ids,
                    event_mode=event_mode,
                    fdr_events_value=fdr_events_value)
                self._log(f'  Task 4a done in {time.perf_counter() - _t0:.1f}s')
                if hap_event_df is None or hap_event_df.empty:
                    self._log('  No haplotype–event associations found; skipping Tasks 4b/4c.')
                    continue


                self._log('  Task 4b: Linking SNVs to nearby events...')
                _t0 = time.perf_counter()
                snv_event_df = self._link_snv_to_events(
                    snv_df, hap_event_df,
                    max_exonic_dist=snv_event_distance)
                self._log(f'  Task 4b done in {time.perf_counter() - _t0:.1f}s')
                if snv_event_df.empty:
                    self._log('  No SNV–event pairs within distance threshold.')


                chi_df = None
                if has_raw_alleles:
                    self._log('  Task 4c: Raw-read chi-squared test for SNV–event pairs...')
                    _t0 = time.perf_counter()
                    chi_df = self._chi_sq_snv_event_raw(
                        snv_event_df,
                        cell_type=ct,
                        scotch_read_cell=scotch_read_cell,
                        cell_type_df=self.cell_type_df,
                        scotch_isoform_df=scotch_isoform_df
                    )
                    self._log(f'  Task 4c done in {time.perf_counter() - _t0:.1f}s')

                event_snv_df = self._assemble_event_snv_output(
                    hap_event_df=hap_event_df,
                    snv_event_df=snv_event_df,
                    chi_df=chi_df,
                    gene_snv_df=gene_snv_df,
                    sample_name=sample_cfg['sample_name'],
                    cell_type=ct
                )
                event_snv_frames.append(event_snv_df)

            gene_snv_output_df = pd.concat(gene_snv_frames, ignore_index=True) if gene_snv_frames else None
            event_snv_output_df = pd.concat(event_snv_frames, ignore_index=True) if event_snv_frames else None

            gene_snv_output_path = os.path.join(self.downstream_output, 'gene_snv.csv')
            if gene_snv_output_df is not None:
                if self.gene_subset is not None:
                    gene_snv_output_df = self._merge_subset_output(
                        gene_snv_output_path, gene_snv_output_df
                    )
                gene_snv_output_df.to_csv(gene_snv_output_path, index=False)

            if actv and gene_snv_output_df is not None:
                _t0 = time.perf_counter()
                self._run_actv(gene_snv_output_df,
                               n_perm=actv_permutations,
                               min_cells=actv_min_cells, seed=actv_seed,
                               min_phasable_reads=actv_min_phasable_reads,
                               max_attempts=actv_max_attempts,
                               min_phasable_frac=actv_min_phasable_frac,
                               min_actv=actv_min_actv,
                               unit=actv_unit, read_hap_df=read_hap_df,
                               scotch_read_cell=scotch_read_cell,
                               scotch_isoform_df=scotch_isoform_df)
                self._log(f'[actv] done in {time.perf_counter() - _t0:.1f}s')

            event_snv_output_path = os.path.join(
                self.downstream_output,
                self._event_mode_filename('event_snv.csv', event_mode, fdr_events_value)
            )
            if self.gene_subset is not None:
                event_snv_output_df = self._merge_subset_output(
                    event_snv_output_path, event_snv_output_df
                )
                if event_snv_output_df is not None:
                    event_snv_output_df.to_csv(event_snv_output_path, index=False)
            elif event_snv_output_df is not None:
                event_snv_output_df.to_csv(event_snv_output_path, index=False)

            self._log(
                'Done. Results written to '
                f'{self.downstream_output} '
                f'(event_snv={os.path.basename(event_snv_output_path)}).'
            )


    def _run_actv(self, gene_snv_df, n_perm=300, min_cells=10, seed=42,
                  unit='cell', read_hap_df=None, scotch_read_cell=None,
                  scotch_isoform_df=None, min_phasable_reads=20,
                  max_attempts=None, min_phasable_frac=0.6, min_actv=0.3):
        if n_perm < 1:
            raise ValueError(f'--actv_permutations must be >= 1, got {n_perm}'
                             f' (0 would print pval=1 for everything and read '
                             f'like a finished calibration)')
        if min_cells < 1:
            raise ValueError(f'--actv_min_cells must be >= 1, got {min_cells}'
                             f' — 0 disables the eligibility floor the option '
                             f'exists to enforce')
        if min_phasable_reads < 0:
            raise ValueError('ACTV min_phasable_reads must be >= 0')
        if max_attempts is not None and max_attempts < n_perm:
            raise ValueError('ACTV max_attempts must be >= n_perm')
        if not (0.0 <= float(min_phasable_frac) <= 1.0):
            raise ValueError(f'--actv_min_phasable_frac must be in [0, 1], got {min_phasable_frac}')
        if not (np.isfinite(float(min_actv)) and float(min_actv) >= 0):
            raise ValueError(f'--actv_min_actv must be a finite number >= 0, got {min_actv}')
        if unit not in ('cell', 'read'):
            raise ValueError(f"--actv_unit must be 'cell' or 'read', got "
                             f"{unit!r}")
        if self.cell_type_df is None:
            self._log('[actv] SKIPPED: --actv needs a cell type table '
                      '(cell_type_df_path) — nothing to permute without '
                      'labels.')
            return None
        if unit == 'read' and (read_hap_df is None or scotch_read_cell is None
                               or scotch_isoform_df is None):
            raise ValueError('--actv_unit read needs read_hap_map.csv and '
                             'the SCOTCH mapping TSV — refuse loudly rather '
                             'than fall back to cell mode on a bulk tree')
        label_of = self.cell_type_df.set_index('Cell')['CellType']
        genes = (gene_snv_df.drop_duplicates('geneID')
                 [['geneID', 'geneName', 'Sample']]
                 if 'Sample' in gene_snv_df.columns else
                 gene_snv_df.drop_duplicates('geneID')[['geneID', 'geneName']])
        sep_dir = os.path.join(self.count_dir, 'all_genes_separate')
        if unit == 'read':


            rh_by_gene = read_hap_df.groupby('geneID')
            iso_by_gene = scotch_isoform_df.groupby('geneID')
            cell_by_gene = scotch_read_cell.groupby('geneID')
            unmapped = (set(scotch_read_cell['Cell'].dropna().unique())
                        - set(self.cell_type_df['Cell']))
            if unmapped:


                self._log(f'[actv] ⚠️ {len(unmapped)} Cell tag(s) in the '
                          f'SCOTCH map have NO row in the cell type table '
                          f'and their reads are dropped: '
                          f'{sorted(unmapped)[:5]}')


        n_workers = int(getattr(self, 'n_workers', 1))
        params = (n_perm, min_cells, seed, min_phasable_reads, max_attempts,
                  float(min_phasable_frac), float(min_actv))
        rows_by_gene = {g: sub for g, sub in gene_snv_df.groupby('geneID', sort=False)}
        label_path = None
        n_missing = [0]

        def _tasks():
            for _, gr in genes.iterrows():
                gene_id, gene_name = gr['geneID'], gr.get('geneName')
                inputs = None
                if unit == 'read':
                    if (gene_id in rh_by_gene.groups
                            and gene_id in iso_by_gene.groups
                            and gene_id in cell_by_gene.groups):
                        rh = rh_by_gene.get_group(gene_id)
                        im = iso_by_gene.get_group(gene_id)
                        cm = cell_by_gene.get_group(gene_id)
                        mA, mB = actv_read_unit_matrices(rh, im)
                        if mA is not None:
                            inputs = {'mA': mA, 'mB': mB,
                                      'labels': cm.set_index('Read')['Cell'].map(label_of)}
                    else:
                        n_missing[0] += 1
                else:
                    base = f"{_path_safe_gene_name(str(gene_name))}_{gene_id}"
                    pa = os.path.join(sep_dir, f'{base}_hapA_phasable.csv')
                    pb = os.path.join(sep_dir, f'{base}_hapB_phasable.csv')
                    if os.path.exists(pa) and os.path.exists(pb):
                        fa = os.path.join(sep_dir, f'{base}_hapA.csv')
                        fb = os.path.join(sep_dir, f'{base}_hapB.csv')
                        if not (os.path.exists(fa) and os.path.exists(fb)):
                            raise FileNotFoundError(
                                f'ACTV needs full cell counts to count expressing cells: {base}')
                        inputs = {'pa': pa, 'pb': pb, 'fa': fa, 'fb': fb}
                gene_rows = rows_by_gene.get(gene_id)
                if gene_rows is None:
                    gene_rows = gene_snv_df.iloc[0:0]
                yield (gene_id, gene_name, gr['Sample'] if 'Sample' in gr else None,
                       unit, inputs, gene_rows, params, label_path)

        self._log(f'[actv] {len(genes)} genes over {n_workers} worker(s)')
        try:
            if unit == 'cell':


                import tempfile
                fd, label_path = tempfile.mkstemp(prefix='.actv_labels_', suffix='.pkl',
                                                  dir=self.downstream_output)
                with os.fdopen(fd, 'wb') as fh:
                    pickle.dump(label_of, fh, protocol=pickle.HIGHEST_PROTOCOL)
            per_gene = _run_parallel(n_workers, _tasks(), _actv_gene_task)
        finally:
            if label_path and os.path.exists(label_path):
                os.remove(label_path)
        n_missing_inputs = n_missing[0]
        rows = [r for gene_rows in per_gene for r in gene_rows]
        if unit == 'read' and n_missing_inputs:

            self._log(f'[actv] ⚠️ {n_missing_inputs} gene(s) in gene_snv have '
                      f'no read-mode inputs (read_hap / SCOTCH map rows) — '
                      f'their ACTV rows are NaN, not zero-signal')
        cols = ['geneID', 'axis', 'actv', 'thr95', 'pval', 'call',
                'eligible_cells', 'delta_by_ct', 'n_cells_by_ct', 'ase_sig_ct', 'astu_sig_ct',
                'n_permutations', 'qualifying_contexts', 'test_status',
                'n_valid_permutations', 'n_permutation_attempts',
                'n_invalid_permutations', 'n_phasable_reads_by_ct',
                'n_expressing_cells_by_ct', 'n_reads_by_ct',
                'min_phasable_frac', 'pass_phasable_frac', 'pass_min_actv']
        out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


        for c in cols:
            if c not in out.columns:
                out[c] = pd.Series(dtype=object)
        out_path = os.path.join(self.downstream_output, 'actv_results.csv')
        if getattr(self, 'gene_subset', None) is not None \
                and os.path.exists(out_path):


            subset = set(self.gene_subset)
            out = out[out['geneID'].isin(subset)]
            old_rows = pd.read_csv(out_path)
            out = pd.concat([old_rows[~old_rows['geneID'].isin(subset)], out],
                            ignore_index=True)


            for c in ('pass_phasable_frac', 'pass_min_actv'):
                out[c] = out[c].map(lambda v: bool(v) if v == v and v is not None else False)
            if 'min_phasable_frac' in out:
                out['min_phasable_frac'] = pd.to_numeric(out['min_phasable_frac'], errors='coerce')
        out.to_csv(out_path, index=False)
        self._log(f'[actv] {out_path}: {len(out)} rows '
                  f'(unit={unit}, '
                  f'{int(out.eligible_cells.sum())} eligible axis-genes, '
                  f'B={n_perm}, min_cells={min_cells}, min_phasable_reads={min_phasable_reads}, '
                  f'min_phasable_frac={min_phasable_frac}, min_actv={min_actv}; '
                  f'pass_phasable_frac {int(out.pass_phasable_frac.map(lambda v: v is True or v == True).sum())}, '
                  f'pass_min_actv {int(out.pass_min_actv.map(lambda v: v is True or v == True).sum())} of '
                  f'{int(out.eligible_cells.astype(bool).sum())} eligible — flags, not filters)')
        return out

    def _discover_cell_types(self, summary_df=None):
        if summary_df is None:
            if not os.path.exists(self.summary_statistics_path):
                return ['Bulk']
            summary_df = pd.read_csv(self.summary_statistics_path, usecols=['CellType'])
        cts = summary_df['CellType'].dropna().unique().tolist()
        bulk = [c for c in cts if c == 'Bulk']
        others = sorted(c for c in cts if c != 'Bulk')
        return bulk + others

    def _load_scotch_tables(self):
        if not os.path.exists(self.scotch_tsv_path):
            return None, None

        chunks = pd.read_csv(
            self.scotch_tsv_path, sep='\t', chunksize=100_000,
            usecols=lambda c: c in {'Read', 'geneID', 'Cell', 'Isoform', 'Keep'})
        frames = []
        for chunk in chunks:
            sub = chunk.loc[chunk['Keep'] == 1, ['Read', 'geneID', 'Cell', 'Isoform']].copy()
            sub['Read'] = sub['Read'].map(_canonicalize_read_name)
            frames.append(sub)
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            empty_read_cell = pd.DataFrame(columns=['Read', 'geneID', 'Cell'])
            empty_isoform = pd.DataFrame(columns=['Read', 'geneID', 'Isoform'])
            return empty_read_cell, empty_isoform

        scotch_df = pd.concat(frames, ignore_index=True)
        scotch_read_cell = scotch_df[['Read', 'geneID', 'Cell']].drop_duplicates(
            subset=['Read', 'geneID']
        ).reset_index(drop=True)
        scotch_isoform_df = scotch_df[['Read', 'geneID', 'Isoform']].drop_duplicates().reset_index(drop=True)
        return scotch_read_cell, scotch_isoform_df

    def _load_scotch_read_cell(self):
        scotch_read_cell, _ = self._load_scotch_tables()
        return scotch_read_cell

    @staticmethod
    def _event_mode_suffix(event_mode='all_events', fdr_events_value=0.05):
        if event_mode == 'all_events':
            return ''
        if event_mode == 'switching_events':
            return '_switching'
        if event_mode == 'fdr_events':
            try:
                value = float(fdr_events_value)
            except (TypeError, ValueError):
                value = 0.05
            value_str = f'{value:g}'.replace('.', '')
            return f'_fdr{value_str}'
        raise ValueError(f'Unsupported event_mode: {event_mode}')

    @classmethod
    def _event_mode_filename(cls, filename, event_mode='all_events',
                             fdr_events_value=0.05):
        stem, ext = os.path.splitext(filename)
        return f'{stem}{cls._event_mode_suffix(event_mode, fdr_events_value)}{ext}'


    def _refine_snv_calls(self):
        snv_df = pd.read_csv(self.snv_hap_map_path)
        snv_df = _collapse_legacy_merge_suffixes(snv_df, self._log)
        h_A = snv_df['h_A'].values.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = np.where(
                (h_A <= 0) | (h_A >= 1),
                0.0,
                -(h_A * np.log2(np.clip(h_A, 1e-15, 1))
                  + (1 - h_A) * np.log2(np.clip(1 - h_A, 1e-15, 1)))
            )
        snv_df['entropy'] = entropy
        hat_Z = snv_df['h_m'] * (1 - entropy)
        snv_df['hat_Z_prob_revised'] = hat_Z.round(2)
        snv_df['hat_Z_binary_revised'] = (snv_df['hat_Z_prob_revised'] >= 0.5).astype(int)
        self._gene_n_snvs_called_map = (
            snv_df.loc[snv_df['hat_Z_prob_revised'] >= 0.5]
            .groupby('geneID', dropna=False)
            .size()
            .to_dict()
        )

        snv_df = snv_df[snv_df['hat_Z_binary_revised'] == 1].drop(
            columns=['hat_Z_binary_revised']).reset_index(drop=True)
        snv_df = self._ensure_alt_column(snv_df)
        return snv_df

    def _ensure_alt_column(self, snv_df):
        if 'alt' in snv_df.columns:
            snv_df['alt'] = snv_df['alt'].replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        elif 'snv_alt' in snv_df.columns:
            snv_df['alt'] = snv_df['snv_alt'].replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        else:
            snv_df['alt'] = np.nan

        missing_alt = snv_df['alt'].isna()
        if not missing_alt.any():
            return snv_df

        alt_cache = {}
        unresolved_keys = []
        missing_sites = snv_df.loc[
            missing_alt, ['geneID', 'chrom', 'pos', 'ref']
        ].drop_duplicates()

        for _, row in missing_sites.iterrows():
            gene_key = None if pd.isna(row['geneID']) else str(row['geneID'])
            chrom = str(row['chrom'])
            pos = int(row['pos'])
            ref_base = str(row['ref']).upper()
            key = (gene_key, chrom, pos, ref_base)

            site_reads_by_gene = (
                None if gene_key is None else self._load_site_reads_for_gene(gene_key)
            )
            alt_cache[key] = self._infer_alt_allele_from_site_reads(
                site_reads_by_gene, chrom=chrom, pos=pos, ref_base=ref_base
            )
            if pd.isna(alt_cache[key]):
                unresolved_keys.append(key)

        unresolved_by_chrom = {}
        for key in unresolved_keys:
            unresolved_by_chrom.setdefault(key[1], []).append(key)

        for chrom, chrom_keys in unresolved_by_chrom.items():
            bam_path = self._resolve_bam_path(chrom)
            if bam_path is None:
                continue
            try:
                bam = pysam.Samfile(bam_path, 'rb')
            except Exception as exc:
                self._log(f'Unable to open BAM for SNV alt recovery ({bam_path}): {exc}')
                continue

            try:
                for key in chrom_keys:
                    alt_cache[key] = self._infer_alt_allele_from_bam(
                        bam, chrom=chrom, pos=key[2], ref_base=key[3]
                    )
            finally:
                bam.close()

        snv_df.loc[missing_alt, 'alt'] = snv_df.loc[missing_alt].apply(
            lambda r: alt_cache.get(
                (
                    None if pd.isna(r['geneID']) else str(r['geneID']),
                    str(r['chrom']),
                    int(r['pos']),
                    str(r['ref']).upper(),
                ),
                np.nan,
            ),
            axis=1
        )

        unresolved = int(snv_df['alt'].isna().sum())
        if unresolved:
            self._log(
                f'Alt allele could not be recovered for {unresolved} SNVs after checking '
                'site_reads.pkl and BAM; raw SNV-event validation will skip those sites.'
            )
        return snv_df

    @staticmethod
    def _infer_alt_allele_from_site_reads(site_reads_by_gene, chrom, pos, ref_base):
        read_alleles = Downstream._read_alleles_from_site_reads(
            site_reads_by_gene, chrom=chrom, pos=pos
        )
        if read_alleles is None:
            return np.nan

        counts = {}
        for allele in read_alleles.values():
            if allele in {'A', 'C', 'G', 'T'} and allele != ref_base:
                counts[allele] = counts.get(allele, 0) + 1

        if not counts:
            return np.nan
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def _resolve_bam_path(self, chrom):
        if not self.bam_path:
            return None

        bam_path = self.bam_path if isinstance(self.bam_path, str) else self.bam_path[0]
        chrom_key = None if chrom is None or pd.isna(chrom) else str(chrom)

        cache = getattr(self, '_resolved_bam_path_cache', None)
        if cache is None:
            self._resolved_bam_path_cache = {}
            cache = self._resolved_bam_path_cache

        cache_key = (bam_path, chrom_key)
        if cache_key in cache:
            return cache[cache_key]

        resolved = None
        if os.path.isfile(bam_path):
            resolved = bam_path
        elif os.path.isdir(bam_path):
            if chrom_key is None:
                self._log(
                    f'Unable to resolve BAM from directory {bam_path}: chromosome was not provided.'
                )
            else:
                bam_names = sorted(
                    f for f in os.listdir(bam_path)
                    if f.endswith('.bam') and f'.{chrom_key}.' in f
                )
                if not bam_names:
                    self._log(
                        f'No chromosome-specific BAM found for {chrom_key} in {bam_path}; '
                        f'expected a file matching *.{chrom_key}.*.bam.'
                    )
                else:
                    if len(bam_names) > 1:
                        self._log(
                            f'Multiple chromosome-specific BAMs found for {chrom_key} in '
                            f'{bam_path}; using {bam_names[0]}.'
                        )
                    resolved = os.path.join(bam_path, bam_names[0])
        else:
            self._log(f'BAM path does not exist or is not accessible: {bam_path}')

        cache[cache_key] = resolved
        return resolved

    @staticmethod
    def _infer_alt_allele_from_bam(bam, chrom, pos, ref_base):
        counts = {}
        try:
            for col in bam.pileup(chrom, int(pos), int(pos) + 1,
                                  stepper='samtools',
                                  min_base_quality=0, min_mapping_quality=0):
                if col.reference_pos != int(pos):
                    continue
                for pr in col.pileups:
                    if pr.is_del or pr.is_refskip:
                        continue
                    base = pr.alignment.query_sequence[pr.query_position].upper()
                    if base in {'A', 'C', 'G', 'T'} and base != ref_base:
                        counts[base] = counts.get(base, 0) + 1
                break
        except Exception:
            return np.nan
        if not counts:
            return np.nan
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


    @staticmethod
    def _derive_shrinkage_k(counts):
        median_counts = pd.to_numeric(counts, errors='coerce').median()
        if pd.isna(median_counts):
            return 1.0
        return max(1.0, float(median_counts) * 0.05)

    def _compute_bulk_shrinkage_k(self, summary_by_cell_type):


        if summary_by_cell_type and 'Bulk' in summary_by_cell_type:
            bulk_summary = summary_by_cell_type['Bulk'].copy()
            if 'n_reads' in bulk_summary.columns:
                return self._derive_shrinkage_k(bulk_summary['n_reads'])

        if not os.path.exists(self.isoform_agg_balance_path):
            return None

        iso_cache = getattr(self, '_isoform_agg_cache', None)
        if iso_cache is not None and self.isoform_agg_balance_path in iso_cache:
            iso_df = iso_cache[self.isoform_agg_balance_path]
        else:
            iso_df = pd.read_csv(self.isoform_agg_balance_path, index_col=0)
            if iso_cache is not None:
                iso_cache[self.isoform_agg_balance_path] = iso_df

        if not {'geneID', 'hapA', 'hapB'}.issubset(iso_df.columns):
            return None

        iso_df = iso_df.copy()
        if self.gene_subset is not None:
            iso_df = iso_df[iso_df['geneID'].isin(self.gene_subset)]
        if iso_df.empty:
            return 1.0

        total_counts = iso_df.groupby('geneID')[['hapA', 'hapB']].sum().sum(axis=1)
        return self._derive_shrinkage_k(total_counts)

    def _compute_effect_sizes(self, snv_df, cell_type='Bulk',
                              summary_df=None, ct_summary=None):
        if ct_summary is None:
            if summary_df is None:
                summary_df = pd.read_csv(self.summary_statistics_path)
            ct_summary = summary_df[summary_df['CellType'] == cell_type]

        ct_summary = ct_summary.copy()
        needed_cols = [
            'geneID', 'n_reads', 'n_reads_phasable', 'n_snvs',
            'alpha_hat', 'alpha_hat_low', 'alpha_hat_high',
            'alpha_hat_block_low', 'alpha_hat_block_high', 'n_phase_blocks',
            'major_hap', 'p_value', 'p_value_gene_adj',
            'p_value_isoform', 'p_value_isoform_high', 'p_value_isoform_low',
            'p_value_isoform_adj', 'p_value_isoform_adj_high', 'p_value_isoform_adj_low'
        ]
        for col in needed_cols:
            if col not in ct_summary.columns:
                if col == 'n_phase_blocks':


                    self._log(
                        f"[conf] {cell_type}: summary lacks n_phase_blocks "
                        f"(pre-step3) — multi-block genes without "
                        f"an astu_block_sums sidecar keep ANCHORED conf_astu; "
                        f"rerun step3 for flip-robust scores.")
                ct_summary[col] = np.nan

        ct_rows = ct_summary[needed_cols].drop_duplicates(subset=['geneID']).copy()
        ct_rows['n_reads'] = pd.to_numeric(ct_rows['n_reads'], errors='coerce')
        ct_rows['n_reads_phasable'] = pd.to_numeric(ct_rows['n_reads_phasable'], errors='coerce')
        ct_rows['gene_n_snvs'] = pd.to_numeric(ct_rows['n_snvs'], errors='coerce')
        gene_n_snvs_called_map = getattr(self, '_gene_n_snvs_called_map', {})
        ct_rows['gene_n_snvs_called'] = ct_rows['geneID'].map(gene_n_snvs_called_map)
        ct_rows['gene_n_snvs_called'] = pd.to_numeric(ct_rows['gene_n_snvs_called'], errors='coerce')
        ct_rows['gene_alpha_hat'] = pd.to_numeric(ct_rows['alpha_hat'], errors='coerce')
        ct_rows['gene_alpha_hat_low'] = pd.to_numeric(ct_rows['alpha_hat_low'], errors='coerce')
        ct_rows['gene_alpha_hat_high'] = pd.to_numeric(ct_rows['alpha_hat_high'], errors='coerce')
        ct_rows['gene_alpha_hat_block_low'] = pd.to_numeric(ct_rows['alpha_hat_block_low'], errors='coerce')
        ct_rows['gene_alpha_hat_block_high'] = pd.to_numeric(ct_rows['alpha_hat_block_high'], errors='coerce')
        ct_rows['gene_n_phase_blocks'] = pd.to_numeric(ct_rows['n_phase_blocks'], errors='coerce')
        ct_rows['gene_alpha_hat_major'] = 1 - ct_rows['gene_alpha_hat']
        ct_rows['gene_alpha_hat_major_low'] = 1 - ct_rows['gene_alpha_hat_high']
        ct_rows['gene_alpha_hat_major_high'] = 1 - ct_rows['gene_alpha_hat_low']
        ct_rows['gene_major_hap'] = ct_rows['major_hap']


        ct_rows['gene_minor_hap'] = np.where(
            ct_rows['gene_major_hap'] == 'A', 'B',
            np.where(ct_rows['gene_major_hap'] == 'B', 'A', 'nan')
        )
        ct_rows['gene_p_value'] = pd.to_numeric(ct_rows['p_value'], errors='coerce')
        ct_rows['gene_p_value_adj'] = pd.to_numeric(ct_rows['p_value_gene_adj'], errors='coerce')
        ct_rows['isoform_p_value'] = pd.to_numeric(ct_rows['p_value_isoform'], errors='coerce')
        ct_rows['isoform_p_value_high'] = pd.to_numeric(ct_rows['p_value_isoform_high'], errors='coerce')
        ct_rows['isoform_p_value_low'] = pd.to_numeric(ct_rows['p_value_isoform_low'], errors='coerce')
        ct_rows['isoform_p_value_adj'] = pd.to_numeric(ct_rows['p_value_isoform_adj'], errors='coerce')
        ct_rows['isoform_p_value_adj_high'] = pd.to_numeric(ct_rows['p_value_isoform_adj_high'], errors='coerce')
        ct_rows['isoform_p_value_adj_low'] = pd.to_numeric(ct_rows['p_value_isoform_adj_low'], errors='coerce')

        bulk_shrinkage_k = getattr(self, '_bulk_shrinkage_k', None)
        if bulk_shrinkage_k is None:
            shrinkage_k = self._derive_shrinkage_k(ct_rows['n_reads'])
        else:
            shrinkage_k = bulk_shrinkage_k


        alpha_mean = (ct_rows['gene_alpha_hat_low'] + ct_rows['gene_alpha_hat_high']) / 2.0
        alpha_mean = alpha_mean.where(alpha_mean.notna(), ct_rows['gene_alpha_hat'])
        alpha_point = ct_rows['gene_alpha_hat'].where(
            ct_rows['gene_alpha_hat'].notna(), alpha_mean)
        ct_rows['shrinkage_k'] = shrinkage_k
        ct_rows['n_reads_minor_hap'] = ct_rows['n_reads'] * alpha_mean + shrinkage_k
        ct_rows['n_reads_major_hap'] = (
            ct_rows['n_reads'] - ct_rows['n_reads'] * alpha_mean + shrinkage_k
        )
        ct_rows['es_ase_cons'] = np.log2(
            ct_rows['n_reads_major_hap'] / ct_rows['n_reads_minor_hap'].clip(lower=shrinkage_k)
        ).round(4)
        _minor_pt = ct_rows['n_reads'] * alpha_point + shrinkage_k
        _major_pt = ct_rows['n_reads'] - ct_rows['n_reads'] * alpha_point + shrinkage_k
        ct_rows['es_ase_point'] = np.log2(
            _major_pt / _minor_pt.clip(lower=shrinkage_k)
        ).round(4)


        ct_rows['es_ase'] = ct_rows['es_ase_cons']

        astu = self._compute_astu_effect(
            ct_rows[['geneID', 'gene_major_hap']].copy(),
            cell_type=cell_type
        )
        if astu is not None:
            ct_rows = ct_rows.merge(astu, on='geneID', how='left')
        else:
            ct_rows['overall_dominant_isoform'] = np.nan
            ct_rows['top_isoform_hap_major'] = np.nan
            ct_rows['top_isoform_hap_minor'] = np.nan
            ct_rows['top_isoform_hap_major_frac'] = np.nan
            ct_rows['top_isoform_hap_minor_frac'] = np.nan
            ct_rows['overall_dominant_frac_hap_major'] = np.nan
            ct_rows['overall_dominant_frac_hap_minor'] = np.nan
            ct_rows['overall_dominant_pref_hap'] = np.nan
            ct_rows['es_astu'] = np.nan
            ct_rows['es_astu_cons'] = np.nan
            ct_rows['astu_source'] = np.nan


        ct_rows['es_astu_point'] = ct_rows['es_astu']


        astu_conf = self._compute_conf_astu(
            ct_rows['geneID'].dropna().unique(), cell_type=cell_type,
            n_blocks_map=dict(zip(ct_rows['geneID'], ct_rows['gene_n_phase_blocks'])))
        ct_rows['conf_astu'] = ct_rows['geneID'].map(astu_conf)
        legacy = getattr(self, 'conf_nonphasable', None)
        theta_astu = legacy if legacy is not None else getattr(
            self, 'conf_nonphasable_astu', 1.0)


        margin = float(getattr(self, 'ase_call_margin', 0.095))
        _lo = ct_rows['gene_alpha_hat_low']
        _hi = ct_rows['gene_alpha_hat_high']
        alpha_mid = (_lo + _hi) / 2.0
        door_evidence = (_hi < 0.5) | (_lo > 0.5)
        door_margin = (alpha_mid - 0.5).abs() >= margin


        _p_ase = pd.to_numeric(ct_rows['gene_p_value_adj'], errors='coerce')
        _p_astu = pd.to_numeric(ct_rows['isoform_p_value_adj'], errors='coerce')
        ct_rows['ASE_call'] = np.select(
            [
                (door_evidence | door_margin) & (_p_ase <= 0.05),
                _p_ase > 0.05,
            ],
            [1, -1],
            default=0
        ).astype(int)
        ct_rows['ASTU_call'] = np.select(
            [
                (ct_rows['conf_astu'] >= theta_astu) & (_p_astu <= 0.05),
                _p_astu > 0.05,
            ],
            [1, -1],
            default=0
        ).astype(int)

        gene_effect_cols = [
            'geneID',
            'n_reads', 'n_reads_phasable', 'gene_n_snvs', 'gene_n_snvs_called',
            'gene_alpha_hat', 'gene_alpha_hat_low', 'gene_alpha_hat_high',
            'gene_alpha_hat_block_low', 'gene_alpha_hat_block_high',
            'gene_alpha_hat_major', 'gene_alpha_hat_major_low', 'gene_alpha_hat_major_high',
            'gene_major_hap', 'gene_minor_hap', 'gene_n_phase_blocks',
            'gene_p_value', 'gene_p_value_adj',
            'ASE_call', 'ASTU_call',
            'overall_dominant_isoform',
            'top_isoform_hap_major', 'top_isoform_hap_minor',
            'top_isoform_hap_major_frac', 'top_isoform_hap_minor_frac',
            'overall_dominant_frac_hap_major', 'overall_dominant_frac_hap_minor',
            'isoform_p_value', 'isoform_p_value_high', 'isoform_p_value_low',
            'isoform_p_value_adj', 'isoform_p_value_adj_high', 'isoform_p_value_adj_low',
            'shrinkage_k',
            'es_ase_point', 'es_ase_cons', 'es_ase',
            'es_astu_point', 'es_astu_cons', 'es_astu', 'astu_source',
            'conf_astu',
            'overall_dominant_pref_hap'
        ]
        result = snv_df.merge(ct_rows[gene_effect_cols], on='geneID', how='left')


        result['snv_hap'] = np.where(result['h_A'] > 0.5, 'A', 'B')
        result['snv_on_minor_hap'] = pd.array([pd.NA] * len(result), dtype=pd.BooleanDtype())
        result['snv_expr_direction'] = pd.array([pd.NA] * len(result), dtype=pd.StringDtype())
        result['snv_es_ase_signed'] = np.nan
        result['snv_astu_direction'] = pd.array([pd.NA] * len(result), dtype=pd.StringDtype())
        result['snv_es_astu_signed'] = np.nan

        major_mask = result['gene_major_hap'].isin(['A', 'B'])
        result.loc[major_mask, 'snv_on_minor_hap'] = (
            result.loc[major_mask, 'snv_hap'] != result.loc[major_mask, 'gene_major_hap']
        )
        result.loc[major_mask, 'snv_expr_direction'] = np.where(
            result.loc[major_mask, 'snv_hap'] == result.loc[major_mask, 'gene_major_hap'],
            '+',
            '-'
        )
        ase_mask = major_mask & result['es_ase'].notna()
        result.loc[ase_mask, 'snv_es_ase_signed'] = np.where(
            result.loc[ase_mask, 'snv_hap'] == result.loc[ase_mask, 'gene_major_hap'],
            result.loc[ase_mask, 'es_ase'],
            -result.loc[ase_mask, 'es_ase']
        )

        dom_mask = result['overall_dominant_pref_hap'].isin(['A', 'B'])
        result.loc[dom_mask, 'snv_astu_direction'] = np.where(
            result.loc[dom_mask, 'snv_hap'] == result.loc[dom_mask, 'overall_dominant_pref_hap'],
            '+',
            '-'
        )
        astu_mask = dom_mask & result['es_astu'].notna()
        result.loc[astu_mask, 'snv_es_astu_signed'] = np.where(
            result.loc[astu_mask, 'snv_hap'] == result.loc[astu_mask, 'overall_dominant_pref_hap'],
            result.loc[astu_mask, 'es_astu'],
            -result.loc[astu_mask, 'es_astu']
        )

        return result


    _ACCOUNT_SIG = 0.05

    def _conf_astu_gene(self, unbal_g, worst_g):
        idx = unbal_g.index.union(worst_g.index)
        uA = unbal_g['hapA'].reindex(idx).fillna(0.0).astype(float).to_numpy()
        uB = unbal_g['hapB'].reindex(idx).fillna(0.0).astype(float).to_numpy()
        bA = worst_g['hapA'].reindex(idx).fillna(0.0).astype(float).to_numpy()
        bB = worst_g['hapB'].reindex(idx).fillna(0.0).astype(float).to_numpy()

        def pv(t):
            a = (1.0 - t) * uA + t * bA
            b = (1.0 - t) * uB + t * bB
            keep = (a + b) > 0
            if keep.sum() < 2 or a[keep].sum() <= 0 or b[keep].sum() <= 0:
                return 1.0
            out = _Hap._chisq_test(
                pd.DataFrame({'hapA': a[keep], 'hapB': b[keep]}))
            p = out.get('p_value_isoform')
            return 1.0 if p is None or pd.isna(p) else float(p)

        p0 = pv(0.0)
        if not np.isfinite(p0):
            return np.nan
        if p0 > self._ACCOUNT_SIG:
            return 0.0
        if pv(1.0) <= self._ACCOUNT_SIG:
            return 1.0
        try:
            return float(brentq(lambda x: pv(x) - self._ACCOUNT_SIG,
                                0.0, 1.0, xtol=1e-4))
        except ValueError:


            return np.nan

    def _conf_astu_orientation_robust(self, gene_id, n_blocks):
        import glob
        pattern = os.path.join(self.count_dir, 'all_genes_isoform_separate',
                               f'*_{gene_id}_astu_block_sums.csv')
        hits = glob.glob(pattern)
        if not hits:
            return None
        sums = pd.read_csv(hits[0])
        import itertools


        flippable = sorted(set(sums.loc[sums['read_block'] > 0, 'read_block']))[1:]
        if 2 ** len(flippable) > _Hap.MAX_ASTU_ORIENTATIONS:
            return np.nan
        cs = []
        for mask in itertools.product((False, True), repeat=len(flippable)):
            flip = {b for b, f in zip(flippable, mask) if f}
            tab_o, bal_o = _Hap._tables_for_orientation(sums, flip)
            unbal_o = _Hap._tab_min_p(tab_o, bal_o)
            worst_o = _Hap._tab_max_p(tab_o, bal_o)
            c = self._conf_astu_gene(unbal_o, worst_o)
            if pd.isna(c):
                return np.nan
            cs.append(c)
        return min(cs) if cs else None

    def _compute_conf_astu(self, gene_ids, cell_type='Bulk', n_blocks_map=None):
        paths = {}
        all_genes_dir = os.path.join(self.count_dir, 'all_genes')
        if cell_type != 'Bulk':
            safe_ct = cell_type.replace('/', '_').replace(' ', '_')
            paths['unbal'] = os.path.join(
                all_genes_dir, f'ct_{safe_ct}_isoform_agg_unbalance.csv')
            paths['bal'] = os.path.join(
                all_genes_dir, f'ct_{safe_ct}_isoform_agg_balance.csv')
            paths['ph'] = os.path.join(
                all_genes_dir, f'ct_{safe_ct}_isoform_agg.csv')
            pmax_path = os.path.join(
                all_genes_dir, f'ct_{safe_ct}_isoform_agg_pmax.csv')
        else:
            paths['unbal'] = getattr(self, 'isoform_agg_unbalance_path', None) or \
                os.path.join(all_genes_dir, 'isoform_agg_unbalance.csv')
            paths['bal'] = getattr(self, 'isoform_agg_balance_path', None) or \
                os.path.join(all_genes_dir, 'isoform_agg_balance.csv')


            paths['ph'] = getattr(self, 'isoform_agg_path', None) or \
                os.path.join(all_genes_dir, 'isoform_agg.csv')
            pmax_path = getattr(self, 'isoform_agg_pmax_path', None) or \
                os.path.join(all_genes_dir, 'isoform_agg_pmax.csv')
        frames = {}
        iso_cache = getattr(self, '_isoform_agg_cache', None)
        for key, path in paths.items():
            if path is None or not os.path.exists(path):
                self._log(
                    f"[conf] conf_astu SKIPPED for {cell_type}: "
                    f"endpoint table missing ({path}) — column will be NaN.")
                return {}
            if iso_cache is not None and path in iso_cache:
                df = iso_cache[path]
            else:
                df = pd.read_csv(path, index_col=0)
                if iso_cache is not None:
                    iso_cache[path] = df
            if not {'geneID', 'hapA', 'hapB'}.issubset(df.columns):
                return {}
            frames[key] = df


        pmax_by_gene = None
        if pmax_path is not None and os.path.exists(pmax_path):
            if iso_cache is not None and pmax_path in iso_cache:
                pm = iso_cache[pmax_path]
            else:
                pm = pd.read_csv(pmax_path, index_col=0)
                if iso_cache is not None:
                    iso_cache[pmax_path] = pm
            if {'geneID', 'hapA', 'hapB'}.issubset(pm.columns):
                pmax_by_gene = dict(tuple(pm[pm['geneID'].isin(set(gene_ids))]
                                          .groupby('geneID', sort=False)))
        if pmax_by_gene is None:
            self._log(
                f"[conf] conf_astu for {cell_type}: max-p table not found "
                f"({pmax_path}); this step4 output predates its writing — "
                f"reconstructing the endpoint on the fly.")
        wanted = set(gene_ids)
        unbal = frames['unbal'][frames['unbal']['geneID'].isin(wanted)]
        bal = frames['bal'][frames['bal']['geneID'].isin(wanted)]
        ph = frames['ph'][frames['ph']['geneID'].isin(wanted)]
        bal_by_gene = dict(tuple(bal.groupby('geneID', sort=False)))
        ph_by_gene = dict(tuple(ph.groupby('geneID', sort=False)))
        out = {}
        for gene_id, ug in unbal.groupby('geneID', sort=False):
            bg = bal_by_gene.get(gene_id)
            if bg is None:
                continue


            pg = ph_by_gene.get(gene_id)
            cols = ['hapA', 'hapB']


            wg = pmax_by_gene.get(gene_id) if pmax_by_gene is not None else None
            if wg is not None:
                worst = wg[cols]
            else:
                ph_part = (pg[cols] if pg is not None
                           else pd.DataFrame(columns=cols, dtype=float))
                worst = _Hap._tab_max_p(ph_part, bg[cols])
            out[gene_id] = self._conf_astu_gene(ug, worst)
        if cell_type == 'Bulk':
            nb_map = n_blocks_map or {}
            for gene_id in list(out.keys()):
                nb = nb_map.get(gene_id)
                nb = None if nb is None or pd.isna(nb) else int(nb)
                if nb is not None and 2 ** (nb - 1) > _Hap.MAX_ASTU_ORIENTATIONS:


                    self._log(
                        f"[conf] {gene_id}: {nb} phase blocks exceed the "
                        f"orientation ceiling — conf_astu NaN.")
                    out[gene_id] = np.nan
                    continue


                robust = self._conf_astu_orientation_robust(gene_id, nb or 0)
                if robust is not None:
                    out[gene_id] = robust
                    continue
                if nb is not None and nb > 1:
                    self._log(
                        f"[conf] {gene_id}: {nb} phase blocks but no usable "
                        f"astu_block_sums sidecar (step3 predates, "
                        f"or blocks exceed the orientation ceiling) — "
                        f"conf_astu NaN, ⛔ not the anchored value. Rerun "
                        f"step3 for a flip-robust score.")
                    out[gene_id] = np.nan
        return out

    def _compute_astu_effect(self, ct_rows_df, cell_type='Bulk'):


        if cell_type != 'Bulk':
            safe_ct = cell_type.replace('/', '_').replace(' ', '_')
            ct_path = os.path.join(
                self.count_dir, 'all_genes',
                f'ct_{safe_ct}_isoform_agg_extrap.csv'
            )
            if os.path.exists(ct_path):
                iso_path = ct_path
                astu_source = 'ct_specific'
            else:
                iso_path = self.isoform_agg_extrap_path
                astu_source = 'bulk_fallback'
        else:
            iso_path = self.isoform_agg_extrap_path
            astu_source = 'bulk'

        if not os.path.exists(iso_path):


            self._log(
                f"[astu] es_astu SKIPPED for {cell_type}: v2 extrapolation "
                f"table missing ({iso_path}). This step4 output predates the "
                f"v2 table — rerun step4, or backfill with "
                f"src/backfill_isoform_extrap.py (no EM rerun needed). "
                f"es_astu will be NaN in this output; do NOT compare it "
                f"against v2-era numbers.")
            return None

        iso_cache = getattr(self, '_isoform_agg_cache', None)
        if iso_cache is not None and iso_path in iso_cache:
            iso_df = iso_cache[iso_path]
        else:
            iso_df = pd.read_csv(iso_path, index_col=0)
            if iso_cache is not None:
                iso_cache[iso_path] = iso_df
        if not {'geneID', 'hapA', 'hapB'}.issubset(iso_df.columns):
            return None


        ph_df = None
        all_genes_dir = os.path.join(self.count_dir, 'all_genes')
        ph_candidates = []
        if cell_type != 'Bulk':
            safe_ct = cell_type.replace('/', '_').replace(' ', '_')
            ph_candidates += [
                os.path.join(all_genes_dir, f'ct_{safe_ct}_isoform_agg.csv.gz'),
                os.path.join(all_genes_dir, f'ct_{safe_ct}_isoform_agg.csv')]
        else:
            ph_candidates += [os.path.join(all_genes_dir, 'isoform_agg.csv.gz'),
                              os.path.join(all_genes_dir, 'isoform_agg.csv')]
        ph_path = next((p for p in ph_candidates if os.path.exists(p)), None)
        if ph_path is not None:
            if iso_cache is not None and ph_path in iso_cache:
                ph_df = iso_cache[ph_path]
            else:
                ph_df = pd.read_csv(ph_path, index_col=0)
                if iso_cache is not None:
                    iso_cache[ph_path] = ph_df
            if not {'geneID', 'hapA', 'hapB'}.issubset(ph_df.columns):
                ph_df = None
        if ph_df is None:
            self._log(
                f"[astu] es_astu_cons SKIPPED for {cell_type}: phasable "
                f"isoform_agg table missing — cons column will be NaN "
                f"(point column unaffected).")

        major_hap_by_gene = (
            ct_rows_df.drop_duplicates(subset=['geneID'])
            .set_index('geneID')['gene_major_hap']
            .to_dict()
        )
        wanted_genes = set(major_hap_by_gene.keys())
        iso_df = iso_df[iso_df['geneID'].isin(wanted_genes)].copy()
        if iso_df.empty:
            return None

        bulk_shrinkage_k = getattr(self, '_bulk_shrinkage_k', None)
        if bulk_shrinkage_k is None:
            total_counts = iso_df.groupby('geneID')[['hapA', 'hapB']].sum().sum(axis=1)
            shrinkage_k = self._derive_shrinkage_k(total_counts)
        else:
            shrinkage_k = bulk_shrinkage_k

        rows = []
        for gene_id, g in iso_df.groupby('geneID', sort=False):
            major_hap = major_hap_by_gene.get(gene_id)
            if major_hap not in {'A', 'B'}:
                continue

            total_A = float(g['hapA'].sum())
            total_B = float(g['hapB'].sum())
            if total_A <= 0 or total_B <= 0:
                continue

            g = g.copy()
            g['_total'] = g['hapA'] + g['hapB']
            n_isoforms = len(g)


            _real = g[~g.index.astype(str).str.endswith('_Other')]
            if _real.empty or float(_real['_total'].sum()) <= 0:


                rows.append({
                    'geneID': gene_id,
                    'overall_dominant_isoform': np.nan,
                    'top_isoform_hap_major': np.nan,
                    'top_isoform_hap_minor': np.nan,
                    'top_isoform_hap_major_frac': np.nan,
                    'top_isoform_hap_minor_frac': np.nan,
                    'overall_dominant_frac_hap_major': np.nan,
                    'overall_dominant_frac_hap_minor': np.nan,
                    'overall_dominant_pref_hap': np.nan,
                    'es_astu': np.nan,
                    'es_astu_cons': np.nan,
                    'astu_source': astu_source,
                })
                continue
            dominant_isoform = str(_real['_total'].idxmax())


            n_iso_scale = np.sqrt(n_isoforms)
            dom_frac_A = (
                float(g.loc[dominant_isoform, 'hapA']) + shrinkage_k
            ) / (total_A + shrinkage_k * n_iso_scale)
            dom_frac_B = (
                float(g.loc[dominant_isoform, 'hapB']) + shrinkage_k
            ) / (total_B + shrinkage_k * n_iso_scale)
            overall_dominant_pref_hap = 'A' if dom_frac_A >= dom_frac_B else 'B'


            dom_raw_frac_A = float(g.loc[dominant_isoform, 'hapA']) / total_A
            dom_raw_frac_B = float(g.loc[dominant_isoform, 'hapB']) / total_B
            top_isoform_A = str(g['hapA'].idxmax())
            top_isoform_B = str(g['hapB'].idxmax())
            top_isoform_A_frac = float(g.loc[top_isoform_A, 'hapA']) / total_A
            top_isoform_B_frac = float(g.loc[top_isoform_B, 'hapB']) / total_B

            if major_hap == 'A':
                top_iso_major, top_iso_minor = top_isoform_A, top_isoform_B
                top_frac_major, top_frac_minor = top_isoform_A_frac, top_isoform_B_frac
                dom_frac_major, dom_frac_minor = dom_raw_frac_A, dom_raw_frac_B
            else:
                top_iso_major, top_iso_minor = top_isoform_B, top_isoform_A
                top_frac_major, top_frac_minor = top_isoform_B_frac, top_isoform_A_frac
                dom_frac_major, dom_frac_minor = dom_raw_frac_B, dom_raw_frac_A

            dom_frac_higher = max(dom_frac_A, dom_frac_B)
            dom_frac_lower = min(dom_frac_A, dom_frac_B)
            es_astu = np.log2(dom_frac_higher / dom_frac_lower)


            es_astu_cons = np.nan
            if ph_df is not None:
                ph_g = ph_df[ph_df['geneID'] == gene_id]


                a = ph_g['hapA'].reindex(g.index).fillna(0.0).astype(float)
                b = ph_g['hapB'].reindex(g.index).fillna(0.0).astype(float)
                R = (g['hapA'] + g['hapB']).astype(float)
                U = (R - a - b).clip(lower=0.0)
                is_dom = (g.index == dominant_isoform)
                ends = []
                for dom_side_A in (True, False):
                    A_end = a + np.where(is_dom == dom_side_A, U, 0.0)
                    B_end = b + np.where(is_dom == dom_side_A, 0.0, U)


                    denA = float(A_end.sum()) + shrinkage_k * n_iso_scale
                    denB = float(B_end.sum()) + shrinkage_k * n_iso_scale
                    if denA <= 0 or denB <= 0:
                        ends = []
                        break
                    uA = (float(A_end.loc[dominant_isoform]) + shrinkage_k) / denA
                    uB = (float(B_end.loc[dominant_isoform]) + shrinkage_k) / denB
                    ends.append((uA, uB))
                if ends:
                    uA_mid = (ends[0][0] + ends[1][0]) / 2.0
                    uB_mid = (ends[0][1] + ends[1][1]) / 2.0
                    if uA_mid > 0 and uB_mid > 0:
                        es_astu_cons = np.log2(
                            max(uA_mid, uB_mid) / min(uA_mid, uB_mid))

            rows.append({
                'geneID': gene_id,
                'overall_dominant_isoform': dominant_isoform,
                'top_isoform_hap_major': top_iso_major,
                'top_isoform_hap_minor': top_iso_minor,
                'top_isoform_hap_major_frac': round(float(top_frac_major), 4),
                'top_isoform_hap_minor_frac': round(float(top_frac_minor), 4),
                'overall_dominant_frac_hap_major': round(float(dom_frac_major), 4),
                'overall_dominant_frac_hap_minor': round(float(dom_frac_minor), 4),
                'overall_dominant_pref_hap': overall_dominant_pref_hap,
                'es_astu': round(float(es_astu), 4),
                'es_astu_cons': (round(float(es_astu_cons), 4)
                                 if pd.notna(es_astu_cons) else np.nan),
                'astu_source': astu_source,
            })

        return pd.DataFrame(rows) if rows else None

    @staticmethod
    def _safe_divide(numerator, denominator):
        num = pd.to_numeric(numerator, errors='coerce')
        den = pd.to_numeric(denominator, errors='coerce')
        return num / den.where(den != 0)

    @staticmethod
    def _build_snv_id_series(chrom, pos, ref, alt):
        out = pd.Series(np.nan, index=chrom.index, dtype=object)
        pos_num = pd.to_numeric(pos, errors='coerce')
        mask = chrom.notna() & pos_num.notna() & ref.notna() & alt.notna()
        if mask.any():
            out.loc[mask] = (
                chrom.loc[mask].astype(str) + ':' +
                pos_num.loc[mask].astype(int).astype(str) + ':' +
                ref.loc[mask].astype(str) + ':' +
                alt.loc[mask].astype(str)
            )
        return out

    @staticmethod
    def _build_event_id_series(event_type, event_start, event_end):
        out = pd.Series(np.nan, index=event_type.index, dtype=object)
        start_num = pd.to_numeric(event_start, errors='coerce')
        end_num = pd.to_numeric(event_end, errors='coerce')
        mask = event_type.notna() & start_num.notna() & end_num.notna()
        if mask.any():
            out.loc[mask] = (
                event_type.loc[mask].astype(str) + ':' +
                start_num.loc[mask].astype(int).astype(str) + '-' +
                end_num.loc[mask].astype(int).astype(str)
            )
        return out

    @staticmethod
    def _ensure_output_columns(df, columns):
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        return df[columns]

    def _merge_subset_output(self, output_path, new_df):
        if not os.path.exists(output_path):
            return new_df

        existing_df = pd.read_csv(output_path)
        if existing_df.empty or 'geneID' not in existing_df.columns:
            if new_df is None:
                return existing_df
            return pd.concat([existing_df, new_df], ignore_index=True)

        subset_keys = pd.DataFrame({'geneID': sorted(self.gene_subset)}) if self.gene_subset else pd.DataFrame()
        if 'Sample' in existing_df.columns:
            subset_keys['Sample'] = getattr(self, '_log_sample_name', None)

        if subset_keys.empty:
            return existing_df if new_df is None else pd.concat([existing_df, new_df], ignore_index=True)


        match_cols = list(subset_keys.columns)
        if new_df is not None and 'CellType' in existing_df.columns and 'CellType' in new_df.columns:
            new_ct = new_df['CellType'].unique()
            replace_keys = subset_keys.assign(key=1).merge(
                pd.DataFrame({'CellType': new_ct, 'key': 1}), on='key'
            ).drop(columns='key')
            match_cols_ct = list(replace_keys.columns)
        else:
            replace_keys = subset_keys
            match_cols_ct = match_cols

        matched_existing = existing_df.merge(
            replace_keys.assign(_subset_replace=True),
            on=match_cols_ct,
            how='left'
        )
        filtered_existing_df = existing_df.loc[
            matched_existing['_subset_replace'].isna().to_numpy()
        ].copy()
        if new_df is None:
            return filtered_existing_df
        return pd.concat([filtered_existing_df, new_df], ignore_index=True)

    def _assemble_gene_snv_output(self, effect_df, sample_name, cell_type):
        output_cols = [
            'Sample', 'CellType',
            'geneID', 'geneName', 'geneChr',
            'n_reads', 'n_reads_phasable', 'gene_n_snvs', 'gene_n_snvs_called',
            'gene_alpha_hat', 'gene_alpha_hat_low', 'gene_alpha_hat_high',
            'gene_alpha_hat_block_low', 'gene_alpha_hat_block_high',
            'gene_alpha_hat_major', 'gene_alpha_hat_major_low', 'gene_alpha_hat_major_high',
            'gene_major_hap', 'gene_minor_hap', 'gene_n_phase_blocks',
            'gene_p_value', 'gene_p_value_adj',
            'ASE_call', 'ASTU_call',
            'overall_dominant_isoform',
            'top_isoform_hap_major', 'top_isoform_hap_minor',
            'top_isoform_hap_major_frac', 'top_isoform_hap_minor_frac',
            'overall_dominant_frac_hap_major', 'overall_dominant_frac_hap_minor',
            'isoform_p_value', 'isoform_p_value_high', 'isoform_p_value_low',
            'isoform_p_value_adj', 'isoform_p_value_adj_high', 'isoform_p_value_adj_low',
            'shrinkage_k',
            'es_ase_point', 'es_ase_cons', 'es_ase',
            'es_astu_point', 'es_astu_cons', 'es_astu',
            'conf_astu',
            'astu_source',
            'snvID',
            'snv_pos', 'snv_ref', 'snv_alt',
            'snv_depth_bulk', 'snv_alt_count_bulk', 'snv_alt_frac_bulk',
            'h_A', 'hat_Z_prob_revised',
            'snv_hap',
            'snv_on_minor_hap',
            'snv_expr_direction',
            'snv_es_ase_signed',
            'overall_dominant_pref_hap',
            'snv_astu_direction',
            'snv_es_astu_signed'
        ]
        if effect_df is None or effect_df.empty:
            return pd.DataFrame(columns=output_cols)

        df = effect_df.copy()
        df['Sample'] = sample_name
        df['CellType'] = cell_type
        df['geneChr'] = df['chrom']
        df['snv_pos'] = df['pos']
        df['snv_ref'] = df['ref']
        df['snv_alt'] = df['alt']
        df['snv_depth_bulk'] = df['depth']
        df['snv_alt_count_bulk'] = df['alt_count']
        df['snv_alt_frac_bulk'] = df['alt_frac']
        df['snvID'] = self._build_snv_id_series(df['chrom'], df['snv_pos'], df['snv_ref'], df['snv_alt'])

        return self._ensure_output_columns(df, output_cols)

    def _assemble_event_snv_output(self, hap_event_df, snv_event_df, chi_df,
                                   gene_snv_df, sample_name, cell_type):
        output_cols = [
            'Sample', 'CellType',
            'geneID', 'geneName', 'geneChr',
            'n_reads', 'n_reads_phasable', 'gene_n_snvs_called',
            'gene_major_hap', 'shrinkage_k', 'es_ase', 'es_astu',
            'ASE_call', 'ASTU_call',
            'overall_dominant_isoform', 'top_isoform_hap_major', 'top_isoform_hap_minor',
            'eventID',
            'event_type', 'event_start', 'event_end', 'event_length',
            'hapA_present', 'hapA_absent', 'hapB_present', 'hapB_absent',
            'obs_hapA_include', 'obs_hapA_skip', 'obs_hapA_unobserved',
            'obs_hapB_include', 'obs_hapB_skip', 'obs_hapB_unobserved',
            'obs_chi2', 'obs_p_value', 'obs_p_value_adj', 'obs_test_type',
            'event_inclusion_frac_A', 'event_inclusion_frac_B',
            'event_pref_hap',
            'event_pref_major_minor',
            'event_chi2', 'event_p_value', 'event_p_value_adj',
            'has_linked_snv', 'linked_snv_count', 'is_nearest_snv_for_event',
            'snvID', 'snv_pos', 'snv_ref', 'snv_alt',
            'snv_hap', 'h_A', 'hat_Z_prob_revised',
            'exonic_distance', 'genomic_distance',
            'snv_expr_direction', 'snv_astu_direction',
            'snv_event_direction',
            'raw_validation_available',
            'raw_ref_present', 'raw_ref_absent', 'raw_alt_present', 'raw_alt_absent',
            'raw_total_reads',
            'raw_chi2', 'raw_p_value', 'raw_p_value_adj', 'raw_test_type'
        ]
        if hap_event_df is None or hap_event_df.empty:
            return pd.DataFrame(columns=output_cols)

        merge_keys = ['geneID', 'event_type', 'event_start', 'event_end']
        snv_pair_cols = merge_keys + [
            'snv_pos', 'snv_ref', 'snv_alt', 'snv_hap',
            'h_A', 'hat_Z_prob_revised', 'exonic_distance', 'genomic_distance'
        ]
        if snv_event_df is None or snv_event_df.empty:
            snv_pair_df = pd.DataFrame(columns=snv_pair_cols)
        else:
            snv_pair_df = snv_event_df[snv_pair_cols].drop_duplicates().copy()

        event_df = hap_event_df.copy().merge(snv_pair_df, on=merge_keys, how='left')

        raw_cols = [
            'raw_ref_present', 'raw_ref_absent', 'raw_alt_present', 'raw_alt_absent',
            'raw_chi2', 'raw_p_value', 'raw_p_value_adj', 'raw_test_type'
        ]
        if chi_df is not None and not chi_df.empty:
            raw_merge_cols = merge_keys + ['snv_pos', 'snv_ref', 'snv_alt'] + raw_cols
            raw_merge_df = chi_df[raw_merge_cols].drop_duplicates().copy()
            event_df = event_df.merge(
                raw_merge_df,
                on=merge_keys + ['snv_pos', 'snv_ref', 'snv_alt'],
                how='left'
            )
        else:
            for col in raw_cols:
                event_df[col] = np.nan

        if gene_snv_df is not None and not gene_snv_df.empty:
            gene_ctx = gene_snv_df[
                ['geneID', 'n_reads', 'n_reads_phasable', 'gene_n_snvs_called',
                 'gene_major_hap', 'shrinkage_k', 'es_ase',
                 'es_astu', 'ASE_call', 'ASTU_call',
                 'overall_dominant_isoform', 'top_isoform_hap_major',
                 'top_isoform_hap_minor']
            ].drop_duplicates(subset=['geneID'])
            event_df = event_df.merge(gene_ctx, on='geneID', how='left')

            snv_ctx = gene_snv_df[
                ['geneID', 'snv_pos', 'snv_ref', 'snv_alt', 'snvID',
                 'snv_expr_direction', 'snv_astu_direction']
            ].drop_duplicates()
            event_df = event_df.merge(
                snv_ctx,
                on=['geneID', 'snv_pos', 'snv_ref', 'snv_alt'],
                how='left'
            )
        else:
            for col in ['n_reads', 'n_reads_phasable', 'gene_n_snvs_called',
                        'gene_major_hap', 'shrinkage_k', 'es_ase',
                        'es_astu', 'ASE_call', 'ASTU_call',
                        'overall_dominant_isoform', 'top_isoform_hap_major',
                        'top_isoform_hap_minor', 'snvID',
                        'snv_expr_direction', 'snv_astu_direction']:
                event_df[col] = np.nan

        if 'snvID' not in event_df.columns:
            event_df['snvID'] = np.nan
        snv_id_fallback = self._build_snv_id_series(
            event_df['geneChr'], event_df['snv_pos'], event_df['snv_ref'], event_df['snv_alt']
        )
        event_df['snvID'] = event_df['snvID'].where(event_df['snvID'].notna(), snv_id_fallback)

        event_df['Sample'] = sample_name
        event_df['CellType'] = cell_type
        event_df['eventID'] = self._build_event_id_series(
            event_df['event_type'], event_df['event_start'], event_df['event_end']
        )
        event_df['event_length'] = (
            pd.to_numeric(event_df['event_end'], errors='coerce') -
            pd.to_numeric(event_df['event_start'], errors='coerce')
        )

        event_df['event_inclusion_frac_A'] = self._safe_divide(
            event_df['hapA_present'],
            pd.to_numeric(event_df['hapA_present'], errors='coerce') +
            pd.to_numeric(event_df['hapA_absent'], errors='coerce')
        ).round(4)
        event_df['event_inclusion_frac_B'] = self._safe_divide(
            event_df['hapB_present'],
            pd.to_numeric(event_df['hapB_present'], errors='coerce') +
            pd.to_numeric(event_df['hapB_absent'], errors='coerce')
        ).round(4)

        event_df['event_pref_hap'] = pd.array([pd.NA] * len(event_df), dtype=pd.StringDtype())
        pref_mask = event_df['event_inclusion_frac_A'].notna() & event_df['event_inclusion_frac_B'].notna()
        event_df.loc[pref_mask, 'event_pref_hap'] = np.where(
            event_df.loc[pref_mask, 'event_inclusion_frac_A'] > event_df.loc[pref_mask, 'event_inclusion_frac_B'],
            'A', 'B'
        )

        event_df['event_pref_major_minor'] = pd.array([pd.NA] * len(event_df), dtype=pd.StringDtype())
        mm_mask = event_df['event_pref_hap'].notna() & event_df['gene_major_hap'].notna()
        event_df.loc[mm_mask, 'event_pref_major_minor'] = np.where(
            event_df.loc[mm_mask, 'event_pref_hap'] == event_df.loc[mm_mask, 'gene_major_hap'],
            'major', 'minor'
        )

        event_df['linked_snv_count'] = (
            event_df.groupby(merge_keys, dropna=False)['snv_pos']
            .transform('count').fillna(0).astype(int)
        )
        event_df['has_linked_snv'] = event_df['linked_snv_count'] > 0
        event_df['is_nearest_snv_for_event'] = False

        linked_mask = event_df['snv_pos'].notna()
        if linked_mask.any():
            ranked = event_df.loc[linked_mask].sort_values(
                merge_keys + ['exonic_distance', 'genomic_distance', 'snv_pos']
            )
            nearest_idx = ranked.groupby(merge_keys, sort=False).head(1).index
            event_df.loc[nearest_idx, 'is_nearest_snv_for_event'] = True

        event_df['snv_event_direction'] = pd.array([pd.NA] * len(event_df), dtype=pd.StringDtype())
        ed_mask = event_df['snv_hap'].notna() & event_df['event_pref_hap'].notna()
        event_df.loc[ed_mask, 'snv_event_direction'] = np.where(
            event_df.loc[ed_mask, 'snv_hap'] == event_df.loc[ed_mask, 'event_pref_hap'],
            'promotes_event', 'reduces_event'
        )

        event_df['raw_validation_available'] = event_df['raw_ref_present'].notna()
        event_df['raw_total_reads'] = event_df[
            ['raw_ref_present', 'raw_ref_absent', 'raw_alt_present', 'raw_alt_absent']
        ].sum(axis=1, min_count=1)

        event_df['event_chi2'] = event_df['chi2']
        event_df['event_p_value'] = event_df['p_value']
        event_df['event_p_value_adj'] = event_df['p_value_adj']

        return self._ensure_output_columns(event_df, output_cols)


    def _haplotype_event_associations(self, min_reads=10, cell_type='Bulk',
                                       scotch_read_cell=None, read_hap_df=None,
                                       scotch_isoform_df=None, summary_df=None,
                                       ct_summary=None, bulk_sig_gene_ids=None,
                                       event_mode='all_events',
                                       fdr_events_value=0.05):
        for path, label in [(self.read_hap_map_path, 'read_hap_map'),
                             (self.scotch_tsv_path, 'SCOTCH TSV')]:
            if not os.path.exists(path):
                self._log(f'{label} not found: {path}')
                return None
        if self.gsi is None:
            self._log('geneStructureInformation not loaded')
            return None

        read_hap = pd.read_csv(self.read_hap_map_path) if read_hap_df is None else read_hap_df


        if cell_type != 'Bulk' and scotch_read_cell is not None and self.cell_type_df is not None:
            ct_cells = set(
                self.cell_type_df.loc[self.cell_type_df['CellType'] == cell_type, 'Cell'])
            ct_reads = scotch_read_cell.loc[
                scotch_read_cell['Cell'].isin(ct_cells), ['Read', 'geneID']]
            read_hap = read_hap.merge(ct_reads, on=['Read', 'geneID'], how='inner')

        if self.astu_sig_from_bulk:
            total_genes = int(read_hap['geneID'].nunique())
            sig_gene_ids = bulk_sig_gene_ids
            if sig_gene_ids is None:
                sig_gene_ids = self._get_significant_astu_gene_ids(
                    cell_type='Bulk', summary_df=summary_df)
            read_hap = read_hap[read_hap['geneID'].isin(sig_gene_ids)].copy()
            kept_genes = int(read_hap['geneID'].nunique())
            self._log(
                '  Task 4 ASTU significance filter '
                f'(Bulk->{cell_type}): kept {kept_genes} of {total_genes} genes '
                f'(threshold={self.astu_sig_threshold}).'
            )
            if read_hap.empty:
                return None
        elif self.astu_sig_only:
            total_genes = int(read_hap['geneID'].nunique())
            sig_gene_ids = self._get_significant_astu_gene_ids(
                cell_type=cell_type,
                summary_df=summary_df,
                ct_summary=ct_summary)
            read_hap = read_hap[read_hap['geneID'].isin(sig_gene_ids)].copy()
            kept_genes = int(read_hap['geneID'].nunique())
            self._log(
                '  Task 4 ASTU significance filter '
                f'({cell_type}): kept {kept_genes} of {total_genes} genes '
                f'(threshold={self.astu_sig_threshold}).'
            )
            if read_hap.empty:
                return None


        read_hap = read_hap[['geneID', 'Read', 'hat_I', 'hat_I_B']].copy()
        relevant_gene_ids = read_hap['geneID'].dropna().unique().tolist()
        if not relevant_gene_ids:
            return None
        relevant_gene_set = set(relevant_gene_ids)

        self._log(
            f'  Task 4a: preparing per-gene SCOTCH joins for '
            f'{len(relevant_gene_ids)} candidate genes ({cell_type}).')


        if scotch_isoform_df is None:
            chunks = pd.read_csv(
                self.scotch_tsv_path, sep='\t', chunksize=100_000,
                usecols=lambda c: c in {'Read', 'geneID', 'Isoform', 'Keep'})
            frames = []
            for chunk in chunks:
                sub = chunk.loc[
                    (chunk['Keep'] == 1) & (chunk['geneID'].isin(relevant_gene_set)),
                    ['Read', 'geneID', 'Isoform']].copy()
                sub['Read'] = sub['Read'].map(_canonicalize_read_name)
                frames.append(sub)
            frames = [frame for frame in frames if not frame.empty]
            if not frames:
                return None
            scotch_df = pd.concat(frames, ignore_index=True).drop_duplicates()
        else:
            scotch_df = scotch_isoform_df.loc[
                scotch_isoform_df['geneID'].isin(relevant_gene_set),
                ['Read', 'geneID', 'Isoform']]
            if scotch_df.empty:
                return None


        scotch_by_gene = getattr(self, '_scotch_by_gene_cache', {})
        if not scotch_by_gene:
            scotch_by_gene = {
                gene_id: sub[['Read', 'Isoform']].set_index('Read')
                for gene_id, sub in scotch_df.groupby('geneID', sort=False)
            }
        candidate_gene_ids = [gid for gid in relevant_gene_ids if gid in scotch_by_gene]
        if not candidate_gene_ids:
            return None

        n_test_genes = len(candidate_gene_ids)
        self._log(f'  Task 4a: testing {n_test_genes} genes for haplotype–event associations ({cell_type}).')


        variant_dir_for_obs = getattr(self, 'variant_dir', None)


        _jobs = []
        for gene_id, g in read_hap.groupby('geneID', sort=False):
            if gene_id not in scotch_by_gene:
                continue
            _cache = self._get_gene_event_cache(gene_id)
            if _cache is None:
                continue
            _jobs.append(delayed(_process_gene_events_joined)(
                gene_id,
                g[['Read', 'hat_I', 'hat_I_B']],
                scotch_by_gene[gene_id],
                None,
                min_reads,
                gene_event_cache=_cache,
                variant_dir=variant_dir_for_obs))
        _blas_vars = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                      'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
        _saved_env = {}
        try:


            if self.n_workers != 1:
                for _v in _blas_vars:
                    _saved_env[_v] = os.environ.get(_v)
                    os.environ[_v] = '1'
                _par = Parallel(n_jobs=self.n_workers, backend='loky',
                                batch_size='auto')
            else:
                _par = Parallel(n_jobs=1, prefer='threads', batch_size='auto')
            results_per_gene = _par(_jobs)
        finally:
            for _v, _val in _saved_env.items():
                if _val is None:
                    os.environ.pop(_v, None)
                else:
                    os.environ[_v] = _val
        rows = [r for gene_rows in results_per_gene for r in gene_rows]

        if not rows:
            return None

        result = pd.DataFrame(rows)


        result['obs_p_value_adj'] = np.nan
        for _, grp in result.groupby('geneID', sort=False):
            result.loc[grp.index, 'p_value_adj'] = multipletests(
                grp['p_value'], method='fdr_bh')[1]
            obs_valid_idx = grp.index[grp['obs_p_value'].notna()]
            if len(obs_valid_idx):
                result.loc[obs_valid_idx, 'obs_p_value_adj'] = multipletests(
                    result.loc[obs_valid_idx, 'obs_p_value'].to_numpy(dtype=float),
                    method='fdr_bh',
                )[1]

        result = result.sort_values(['geneID', 'p_value_adj']).reset_index(drop=True)
        result = self._filter_haplotype_events(
            result,
            cell_type=cell_type,
            event_mode=event_mode,
            fdr_events_value=fdr_events_value,
        )
        if result is None or result.empty:
            return None
        return result

    def _get_significant_astu_gene_ids(self, cell_type='Bulk',
                                       summary_df=None, ct_summary=None):
        if ct_summary is None:
            if summary_df is None:
                if not os.path.exists(self.summary_statistics_path):
                    self._log(
                        f'summary_statistics.csv not found for ASTU significance filter: '
                        f'{self.summary_statistics_path}'
                    )
                    return set()

                summary_df = pd.read_csv(
                    self.summary_statistics_path,
                    usecols=['geneID', 'CellType', 'p_value_isoform_adj']
                )
            ct_summary = summary_df[summary_df['CellType'] == cell_type].copy()
        else:
            ct_summary = ct_summary[['geneID', 'p_value_isoform_adj']].copy()

        if ct_summary.empty:
            return set()

        ct_summary['p_value_isoform_adj'] = pd.to_numeric(
            ct_summary['p_value_isoform_adj'], errors='coerce'
        )
        sig_gene_ids = ct_summary.loc[
            ct_summary['p_value_isoform_adj'] <= self.astu_sig_threshold,
            'geneID'
        ].dropna().unique()
        return set(sig_gene_ids)

    def _filter_haplotype_events(self, hap_event_df, cell_type='Bulk',
                                 event_mode='all_events',
                                 fdr_events_value=0.05):
        if hap_event_df is None or hap_event_df.empty:
            return hap_event_df
        if event_mode == 'all_events':
            return hap_event_df
        if event_mode == 'fdr_events':
            mask = pd.to_numeric(
                hap_event_df['p_value_adj'], errors='coerce'
            ) <= float(fdr_events_value)
            filtered = hap_event_df.loc[mask].copy()
            self._log(
                f'  Task 4a event filter ({cell_type}, fdr_events<={fdr_events_value}): '
                f'kept {len(filtered)} of {len(hap_event_df)} events.'
            )
            return filtered.reset_index(drop=True)
        if event_mode != 'switching_events':
            raise ValueError(f'Unsupported event_mode: {event_mode}')

        keep_indices = []
        gene_kept = 0
        for gene_id, grp in hap_event_df.groupby('geneID', sort=False):
            gene_name = None
            if 'geneName' in grp.columns and not grp['geneName'].dropna().empty:
                gene_name = str(grp['geneName'].dropna().iloc[0])
            iso_pair = self._identify_switching_pair(
                gene_id=gene_id, gene_name=gene_name, cell_type=cell_type
            )
            if not iso_pair:
                continue
            iso_reduced, iso_increased = iso_pair
            boundary_events = self._find_switching_boundary_events(
                gene_id, iso_reduced, iso_increased
            )
            if not boundary_events:
                continue
            event_keys = list(
                zip(grp['event_type'], grp['event_start'], grp['event_end'])
            )
            grp_keep = [idx for idx, key in zip(grp.index, event_keys) if key in boundary_events]
            if grp_keep:
                gene_kept += 1
                keep_indices.extend(grp_keep)

        filtered = hap_event_df.loc[sorted(keep_indices)].copy()
        self._log(
            f'  Task 4a event filter ({cell_type}, switching_events): '
            f'kept {len(filtered)} of {len(hap_event_df)} events across {gene_kept} genes.'
        )
        return filtered.reset_index(drop=True)

    def _resolve_isoform_agg_path(self, cell_type='Bulk'):
        all_genes_dir = os.path.join(self.count_dir, 'all_genes')
        candidates = []
        if cell_type != 'Bulk':
            safe_ct = cell_type.replace('/', '_').replace(' ', '_')
            candidates.extend([
                os.path.join(all_genes_dir, f'ct_{safe_ct}_isoform_agg.csv.gz'),
                os.path.join(all_genes_dir, f'ct_{safe_ct}_isoform_agg.csv'),
                os.path.join(all_genes_dir, f'ct_{safe_ct}_isoform_agg_balance.csv.gz'),
                os.path.join(all_genes_dir, f'ct_{safe_ct}_isoform_agg_balance.csv'),
            ])
        candidates.extend([
            os.path.join(all_genes_dir, 'isoform_agg.csv.gz'),
            os.path.join(all_genes_dir, 'isoform_agg.csv'),
            os.path.join(all_genes_dir, 'isoform_agg_balance.csv.gz'),
            os.path.join(all_genes_dir, 'isoform_agg_balance.csv'),
        ])
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _load_isoform_agg_table(self, cell_type='Bulk'):
        iso_path = self._resolve_isoform_agg_path(cell_type=cell_type)
        if iso_path is None:
            self._log(
                f'Isoform aggregate table not found for switching-event filtering '
                f'({cell_type}).'
            )
            return None
        iso_cache = getattr(self, '_isoform_agg_cache', None)
        if iso_cache is not None and iso_path in iso_cache:
            return iso_cache[iso_path]
        iso_df = pd.read_csv(iso_path, index_col=0)
        if iso_cache is not None:
            iso_cache[iso_path] = iso_df
        return iso_df

    @staticmethod
    def _extract_transcript_id(label):
        if label is None or pd.isna(label):
            return None
        match = re.search(r'ENST\d+', str(label))
        return match.group(0) if match else None

    def _identify_switching_pair(self, gene_id, gene_name=None, cell_type='Bulk'):
        iso_df = self._load_isoform_agg_table(cell_type=cell_type)
        if iso_df is None or iso_df.empty:
            return None
        if 'geneID' not in iso_df.columns or 'hapA' not in iso_df.columns or 'hapB' not in iso_df.columns:
            return None

        gene_df = iso_df.loc[iso_df['geneID'] == gene_id].copy()
        if gene_df.empty:
            return None
        gene_df['transcript_id'] = [
            self._extract_transcript_id(idx) for idx in gene_df.index.astype(str)
        ]
        gene_df = gene_df[gene_df['transcript_id'].notna()].copy()
        if gene_df.empty:
            return None

        grouped = (
            gene_df.groupby('transcript_id', sort=False)[['hapA', 'hapB']]
            .sum()
            .reset_index()
        )
        total_A = float(pd.to_numeric(grouped['hapA'], errors='coerce').sum())
        total_B = float(pd.to_numeric(grouped['hapB'], errors='coerce').sum())
        if total_A <= 0 or total_B <= 0 or len(grouped) < 2:
            return None

        grouped['frac_A'] = grouped['hapA'] / total_A
        grouped['frac_B'] = grouped['hapB'] / total_B
        grouped['delta'] = grouped['frac_B'] - grouped['frac_A']
        grouped = grouped.sort_values(['delta', 'transcript_id']).reset_index(drop=True)
        iso_reduced = str(grouped.iloc[0]['transcript_id'])
        iso_increased = str(grouped.iloc[-1]['transcript_id'])
        if iso_reduced == iso_increased:
            return None
        return iso_reduced, iso_increased

    @staticmethod
    def _merge_intervals(intervals, merge_adjacent=0):
        cleaned = sorted(
            {(int(start), int(end)) for start, end in intervals if end > start},
            key=lambda x: (x[0], x[1])
        )
        if not cleaned:
            return []
        merged = [list(cleaned[0])]
        for start, end in cleaned[1:]:
            if start <= merged[-1][1] + int(merge_adjacent):
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [tuple(x) for x in merged]

    @staticmethod
    def _sort_intervals(intervals):
        return sorted(
            {(int(start), int(end)) for start, end in intervals if end > start},
            key=lambda x: (x[0], x[1])
        )

    def _get_transcript_structures(self, gene_id):
        gene_event_cache = self._get_gene_event_cache(gene_id)
        if gene_event_cache is None:
            return {}
        transcript_structures = gene_event_cache.get('transcript_structures')
        if transcript_structures is not None:
            return transcript_structures

        transcript_structures = {}
        for iso_name, exons in gene_event_cache['iso_exon_map'].items():
            transcript_id = self._extract_transcript_id(iso_name)
            if transcript_id is None:
                continue
            raw_exons = self._sort_intervals(list(exons))
            merged_exons = self._merge_intervals(raw_exons)
            junctions = self._sort_intervals(
                list(gene_event_cache['iso_junction_map'].get(iso_name, set()))
            )
            struct = transcript_structures.setdefault(
                transcript_id,
                {'isoform_keys': [], 'event_exons': [], 'merged_exons': [], 'junctions': []}
            )
            struct['isoform_keys'].append(iso_name)
            struct['event_exons'].extend(raw_exons)
            struct['merged_exons'].extend(merged_exons)
            struct['junctions'].extend(junctions)

        for transcript_id, struct in transcript_structures.items():
            struct['event_exons'] = self._sort_intervals(struct['event_exons'])
            struct['merged_exons'] = self._merge_intervals(struct['merged_exons'])
            struct['junctions'] = self._sort_intervals(struct['junctions'])

        gene_event_cache['transcript_structures'] = transcript_structures
        return transcript_structures

    def _find_switching_boundary_events(self, gene_id, iso_reduced, iso_increased):
        transcript_structures = self._get_transcript_structures(gene_id)
        struct_a = transcript_structures.get(iso_reduced)
        struct_b = transcript_structures.get(iso_increased)
        gene_event_cache = self._get_gene_event_cache(gene_id)
        if struct_a is None or struct_b is None or gene_event_cache is None:
            return set()

        event_exons_a = self._sort_intervals(struct_a.get('event_exons', []))
        event_exons_b = self._sort_intervals(struct_b.get('event_exons', []))
        junctions_a = self._sort_intervals(struct_a.get('junctions', []))
        junctions_b = self._sort_intervals(struct_b.get('junctions', []))
        if not event_exons_a or not event_exons_b:
            return set()

        def build_event_set(exons, junctions):
            events = {('exon', int(start), int(end)) for start, end in exons}
            events.update(('junction', int(start), int(end)) for start, end in junctions)
            return events

        def build_boundary_sides(exons, junctions):
            left = {}
            right = {}
            for event_type, events in (('exon', exons), ('junction', junctions)):
                for start, end in events:
                    event_key = (event_type, int(start), int(end))
                    left.setdefault(int(end), set()).add(event_key)
                    right.setdefault(int(start), set()).add(event_key)
            return left, right

        def keep_boundaries(shared_boundaries, left_map, right_map, other_events):
            kept = set()
            for boundary in shared_boundaries:
                all_events_at_b = left_map.get(boundary, set()) | right_map.get(boundary, set())
                if not all_events_at_b:
                    continue
                in_other = [ev in other_events for ev in all_events_at_b]

                if any(in_other) and not all(in_other):
                    kept.add(boundary)

                elif len(all_events_at_b) == 1 and not in_other[0]:
                    kept.add(boundary)
            return kept

        boundaries_a = {int(pos) for exon in event_exons_a for pos in exon}
        boundaries_b = {int(pos) for exon in event_exons_b for pos in exon}
        shared_boundaries = boundaries_a & boundaries_b
        if not shared_boundaries:
            return set()

        events_a = build_event_set(event_exons_a, junctions_a)
        events_b = build_event_set(event_exons_b, junctions_b)
        left_a, right_a = build_boundary_sides(event_exons_a, junctions_a)
        left_b, right_b = build_boundary_sides(event_exons_b, junctions_b)
        kept_boundaries = keep_boundaries(shared_boundaries, left_a, right_a, events_b)
        kept_boundaries.update(keep_boundaries(shared_boundaries, left_b, right_b, events_a))
        if not kept_boundaries:
            return set()

        boundary_event_map = {}
        for event_type, events in (
            ('exon', gene_event_cache.get('all_exons', [])),
            ('junction', gene_event_cache.get('all_junctions', [])),
        ):
            for start, end in events:
                event_key = (event_type, int(start), int(end))
                boundary_event_map.setdefault(int(start), set()).add(event_key)
                boundary_event_map.setdefault(int(end), set()).add(event_key)


        shared_events = events_a & events_b

        boundary_events = set()
        for boundary in kept_boundaries:
            for ev in boundary_event_map.get(int(boundary), set()):
                if ev not in shared_events:
                    boundary_events.add(ev)

        if not boundary_events:
            return set()

        queue = list(boundary_events)
        visited = set(boundary_events)
        while queue:
            _event_type, start, end = queue.pop(0)
            for pos in (int(start), int(end)):
                for neighbor in boundary_event_map.get(pos, set()):
                    if neighbor in shared_events or neighbor in visited:
                        continue
                    visited.add(neighbor)
                    boundary_events.add(neighbor)
                    queue.append(neighbor)
        return boundary_events

    @staticmethod
    def _extract_gtf_attribute(attribute_text, key):
        token = f'{key} "'
        start = attribute_text.find(token)
        while start != -1:
            if start == 0 or attribute_text[start - 1] in {' ', ';'}:
                start += len(token)
                end = attribute_text.find('"', start)
                if end != -1:
                    return attribute_text[start:end]
                return None
            start = attribute_text.find(token, start + 1)
        return None

    @staticmethod
    def _build_isoform_junction_map_from_transcript_exons(transcript_exons):
        iso_junction_map = {}
        if not transcript_exons:
            return iso_junction_map
        for iso_name, exons in transcript_exons.items():
            exons_sorted = sorted(
                {tuple(exon) for exon in exons},
                key=lambda x: (x[0], x[1])
            )
            junctions = set()
            for i in range(len(exons_sorted) - 1):
                donor_end = exons_sorted[i][1]
                acceptor_start = exons_sorted[i + 1][0]
                if acceptor_start > donor_end:
                    junctions.add((donor_end, acceptor_start))
            iso_junction_map[iso_name] = junctions
        return iso_junction_map

    def _load_gtf_junction_index(self):
        gtf_path = getattr(self, 'scotch_gtf_path', None)
        cache_key = gtf_path if gtf_path else ''
        if self._sample_gtf_junction_index_path == cache_key:
            return self._sample_gtf_junction_index

        if not gtf_path or not os.path.isfile(gtf_path):
            if gtf_path:
                self._log(
                    f'SCOTCH GTF not found for real splice-junction annotation: {gtf_path}; '
                    'falling back to pkl-derived junctions.'
                )
            else:
                self._log(
                    'SCOTCH GTF not configured for real splice-junction annotation; '
                    'falling back to pkl-derived junctions.'
                )
            self._sample_gtf_junction_index = None
            self._sample_gtf_junction_index_path = cache_key
            return None

        transcript_exons_by_gene = {}
        exon_rows = 0
        try:
            with open(gtf_path, 'r') as handle:
                for line in handle:
                    if not line or line[0] == '#':
                        continue
                    fields = line.rstrip('\n').split('\t', 8)
                    if len(fields) < 9 or fields[2] != 'exon':
                        continue
                    attributes = fields[8]
                    gene_id = self._extract_gtf_attribute(attributes, 'gene_id')
                    transcript_id = self._extract_gtf_attribute(attributes, 'transcript_id')
                    if not gene_id or not transcript_id:
                        continue
                    try:
                        start = int(fields[3]) - 1
                        end = int(fields[4])
                    except ValueError:
                        continue
                    if start < 0 or end <= start:
                        continue
                    gene_exons = transcript_exons_by_gene.setdefault(gene_id, {})
                    gene_exons.setdefault(transcript_id, []).append((start, end))
                    exon_rows += 1
        except Exception as exc:
            self._log(
                f'Unable to parse SCOTCH GTF for real splice-junction annotation ({gtf_path}): {exc}; '
                'falling back to pkl-derived junctions.'
            )
            self._sample_gtf_junction_index = None
            self._sample_gtf_junction_index_path = cache_key
            return None

        self._sample_gtf_junction_index = {
            gene_id: self._build_isoform_junction_map_from_transcript_exons(transcript_exons)
            for gene_id, transcript_exons in transcript_exons_by_gene.items()
        }
        self._sample_gtf_junction_index_path = cache_key
        self._log(
            f'Loaded SCOTCH GTF real-junction index from {gtf_path} '
            f'({exon_rows} exon rows, {len(self._sample_gtf_junction_index)} genes).'
        )
        return self._sample_gtf_junction_index

    def _get_gene_event_cache(self, gene_id):
        cache = getattr(self, '_gene_event_map_cache', None)
        if cache is None:
            self._gene_event_map_cache = {}
            cache = self._gene_event_map_cache

        if gene_id in cache:
            cached = cache[gene_id]


            if cached is not None or self.gsi is None or gene_id not in self.gsi:
                return cached

        if self.gsi is None or gene_id not in self.gsi:
            cache[gene_id] = None
            return None

        geneInfo, exon_positions, exon_isoform_dict = self.gsi[gene_id]
        iso_exon_map, fallback_junction_map = self._build_isoform_event_maps(
            exon_positions, exon_isoform_dict)

        iso_junction_map = fallback_junction_map
        gtf_junction_index = self._load_gtf_junction_index()
        if gtf_junction_index is not None:
            gtf_iso_junction_map = gtf_junction_index.get(gene_id)
            if gtf_iso_junction_map is not None:
                iso_junction_map = {
                    iso_name: set(gtf_iso_junction_map.get(iso_name, set()))
                    for iso_name in iso_exon_map
                }

        all_exons = sorted({e for s in iso_exon_map.values() for e in s
                            if e[1] - e[0] >= 5})
        all_junctions = sorted({j for s in iso_junction_map.values() for j in s
                                if j[1] != j[0]})

        exons_sorted = (
            sorted([tuple(e) for e in exon_positions], key=lambda x: x[0])
            if exon_positions else [])
        cache[gene_id] = {
            'geneInfo': geneInfo,
            'iso_exon_map': iso_exon_map,
            'iso_junction_map': iso_junction_map,
            'all_exons': all_exons,
            'all_junctions': all_junctions,
            'iso_exon_event_indices': _build_event_indices(all_exons, iso_exon_map),
            'iso_junction_event_indices': _build_event_indices(all_junctions, iso_junction_map),
            'exons_sorted': exons_sorted,
            'transcript_structures': None,
        }
        return cache[gene_id]

    @staticmethod
    def _build_isoform_event_maps(exon_positions, exon_isoform_dict):
        iso_exon_map = {}
        iso_junction_map = {}
        if not exon_isoform_dict or not exon_positions:
            return iso_exon_map, iso_junction_map

        for iso_name, exon_indices in exon_isoform_dict.items():
            exons = sorted(
                [tuple(exon_positions[i]) for i in exon_indices],
                key=lambda x: x[0]
            )
            iso_exon_map[iso_name] = set(exons)
            iso_junction_map[iso_name] = {
                (exons[i][1], exons[i + 1][0]) for i in range(len(exons) - 1)
            }
        return iso_exon_map, iso_junction_map


    def _link_snv_to_events(self, snv_df, hap_event_df,
                             max_exonic_dist=50):
        if hap_event_df.empty:
            return pd.DataFrame()

        rows = []
        for _, ev in hap_event_df.iterrows():
            gene_id = ev['geneID']
            gene_event_cache = self._get_gene_event_cache(gene_id)
            if gene_event_cache is None:
                continue
            exons_sorted = gene_event_cache['exons_sorted']

            ev_start, ev_end = int(ev['event_start']), int(ev['event_end'])
            gene_snvs = snv_df[
                (snv_df['geneID'] == gene_id) & (snv_df['chrom'] == ev['geneChr'])
            ]

            for _, snv in gene_snvs.iterrows():
                snv_pos = int(snv['pos'])
                exonic_dist = self._exonic_distance(snv_pos, ev_start, ev_end, exons_sorted)
                if exonic_dist <= max_exonic_dist:
                    rows.append({
                        'geneID': gene_id,
                        'geneName': ev['geneName'],
                        'chrom': ev['geneChr'],
                        'snv_pos': snv_pos,
                        'snv_ref': snv['ref'],
                        'snv_alt': snv.get('alt', np.nan),
                        'h_A': snv['h_A'],
                        'h_m': snv['h_m'],
                        'hat_Z_prob_revised': snv.get('hat_Z_prob_revised', np.nan),
                        'snv_hap': 'A' if snv['h_A'] > 0.5 else 'B',
                        'event_type': ev['event_type'],
                        'event_start': ev_start,
                        'event_end': ev_end,
                        'exonic_distance': exonic_dist,
                        'genomic_distance': min(abs(snv_pos - ev_start), abs(snv_pos - ev_end)),
                        'event_chi2': ev['chi2'],
                        'event_p_value': ev['p_value'],
                        'event_p_value_adj': ev['p_value_adj'],
                    })

        return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()

    @staticmethod
    def _exonic_distance(snv_pos, event_start, event_end, exons_sorted):
        if event_start <= snv_pos <= event_end:
            return 0

        def to_exonic(pos):
            cum = 0
            for es, ee in exons_sorted:
                if pos < es:
                    return cum
                if pos <= ee:
                    return cum + (pos - es)
                cum += (ee - es)
            return cum

        snv_ex = to_exonic(snv_pos)
        return min(abs(snv_ex - to_exonic(event_start)),
                   abs(snv_ex - to_exonic(event_end)))


    def _chi_sq_snv_event_raw(self, snv_event_df, cell_type='Bulk',
                              scotch_read_cell=None, cell_type_df=None,
                              scotch_isoform_df=None):
        if snv_event_df.empty:
            return snv_event_df

        relevant_genes = set(snv_event_df['geneID'].unique())
        allele_cache = {}
        allowed_reads_by_gene = None
        if cell_type != 'Bulk' and scotch_read_cell is not None and cell_type_df is not None:
            ct_cells = set(cell_type_df.loc[cell_type_df['CellType'] == cell_type, 'Cell'])
            ct_reads = scotch_read_cell.loc[
                scotch_read_cell['Cell'].isin(ct_cells), ['Read', 'geneID']
            ]
            if ct_reads.empty:
                return snv_event_df.assign(
                    raw_ref_present=None, raw_ref_absent=None,
                    raw_alt_present=None, raw_alt_absent=None,
                    raw_chi2=None, raw_p_value=None, raw_test_type=None,
                    raw_p_value_adj=None,
                )
            ct_reads = ct_reads[ct_reads['geneID'].isin(relevant_genes)].drop_duplicates()
            allowed_reads_by_gene = {
                gid: set(sub['Read'])
                for gid, sub in ct_reads.groupby('geneID')
            }

        if scotch_isoform_df is None:
            chunks = pd.read_csv(
                self.scotch_tsv_path, sep='\t', chunksize=100_000,
                usecols=lambda c: c in {'Read', 'geneID', 'Isoform', 'Keep'})
            frames = []
            for chunk in chunks:
                sub = chunk.loc[
                    (chunk['Keep'] == 1) & (chunk['geneID'].isin(relevant_genes)),
                    ['Read', 'geneID', 'Isoform']].copy()
                sub['Read'] = sub['Read'].map(_canonicalize_read_name)
                frames.append(sub)
            frames = [frame for frame in frames if not frame.empty]
            if not frames:
                return snv_event_df.assign(
                    raw_ref_present=None, raw_ref_absent=None,
                    raw_alt_present=None, raw_alt_absent=None,
                    raw_chi2=None, raw_p_value=None, raw_test_type=None,
                    raw_p_value_adj=None,
                )
            scotch_df = pd.concat(frames, ignore_index=True).drop_duplicates()
        else:
            scotch_df = scotch_isoform_df[scotch_isoform_df['geneID'].isin(relevant_genes)]
            if scotch_df.empty:
                return snv_event_df.assign(
                    raw_ref_present=None, raw_ref_absent=None,
                    raw_alt_present=None, raw_alt_absent=None,
                    raw_chi2=None, raw_p_value=None, raw_test_type=None,
                    raw_p_value_adj=None,
                )
        if allowed_reads_by_gene is not None:
            allowed_reads_df = pd.DataFrame(
                (
                    (gid, read_name)
                    for gid, reads in allowed_reads_by_gene.items()
                    for read_name in reads
                ),
                columns=['geneID', 'Read']
            )
            scotch_df = scotch_df.merge(
                allowed_reads_df, on=['geneID', 'Read'], how='inner'
            )
            if scotch_df.empty:
                return snv_event_df.assign(
                    raw_ref_present=None, raw_ref_absent=None,
                    raw_alt_present=None, raw_alt_absent=None,
                    raw_chi2=None, raw_p_value=None, raw_test_type=None,
                    raw_p_value_adj=None,
                )


        cached = getattr(self, '_scotch_by_gene_cache', {})
        if cached:
            scotch_by_gene = {
                gid: read_iso.groupby(level=0)['Isoform'].apply(list).to_dict()
                for gid, read_iso in cached.items()
                if gid in relevant_genes
            }
        else:
            scotch_by_gene = {
                gid: sub.groupby('Read')['Isoform'].apply(list).to_dict()
                for gid, sub in scotch_df.groupby('geneID')
            }

        bam_handles = {}

        result_rows = []
        group_keys = ['geneID', 'event_type', 'event_start', 'event_end', 'snv_pos', 'snv_ref', 'snv_alt']
        for keys, group in snv_event_df.groupby(group_keys, dropna=False):
            gene_id, ev_type, ev_start, ev_end, snv_pos, snv_ref, snv_alt = keys
            chrom = group.iloc[0]['chrom']

            extra = dict(raw_ref_present=None, raw_ref_absent=None,
                         raw_alt_present=None, raw_alt_absent=None,
                         raw_chi2=None, raw_p_value=None,
                         raw_test_type=None)

            alt_base = str(snv_alt).upper()
            if pd.isna(snv_alt) or alt_base not in {'A', 'C', 'G', 'T'}:
                result_rows.append(group.assign(**extra))
                continue


            intra_event = (ev_type == 'exon'
                           and int(ev_start) <= int(snv_pos) <= int(ev_end))

            gene_event_cache = self._get_gene_event_cache(gene_id)
            if gene_event_cache is not None:
                event_set_map = (
                    gene_event_cache['iso_exon_map']
                    if ev_type == 'exon'
                    else gene_event_cache['iso_junction_map']
                )
                event_tuple = (int(ev_start), int(ev_end))
                iso_lookup = scotch_by_gene.get(gene_id, {})


                allowed_reads = (None if allowed_reads_by_gene is None
                                 else allowed_reads_by_gene.get(gene_id, set()))
                cache_key = (gene_id, chrom, int(snv_pos))
                if cache_key in allele_cache:
                    cached_alleles = allele_cache[cache_key]
                else:
                    site_reads_by_gene = self._load_site_reads_for_gene(gene_id)
                    cached_alleles = self._read_alleles_from_site_reads(
                        site_reads_by_gene, chrom=chrom, pos=int(snv_pos))
                    if cached_alleles is None:
                        bam_path = self._resolve_bam_path(chrom)
                        bam = None
                        if bam_path is not None:
                            bam = bam_handles.get(bam_path)
                            if bam_path not in bam_handles:
                                try:
                                    bam = pysam.Samfile(bam_path, 'rb')
                                except Exception as exc:
                                    self._log(
                                        f'Unable to open BAM for raw SNV-event validation '
                                        f'({bam_path}): {exc}'
                                    )
                                    bam = None
                                bam_handles[bam_path] = bam
                        if bam is not None:
                            cached_alleles = self._read_alleles_from_bam(
                                bam, chrom=chrom, pos=int(snv_pos))
                    if cached_alleles is None:
                        cached_alleles = {}
                    allele_cache[cache_key] = cached_alleles
                if allowed_reads is None:
                    read_alleles = cached_alleles
                else:
                    read_alleles = {
                        read_name: allele
                        for read_name, allele in cached_alleles.items()
                        if read_name in allowed_reads
                    }

                ref_base = str(snv_ref).upper()
                rp = ra = ap = aa = 0
                for read_name, allele in read_alleles.items():
                    if intra_event:


                        if allele == alt_base:
                            ap += 1
                        elif allele == ref_base:
                            rp += 1
                    else:
                        isos = iso_lookup.get(read_name)
                        if not isos:
                            continue


                        memberships = {event_tuple in event_set_map.get(iso, set())
                                       for iso in isos}
                        if len(memberships) != 1:
                            continue
                        present = memberships.pop()
                        if allele == alt_base:
                            if present:
                                ap += 1
                            else:
                                aa += 1
                        elif allele == ref_base:
                            if present:
                                rp += 1
                            else:
                                ra += 1

                if (not intra_event and read_alleles
                        and rp == 0 and ra == 0 and ap == 0 and aa == 0):
                    site_names = set(read_alleles.keys())
                    iso_names = set(iso_lookup.keys())
                    exact_overlap = len(site_names & iso_names)
                    ex_site = list(site_names)[:3]
                    ex_iso = list(iso_names)[:3]
                    self._log(
                        f'  Task 4c WARNING: {gene_id} SNV {chrom}:{snv_pos} has '
                        f'{len(read_alleles)} site reads but 0 joined to SCOTCH isoforms. '
                        f'Overlap: {exact_overlap}/{len(site_names)} site reads vs '
                        f'{len(iso_names)} SCOTCH reads. '
                        f'Example site reads: {ex_site}, '
                        f'Example SCOTCH reads: {ex_iso}'
                    )

                chi2_val = p_val = None
                if intra_event:
                    test_type = 'binomial_intra_event'
                    n = rp + ap
                    if n >= 10:
                        try:
                            p_val = float(binomtest(ap, n, p=0.5).pvalue)
                        except Exception:
                            p_val = None
                else:
                    test_type = 'chi2_cross_event'
                    table = np.array([[rp, ra], [ap, aa]])
                    if table.sum() >= 10 and not ((table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any()):
                        try:
                            chi2_val, p_val, _, _ = chi2_contingency(table, correction=False)
                        except Exception:
                            pass

                extra = dict(
                    raw_ref_present=rp, raw_ref_absent=ra,
                    raw_alt_present=ap, raw_alt_absent=aa,
                    raw_chi2=round(chi2_val, 4) if chi2_val is not None else None,
                    raw_p_value=p_val,
                    raw_test_type=test_type,
                )

            result_rows.append(group.assign(**extra))

        for bam in bam_handles.values():
            if bam is not None:
                try:
                    bam.close()
                except (OSError, ValueError):
                    pass
        combined = pd.concat(result_rows, ignore_index=True) if result_rows else snv_event_df


        if not combined.empty and 'raw_p_value' in combined.columns:
            combined['raw_p_value_adj'] = np.nan
            valid_mask = pd.to_numeric(combined['raw_p_value'], errors='coerce').notna()
            if valid_mask.any():
                for _, grp in combined.loc[valid_mask].groupby('geneID', sort=False):
                    combined.loc[grp.index, 'raw_p_value_adj'] = multipletests(
                        pd.to_numeric(grp['raw_p_value'], errors='coerce').to_numpy(dtype=float),
                        method='fdr_bh',
                    )[1]
        return combined

    def _load_site_reads_for_gene(self, gene_id):
        cache = getattr(self, '_sample_site_reads_cache', None)
        if cache is not None and gene_id in cache:
            return cache[gene_id]
        if not getattr(self, 'variant_dir', None):
            if cache is not None:
                cache[gene_id] = None
            return None
        path = os.path.join(self.variant_dir, f'{gene_id}_site_reads.pkl')
        site_reads = load_pickle(path)
        if cache is not None:
            cache[gene_id] = site_reads
        return site_reads

    @staticmethod
    def _read_alleles_from_site_reads(site_reads_by_gene, chrom, pos, allowed_reads=None):
        if site_reads_by_gene is None:
            return None
        tuples = site_reads_by_gene.get((str(chrom), int(pos)))
        if tuples is None:
            tuples = site_reads_by_gene.get((chrom, int(pos)))
        if tuples is None:
            return {}
        read_alleles = {}
        for read_name, _qpos, base, _bq, _mapq, _read_len, _is_reverse in tuples:
            read_name = _canonicalize_read_name(read_name)
            if allowed_reads is not None and read_name not in allowed_reads:
                continue
            if not isinstance(base, str):
                continue
            allele = base.upper()
            if allele in {'A', 'C', 'G', 'T'}:
                read_alleles[read_name] = allele
        return read_alleles

    @staticmethod
    def _read_alleles_from_bam(bam, chrom, pos, allowed_reads=None):
        read_alleles = {}
        try:
            for col in bam.pileup(chrom, int(pos), int(pos) + 1,
                                  stepper='samtools',
                                  min_base_quality=0, min_mapping_quality=0):
                if col.reference_pos != int(pos):
                    continue
                for pr in col.pileups:
                    if pr.is_del or pr.is_refskip:
                        continue
                    qn = _canonicalize_read_name(pr.alignment.query_name)
                    if allowed_reads is not None and qn not in allowed_reads:
                        continue
                    base = pr.alignment.query_sequence[pr.query_position].upper()
                    if base in {'A', 'C', 'G', 'T'}:
                        read_alleles[qn] = base
                break
        except (ValueError, OSError):
            pass
        return read_alleles

    def _has_variant_site_read_pkls(self):
        if not getattr(self, 'variant_dir', None) or not os.path.isdir(self.variant_dir):
            return False
        return any(name.endswith('_site_reads.pkl') for name in os.listdir(self.variant_dir))

    @staticmethod
    def _normalize_optional_list(values, n_samples):
        if values is None:
            return None
        if isinstance(values, str):
            values = [values]
        else:
            values = list(values)
        if len(values) == 1:
            return values * n_samples
        if len(values) != n_samples:
            raise ValueError('Expected one value or one value per sample.')
        return values

    @staticmethod
    def _normalize_sample_names(sample_names, scotch_target):
        n_samples = len(scotch_target)
        if sample_names is None:
            return [os.path.basename(st) for st in scotch_target]
        if isinstance(sample_names, str):
            sample_names = [sample_names]
        else:
            sample_names = list(sample_names)
        if len(sample_names) == 1 and n_samples == 1:
            return sample_names
        if len(sample_names) != n_samples:
            raise ValueError('sample_names must contain one entry per sample.')
        return sample_names

    def _resolve_reference_pickle_path(self):
        if self.ref_pickle_path is not None:
            return self.ref_pickle_path


        st = self.scotch_target[0]
        ref_dir = os.path.join(st, 'reference')
        candidates = [
            'geneStructureInformationupdated.pkl',
            'metageneStructureInformationwnovel.pkl',
            'geneStructureInformation.pkl',
            'metageneStructureInformation.pkl',
        ]
        for name in candidates:
            path = os.path.join(ref_dir, name)
            if os.path.isfile(path):
                self._log(f'Resolved reference pickle: {name}')
                return path
        return None

    def _resolve_scotch_gtf_path(self, scotch_target, sample_name=None):
        ref_dir = os.path.join(scotch_target, 'reference')
        exact = os.path.join(ref_dir, 'SCOTCH_updated_annotation_filtered.gtf')
        if os.path.isfile(exact):
            return exact
        if not os.path.isdir(ref_dir):
            return None
        matches = sorted(
            os.path.join(ref_dir, name)
            for name in os.listdir(ref_dir)
            if name.startswith('SCOTCH_updated_annotation_filtered') and name.endswith('.gtf')
        )
        if not matches:
            return None
        if len(matches) > 1:
            label = f' for sample {sample_name}' if sample_name else ''
            self._log(
                f'Multiple SCOTCH GTF files found{label} in {ref_dir}; '
                f'using {os.path.basename(matches[0])}.'
            )
        return matches[0]

    def _load_gene_structure_information(self, gsi_path):
        if not gsi_path:
            self._log(
                'geneStructureInformation pickle not found for current sample; '
                'Task 4 event annotation will be skipped.'
            )
            return None
        gsi = load_pickle(gsi_path)
        if gsi is None:
            self._log(f'Unable to load geneStructureInformation pickle: {gsi_path}')
            return gsi


        if 'meta' in os.path.basename(gsi_path).lower():
            flat = {}
            for genes_info_list in gsi.values():
                for gene_info, exon_info, isoform_info in genes_info_list:
                    flat[gene_info['geneID']] = (gene_info, exon_info, isoform_info)
            self._log(f'Flattened meta pickle to {len(flat)} genes')
            gsi = flat
        return gsi

    def _build_sample_configs(self):
        pfx = self.prefix or ''
        configs = []


        variant_dir = os.path.join(self.output_folder, 'variant_align1', 'variants_by_gene')
        for idx, (scotch_target, sample_name) in enumerate(zip(self.scotch_target, self.sample_names)):
            if len(self.scotch_target) == 1:
                snv_hap_dir = (os.path.join(self.output_folder, f'snv_hap_{pfx}') if pfx
                               else os.path.join(self.output_folder, 'snv_hap'))
                summary_dir = (os.path.join(self.output_folder, f'summary_statistics_{pfx}') if pfx
                               else os.path.join(self.output_folder, 'summary_statistics'))
                count_dir = (os.path.join(self.output_folder, f'count_matrix_hap_{pfx}') if pfx
                             else os.path.join(self.output_folder, 'count_matrix_hap'))
                downstream_output = (os.path.join(self.output_folder, f'downstream_{pfx}') if pfx
                                     else os.path.join(self.output_folder, 'downstream'))
            else:
                snv_hap_dir = (os.path.join(self.output_folder, sample_name, f'snv_hap_{pfx}') if pfx
                               else os.path.join(self.output_folder, sample_name, 'snv_hap'))
                summary_dir = (os.path.join(self.output_folder, sample_name, f'summary_statistics_{pfx}') if pfx
                               else os.path.join(self.output_folder, sample_name, 'summary_statistics'))
                count_dir = (os.path.join(self.output_folder, sample_name, f'count_matrix_hap_{pfx}') if pfx
                             else os.path.join(self.output_folder, sample_name, 'count_matrix_hap'))
                downstream_output = (os.path.join(self.output_folder, sample_name, f'downstream_{pfx}') if pfx
                                     else os.path.join(self.output_folder, sample_name, 'downstream'))

            scotch_tsv_path = resolve_scotch_auxiliary_tsv(scotch_target, self.sample_name_parse or None)

            configs.append({
                'sample_name': sample_name,
                'snv_hap_map_path': os.path.join(snv_hap_dir, 'snv_hap_map.csv'),
                'read_hap_map_path': os.path.join(snv_hap_dir, 'read_hap_map.csv'),
                'summary_statistics_path': os.path.join(summary_dir, 'summary_statistics.csv'),
                'count_dir': count_dir,
                'isoform_agg_balance_path': os.path.join(count_dir, 'all_genes', 'isoform_agg_balance.csv'),
                'isoform_agg_path': os.path.join(count_dir, 'all_genes', 'isoform_agg.csv'),


                'isoform_agg_extrap_path': os.path.join(count_dir, 'all_genes', 'isoform_agg_extrap.csv'),


                'isoform_agg_unbalance_path': os.path.join(count_dir, 'all_genes', 'isoform_agg_unbalance.csv'),
                'isoform_agg_pmax_path': os.path.join(count_dir, 'all_genes', 'isoform_agg_pmax.csv'),
                'variant_dir': variant_dir,
                'downstream_output': downstream_output,
                'scotch_tsv_path': scotch_tsv_path,
                'gsi_path': self._resolve_reference_pickle_path(),
                'scotch_gtf_path': self._resolve_scotch_gtf_path(
                    scotch_target, sample_name=sample_name),
                'bam_path': None if self.bam_paths is None else self.bam_paths[idx],
            })
        return configs

    def _set_sample_context(self, sample_idx):
        cfg = self.sample_configs[sample_idx]
        prev_gsi_path = self.gsi_path
        prev_gtf_path = self.scotch_gtf_path

        self._log_sample_name = cfg.get('sample_name')
        self.snv_hap_map_path = cfg['snv_hap_map_path']
        self.read_hap_map_path = cfg['read_hap_map_path']
        self.summary_statistics_path = cfg['summary_statistics_path']
        self.count_dir = cfg['count_dir']
        self.isoform_agg_balance_path = cfg['isoform_agg_balance_path']
        self.isoform_agg_path = cfg.get('isoform_agg_path')
        self.isoform_agg_extrap_path = cfg.get('isoform_agg_extrap_path')
        self.isoform_agg_unbalance_path = cfg.get('isoform_agg_unbalance_path')
        self.isoform_agg_pmax_path = cfg.get('isoform_agg_pmax_path')
        self.variant_dir = cfg['variant_dir']
        self.downstream_output = cfg['downstream_output']
        self.scotch_tsv_path = cfg['scotch_tsv_path']
        self.gsi_path = cfg['gsi_path']
        self.scotch_gtf_path = cfg['scotch_gtf_path']
        self.bam_path = cfg['bam_path']
        self.cell_type_df = (None if self.cell_type_df_list is None
                             else self.cell_type_df_list[sample_idx])

        if self.gsi_path != prev_gsi_path or self.gsi is None:
            self.gsi = self._load_gene_structure_information(self.gsi_path)
            self.meta = self.gsi
        if self.scotch_gtf_path != prev_gtf_path:
            self._sample_gtf_junction_index = None
            self._sample_gtf_junction_index_path = None

    def _log(self, msg):
        sample_name = getattr(self, '_log_sample_name', None)
        if sample_name:
            msg = f'[{sample_name}] {msg}'
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
