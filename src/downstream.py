import os
import pickle
import re
import pysam
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, binomtest
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from src.compat import _collapse_legacy_merge_suffixes
from src.utils import canonicalize_read_name as _canonicalize_read_name


def load_pickle(file):
    if os.path.exists(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
    return None


# Defaults: junction tolerance / flank wider than exon overlap because long-read
# splice-site placement is noisy (PacBio CCS ±5–15 bp, ONT direct RNA worse).
_OBS_EXON_MIN_OVERLAP_BP = 20
_OBS_EXON_MIN_OVERLAP_FRAC = 0.30
_OBS_JUNCTION_TOLERANCE_BP = 20
_OBS_JUNCTION_FLANK_MIN_BP = 20


def _judge_event_obs(event_type, event, blocks, intron_spans):
    """Judge whether a read's alignment includes / skips / fails to observe an event.

    Returns one of 'include', 'skip', 'unobserved'.
    Thresholds match _OBS_* constants (calibrated to tolerate long-read noise).
    """
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
        # No (or too little) aligned overlap — check whether the read spans the
        # event with an N gap that swallows it (true cassette skip).
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
    """Per-gene haplotype–event association (module-level for joblib pickling).

    Uses isoform-level aggregation: sum hat_I/hat_I_B by isoform first,
    then scatter into event vectors via precomputed index arrays.
    This is O(n_isoforms × events_per_isoform) instead of O(n_reads × n_events).
    """
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

    # Aggregate weights by isoform (collapses reads → isoforms)
    iso_weights = g.groupby('Isoform', sort=False)[['hat_I', 'hat_I_B']].sum()
    if iso_weights.empty:
        return []

    iso_names = iso_weights.index.to_numpy()
    hat_I_by_iso = iso_weights['hat_I'].to_numpy(dtype=float)
    hat_I_B_by_iso = iso_weights['hat_I_B'].to_numpy(dtype=float)
    total_A = float(hat_I_by_iso.sum())
    total_B = float(hat_I_B_by_iso.sum())

    # Optional: per-read CIGAR-observed event status for the same joined read
    # pool. Loaded from the per-gene read_blocks.pkl written by step1.5
    # (process_read_blocks_round1_5 + merge_read_blocks_round1_5); provides
    # "obs_*" columns parallel to the isoform-inferred ones. Returns None when
    # the pkl is absent (step1.5 + step1_5_merge never ran).
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
    """Precompute {isoform: np.array of event indices} for fast scatter."""
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
    """Join one gene's read-haplotype rows to SCOTCH isoforms, then test events."""
    merged = read_hap_gene.join(scotch_gene_isoform, on='Read', how='inner')
    if merged.empty:
        return []
    return _process_gene_events(
        gene_id, merged, gsi, min_reads, gene_event_cache=gene_event_cache,
        variant_dir=variant_dir,
    )


def _gather_read_obs_data(g, variant_dir, gene_id):
    """Per-read alignment blocks + intron spans for the joined read pool.

    Reads from `{variant_dir}/{gene_id}_read_blocks.pkl` (written by step1.5
    merge: utils.py:merge_read_blocks_round1_5, fed by per-sample intermediates
    from process_read_blocks_round1_5). Disk read instead of BAM fetch avoids
    the petagene + joblib concurrent-BAM corruption that hit AD hafobs job
    9783141 (2026-05-23). Returns None when the pkl is absent (step1_5 +
    step1_5_merge never ran) — caller emits `obs_test_type='no_bam'`.

    Returns dict {canonical_read_name: (hat_I, hat_I_B, blocks, intron_spans)}.
    Reads present in `g` but missing from the pkl have blocks=None and count
    as 'unobserved' for every event.
    """
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
            # Malformed entry (schema corrupt) — treat as unobserved rather
            # than kill the worker.
            out[rn] = (hat_I, hat_I_B, None, None)
            continue
        out[rn] = (hat_I, hat_I_B, blocks, intron_spans)
    return out


def _aggregate_obs_per_event(read_obs_data, event_type, events):
    """Per-event CIGAR-observed include/skip/unobserved weighted by hat_I / hat_I_B."""
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
    """Build the 9 obs_* column values for a single event row."""
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
    # Guard on non-degenerate margins (no all-zero row/column — a zero margin
    # means there is genuinely nothing to test), NOT on individual zero cells.
    # A single zero cell with non-zero margins (e.g. complete include/skip on
    # one hap) is the STRONGEST allele-specific signal and is perfectly
    # testable: an observed zero cell does not invalidate the chi-square (what
    # matters is the expected cell counts, which stay well-behaved when the
    # margins are non-zero). The old `table.min() >= 1` guard silently dropped
    # these as insufficient_data. Keep chi-square on the hat_I-weighted table
    # throughout for continuity with the pipeline's other 2×2 tests
    # (_chi_sq_snv_event_raw uses the same margin-only guard).
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
    """
    Post-haplotyping downstream analyses:
      Task 1 – Refine SNV calls with entropy-weighted confidence scores.
      Task 2 – SNV → gene expression effect size (es_ase).
      Task 3 – SNV → aStu effect size via dominant isoform fraction change (es_astu).
      Task 4 – Haplotype–event linkage (exon / splice-junction associations),
               followed by SNV–event proximity linking and raw-read chi-squared validation.
    """

    def __init__(self, output_folder, scotch_target, bam_path=None,
                 ref_pickle_path=None, sample_name_parse=None,
                 prefix='LongAllele', sample_names=None,
                 cell_type_df_path=None, n_workers=1, logger=None,
                 astu_sig_only=False, astu_sig_from_bulk=False,
                 astu_sig_threshold=0.05, n_jobs=1, job_index=0,
                 job_array_by_sample=False, gene_subset=None):
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

        # Cell type mapping (Cell → CellType) — must be set before _set_sample_context
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

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run_all(self, event_min_reads=10,
                snv_event_distance=50,
                event_mode='all_events',
                fdr_events_value=0.05):
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

            # --- Cache sample-wide files once ---
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

            # Task 1: entropy-weighted SNV confidence (gene-level, same for all cell types)
            self._log('Task 1: Refining SNV calls...')
            snv_df = self._refine_snv_calls()
            if self.gene_subset is not None and 'geneID' in snv_df.columns:
                snv_df = snv_df[snv_df['geneID'].isin(self.gene_subset)].copy()

            # Discover cell types present in summary statistics
            cell_types = self._discover_cell_types(summary_df=summary_df)   # ['Bulk'] or ['Bulk', 'TypeA', ...]

            # Single SCOTCH TSV load: derive both Read→Cell and Read→Isoform views
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

                # Tasks 2 & 3: ASE and aStu effect sizes
                self._log('  Tasks 2/3: Computing ASE / aStu effect sizes...')
                effect_df = self._compute_effect_sizes(
                    snv_df, cell_type=ct,
                    summary_df=summary_df, ct_summary=ct_summary)
                gene_snv_df = self._assemble_gene_snv_output(
                    effect_df,
                    sample_name=sample_cfg['sample_name'],
                    cell_type=ct
                )
                gene_snv_frames.append(gene_snv_df)

                # Task 4a: haplotype–event associations
                self._log('  Task 4a: Haplotype–event associations...')
                hap_event_df = self._haplotype_event_associations(
                    min_reads=event_min_reads, cell_type=ct,
                    scotch_read_cell=scotch_read_cell,
                    read_hap_df=read_hap_df,
                    scotch_isoform_df=scotch_isoform_df,
                    summary_df=summary_df, ct_summary=ct_summary,
                    bulk_sig_gene_ids=bulk_sig_gene_ids,
                    event_mode=event_mode,
                    fdr_events_value=fdr_events_value)
                if hap_event_df is None or hap_event_df.empty:
                    self._log('  No haplotype–event associations found; skipping Tasks 4b/4c.')
                    continue

                # Task 4b: link SNVs to nearby events
                self._log('  Task 4b: Linking SNVs to nearby events...')
                snv_event_df = self._link_snv_to_events(
                    snv_df, hap_event_df,
                    max_exonic_dist=snv_event_distance)
                if snv_event_df.empty:
                    self._log('  No SNV–event pairs within distance threshold.')

                # Task 4c: chi-squared validation with raw reads
                chi_df = None
                if has_raw_alleles:
                    self._log('  Task 4c: Raw-read chi-squared test for SNV–event pairs...')
                    chi_df = self._chi_sq_snv_event_raw(
                        snv_event_df,
                        cell_type=ct,
                        scotch_read_cell=scotch_read_cell,
                        cell_type_df=self.cell_type_df,
                        scotch_isoform_df=scotch_isoform_df
                    )

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

    # ------------------------------------------------------------------
    # Helpers: cell-type discovery and read→cell mapping
    # ------------------------------------------------------------------

    def _discover_cell_types(self, summary_df=None):
        """Return ['Bulk'] + sorted list of other cell types found in summary_statistics.csv."""
        if summary_df is None:
            if not os.path.exists(self.summary_statistics_path):
                return ['Bulk']
            summary_df = pd.read_csv(self.summary_statistics_path, usecols=['CellType'])
        cts = summary_df['CellType'].dropna().unique().tolist()
        bulk = [c for c in cts if c == 'Bulk']
        others = sorted(c for c in cts if c != 'Bulk')
        return bulk + others

    def _load_scotch_tables(self):
        """Load kept SCOTCH rows once and derive both Read→Cell and Read→Isoform views."""
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
        """Load a Read→Cell mapping from the SCOTCH TSV (needed for CT filtering in Task 4)."""
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

    # ------------------------------------------------------------------
    # Task 1: Refine SNV calls
    # ------------------------------------------------------------------

    def _refine_snv_calls(self):
        """
        Add entropy, hat_Z_prob_revised, hat_Z_binary_revised to snv_hap_map.csv.
        Formula (from R post-processing):
          entropy             = ifelse(h_A==0|h_A==1, 0, -(h_A*log2(h_A)+(1-h_A)*log2(1-h_A)))
          hat_Z_prob_revised  = round(h_m*(1-entropy), 2)
          hat_Z_binary_revised = 1 if hat_Z_prob_revised >= 0.5 else 0
        """
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
        # Keep only confidently phased SNVs; retain the probability score downstream.
        snv_df = snv_df[snv_df['hat_Z_binary_revised'] == 1].drop(
            columns=['hat_Z_binary_revised']).reset_index(drop=True)
        snv_df = self._ensure_alt_column(snv_df)
        return snv_df

    def _ensure_alt_column(self, snv_df):
        """
        Ensure an SNV alt allele column exists for downstream linking / validation.
        Prefer existing upstream columns; otherwise infer the dominant non-ref base
        from site_reads.pkl first, then fall back to chromosome-specific BAM pileup.
        """
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

    # ------------------------------------------------------------------
    # Tasks 2 & 3: Effect sizes
    # ------------------------------------------------------------------

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
        """
        Compute gene-level ASE / ASTU context for each confident SNV row.
        Returns one row per SNV with gene-level summary fields repeated.
        """
        if ct_summary is None:
            if summary_df is None:
                summary_df = pd.read_csv(self.summary_statistics_path)
            ct_summary = summary_df[summary_df['CellType'] == cell_type]

        ct_summary = ct_summary.copy()
        needed_cols = [
            'geneID', 'n_reads', 'n_reads_phasable', 'n_snvs',
            'alpha_hat', 'alpha_hat_low', 'alpha_hat_high',
            'major_hap', 'p_value', 'p_value_gene_adj',
            'p_value_isoform', 'p_value_isoform_high', 'p_value_isoform_low',
            'p_value_isoform_adj', 'p_value_isoform_adj_high', 'p_value_isoform_adj_low'
        ]
        for col in needed_cols:
            if col not in ct_summary.columns:
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
        ct_rows['gene_alpha_hat_major'] = 1 - ct_rows['gene_alpha_hat']
        ct_rows['gene_alpha_hat_major_low'] = 1 - ct_rows['gene_alpha_hat_high']
        ct_rows['gene_alpha_hat_major_high'] = 1 - ct_rows['gene_alpha_hat_low']
        ct_rows['gene_major_hap'] = ct_rows['major_hap']
        ct_rows['gene_minor_hap'] = np.where(
            ct_rows['gene_major_hap'] == 'A', 'B',
            np.where(ct_rows['gene_major_hap'] == 'B', 'A', np.nan)
        )
        ct_rows['gene_p_value'] = pd.to_numeric(ct_rows['p_value'], errors='coerce')
        ct_rows['gene_p_value_adj'] = pd.to_numeric(ct_rows['p_value_gene_adj'], errors='coerce')
        ct_rows['isoform_p_value'] = pd.to_numeric(ct_rows['p_value_isoform'], errors='coerce')
        ct_rows['isoform_p_value_high'] = pd.to_numeric(ct_rows['p_value_isoform_high'], errors='coerce')
        ct_rows['isoform_p_value_low'] = pd.to_numeric(ct_rows['p_value_isoform_low'], errors='coerce')
        ct_rows['isoform_p_value_adj'] = pd.to_numeric(ct_rows['p_value_isoform_adj'], errors='coerce')
        ct_rows['isoform_p_value_adj_high'] = pd.to_numeric(ct_rows['p_value_isoform_adj_high'], errors='coerce')
        ct_rows['isoform_p_value_adj_low'] = pd.to_numeric(ct_rows['p_value_isoform_adj_low'], errors='coerce')
        ct_rows['ASE_call'] = np.select(
            [
                (ct_rows['gene_alpha_hat_high'] < 0.5) & (ct_rows['gene_p_value_adj'] <= 0.05),
                ct_rows['gene_p_value_adj'] > 0.05,
            ],
            [1, -1],
            default=0
        ).astype(int)
        ct_rows['ASTU_call'] = np.select(
            [
                ct_rows['isoform_p_value_adj_high'] <= 0.05,
                ct_rows['isoform_p_value_adj_low'] > 0.05,
            ],
            [1, -1],
            default=0
        ).astype(int)

        bulk_shrinkage_k = getattr(self, '_bulk_shrinkage_k', None)
        if bulk_shrinkage_k is None:
            shrinkage_k = self._derive_shrinkage_k(ct_rows['n_reads'])
        else:
            shrinkage_k = bulk_shrinkage_k

        alpha_mean = (ct_rows['gene_alpha_hat_low'] + ct_rows['gene_alpha_hat_high']) / 2.0
        alpha_mean = alpha_mean.where(alpha_mean.notna(), ct_rows['gene_alpha_hat'])
        ct_rows['shrinkage_k'] = shrinkage_k
        ct_rows['n_reads_minor_hap'] = ct_rows['n_reads'] * alpha_mean + shrinkage_k
        ct_rows['n_reads_major_hap'] = (
            ct_rows['n_reads'] - ct_rows['n_reads'] * alpha_mean + shrinkage_k
        )
        ct_rows['es_ase'] = np.log2(
            ct_rows['n_reads_major_hap'] / ct_rows['n_reads_minor_hap'].clip(lower=shrinkage_k)
        ).round(4)

        astu = self._compute_astu_effect(
            ct_rows[['geneID', 'gene_major_hap']].copy(),
            cell_type=cell_type
        )
        if astu is not None:
            ct_rows = ct_rows.merge(astu, on='geneID', how='left')
        else:
            ct_rows['dominant_isoform_overall'] = np.nan
            ct_rows['top_isoform_hap_major'] = np.nan
            ct_rows['top_isoform_hap_minor'] = np.nan
            ct_rows['top_isoform_hap_major_frac'] = np.nan
            ct_rows['top_isoform_hap_minor_frac'] = np.nan
            ct_rows['dominant_isoform_pref_hap'] = np.nan
            ct_rows['es_astu'] = np.nan
            ct_rows['astu_source'] = np.nan

        gene_effect_cols = [
            'geneID',
            'n_reads', 'n_reads_phasable', 'gene_n_snvs', 'gene_n_snvs_called',
            'gene_alpha_hat', 'gene_alpha_hat_low', 'gene_alpha_hat_high',
            'gene_alpha_hat_major', 'gene_alpha_hat_major_low', 'gene_alpha_hat_major_high',
            'gene_major_hap', 'gene_minor_hap',
            'gene_p_value', 'gene_p_value_adj',
            'ASE_call', 'ASTU_call',
            'dominant_isoform_overall',
            'top_isoform_hap_major', 'top_isoform_hap_minor',
            'top_isoform_hap_major_frac', 'top_isoform_hap_minor_frac',
            'isoform_p_value', 'isoform_p_value_high', 'isoform_p_value_low',
            'isoform_p_value_adj', 'isoform_p_value_adj_high', 'isoform_p_value_adj_low',
            'shrinkage_k', 'es_ase',
            'es_astu', 'astu_source',
            'dominant_isoform_pref_hap'
        ]
        result = snv_df.merge(ct_rows[gene_effect_cols], on='geneID', how='left')

        # h_A > 0.5 → alt on hap A; h_A == 0.5 (ambiguous/unphased) defaults to B
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

        dom_mask = result['dominant_isoform_pref_hap'].isin(['A', 'B'])
        result.loc[dom_mask, 'snv_astu_direction'] = np.where(
            result.loc[dom_mask, 'snv_hap'] == result.loc[dom_mask, 'dominant_isoform_pref_hap'],
            '+',
            '-'
        )
        astu_mask = dom_mask & result['es_astu'].notna()
        result.loc[astu_mask, 'snv_es_astu_signed'] = np.where(
            result.loc[astu_mask, 'snv_hap'] == result.loc[astu_mask, 'dominant_isoform_pref_hap'],
            result.loc[astu_mask, 'es_astu'],
            -result.loc[astu_mask, 'es_astu']
        )

        return result

    def _compute_astu_effect(self, ct_rows_df, cell_type='Bulk'):
        """
        Compute dominant-isoform ASTU context and hap-specific top isoforms.
        """
        if cell_type != 'Bulk':
            safe_ct = cell_type.replace('/', '_').replace(' ', '_')
            ct_path = os.path.join(
                self.count_dir, 'all_genes',
                f'ct_{safe_ct}_isoform_agg_balance.csv'
            )
            if os.path.exists(ct_path):
                iso_path = ct_path
                astu_source = 'ct_specific'
            else:
                iso_path = self.isoform_agg_balance_path
                astu_source = 'bulk_fallback'
        else:
            iso_path = self.isoform_agg_balance_path
            astu_source = 'bulk'

        if not os.path.exists(iso_path):
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

            dominant_isoform = str(g['_total'].idxmax())
            # Shrinkage denominator uses sqrt(n_isoforms) (sub-linear compression)
            # instead of linear n_isoforms. Linear scaling over-penalises multi-isoform
            # genes under low per-cell-type coverage, compressing es_astu asymmetrically
            # between per-cell-type (heavy shrinkage) and bulk (minimal shrinkage) rows —
            # flipping the sign of the ASTU x constraint slope from its true bulk value.
            # sqrt retains Laplace/Dirichlet categorical smoothing intent while keeping
            # per-ct and bulk slopes aligned. Empirically verified 2026-04-17 (see
            # research/brain_constraint_allelic/shrinkage_fix/astu_shrinkage_niso_compress.R).
            n_iso_scale = np.sqrt(n_isoforms)
            dom_frac_A = (
                float(g.loc[dominant_isoform, 'hapA']) + shrinkage_k
            ) / (total_A + shrinkage_k * n_iso_scale)
            dom_frac_B = (
                float(g.loc[dominant_isoform, 'hapB']) + shrinkage_k
            ) / (total_B + shrinkage_k * n_iso_scale)
            dominant_isoform_pref_hap = 'A' if dom_frac_A >= dom_frac_B else 'B'

            top_isoform_A = str(g['hapA'].idxmax())
            top_isoform_B = str(g['hapB'].idxmax())
            top_isoform_A_frac = float(g.loc[top_isoform_A, 'hapA']) / total_A
            top_isoform_B_frac = float(g.loc[top_isoform_B, 'hapB']) / total_B

            if major_hap == 'A':
                top_iso_major, top_iso_minor = top_isoform_A, top_isoform_B
                top_frac_major, top_frac_minor = top_isoform_A_frac, top_isoform_B_frac
            else:
                top_iso_major, top_iso_minor = top_isoform_B, top_isoform_A
                top_frac_major, top_frac_minor = top_isoform_B_frac, top_isoform_A_frac

            dom_frac_higher = max(dom_frac_A, dom_frac_B)
            dom_frac_lower = min(dom_frac_A, dom_frac_B)
            es_astu = np.log2(dom_frac_higher / dom_frac_lower)

            rows.append({
                'geneID': gene_id,
                'dominant_isoform_overall': dominant_isoform,
                'top_isoform_hap_major': top_iso_major,
                'top_isoform_hap_minor': top_iso_minor,
                'top_isoform_hap_major_frac': round(float(top_frac_major), 4),
                'top_isoform_hap_minor_frac': round(float(top_frac_minor), 4),
                'dominant_isoform_pref_hap': dominant_isoform_pref_hap,
                'es_astu': round(float(es_astu), 4),
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

        # Build match keys including CellType so that only cell types
        # present in new_df are replaced; cell types from prior runs
        # that are absent in new_df are preserved.
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
            'gene_alpha_hat_major', 'gene_alpha_hat_major_low', 'gene_alpha_hat_major_high',
            'gene_major_hap', 'gene_minor_hap',
            'gene_p_value', 'gene_p_value_adj',
            'ASE_call', 'ASTU_call',
            'dominant_isoform_overall',
            'top_isoform_hap_major', 'top_isoform_hap_minor',
            'top_isoform_hap_major_frac', 'top_isoform_hap_minor_frac',
            'isoform_p_value', 'isoform_p_value_high', 'isoform_p_value_low',
            'isoform_p_value_adj', 'isoform_p_value_adj_high', 'isoform_p_value_adj_low',
            'shrinkage_k', 'es_ase',
            'es_astu',
            'astu_source',
            'snvID',
            'snv_pos', 'snv_ref', 'snv_alt',
            'snv_depth_bulk', 'snv_alt_count_bulk', 'snv_alt_frac_bulk',
            'h_A', 'hat_Z_prob_revised',
            'snv_hap',
            'snv_on_minor_hap',
            'snv_expr_direction',
            'snv_es_ase_signed',
            'dominant_isoform_pref_hap',
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
            'dominant_isoform_overall', 'top_isoform_hap_major', 'top_isoform_hap_minor',
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
                 'dominant_isoform_overall', 'top_isoform_hap_major',
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
                        'dominant_isoform_overall', 'top_isoform_hap_major',
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

    # ------------------------------------------------------------------
    # Task 4a: Haplotype–event associations
    # ------------------------------------------------------------------

    def _haplotype_event_associations(self, min_reads=10, cell_type='Bulk',
                                       scotch_read_cell=None, read_hap_df=None,
                                       scotch_isoform_df=None, summary_df=None,
                                       ct_summary=None, bulk_sig_gene_ids=None,
                                       event_mode='all_events',
                                       fdr_events_value=0.05):
        """
        For each gene, test whether exon inclusion / splice-junction usage is
        associated with haplotype label, using read-haplotype probability weights.

        read-hap weights are from hat_I (hapA) and hat_I_B (hapB); exon/junction
        membership is derived from SCOTCH isoform annotation (not raw BAM), avoiding
        artefacts from read truncation.

        When cell_type != 'Bulk', reads are filtered to cells of that type using
        the scotch_read_cell lookup (Read+geneID → Cell) joined with cell_type_df.

        Within-gene FDR correction (BH) is applied.
        """
        for path, label in [(self.read_hap_map_path, 'read_hap_map'),
                             (self.scotch_tsv_path, 'SCOTCH TSV')]:
            if not os.path.exists(path):
                self._log(f'{label} not found: {path}')
                return None
        if self.gsi is None:
            self._log('geneStructureInformation not loaded')
            return None

        read_hap = pd.read_csv(self.read_hap_map_path) if read_hap_df is None else read_hap_df

        # Cell-type filtering: keep only reads from cells of the given CT
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

        # Narrow to relevant columns and genes for the per-gene join
        read_hap = read_hap[['geneID', 'Read', 'hat_I', 'hat_I_B']].copy()
        relevant_gene_ids = read_hap['geneID'].dropna().unique().tolist()
        if not relevant_gene_ids:
            return None
        relevant_gene_set = set(relevant_gene_ids)

        self._log(
            f'  Task 4a: preparing per-gene SCOTCH joins for '
            f'{len(relevant_gene_ids)} candidate genes ({cell_type}).')

        # Filter SCOTCH rows to only genes that survived filtering
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

        # Build per-gene Read→Isoform index for fast joins
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

        # variant_dir is the per-sample step1 output folder (where read_blocks.pkl
        # lives). Runs without step1_5+merge lack the pkl files → obs_test_type='no_bam'.
        variant_dir_for_obs = getattr(self, 'variant_dir', None)

        results_per_gene = Parallel(
            n_jobs=self.n_workers,
            prefer='threads',
            batch_size='auto',
        )(
            delayed(_process_gene_events_joined)(
                gene_id,
                g[['Read', 'hat_I', 'hat_I_B']],
                scotch_by_gene[gene_id],
                self.gsi,
                min_reads,
                gene_event_cache=self._get_gene_event_cache(gene_id),
                variant_dir=variant_dir_for_obs)
            for gene_id, g in read_hap.groupby('geneID', sort=False)
            if gene_id in scotch_by_gene
        )
        rows = [r for gene_rows in results_per_gene for r in gene_rows]

        if not rows:
            return None

        result = pd.DataFrame(rows)

        # Within-gene FDR for event chi-sq (p_value) and CIGAR-observed (obs_p_value).
        # obs_p_value can be NaN when read_blocks.pkl is absent (no_bam) or the
        # 2×2 table is too sparse (insufficient_data); those rows get NaN adj.
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
        """Find the two isoforms with largest opposite usage change between haplotypes."""
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
        """Collect gene-level events that touch switching boundaries for an isoform pair."""
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
                # Keep if at least one event is shared and at least one differs
                if any(in_other) and not all(in_other):
                    kept.add(boundary)
                # Terminal: only one event and it differs
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

        # Exclude events shared by both isoforms
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
                        start = int(fields[3]) - 1  # GTF 1-based → 0-based
                        end = int(fields[4])         # GTF end stays same
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
            return cache[gene_id]

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
        """
        Convert SCOTCH geneStructureInformation[geneID][1:3] to per-isoform event sets.

        exon_positions    : list[(start, end)] – gene-level SCOTCH sub-exons
        exon_isoform_dict : dict{isoform_name: [exon_index, ...]} – 0-based indices

        Returns
          iso_exon_map     : {isoform: set of sub-exon (start, end) tuples}
          iso_junction_map : fallback-only map from adjacent sub-exons when no GTF is available
        """
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

    # ------------------------------------------------------------------
    # Task 4b: Link SNVs to nearby events
    # ------------------------------------------------------------------

    def _link_snv_to_events(self, snv_df, hap_event_df,
                             max_exonic_dist=50):
        """
        For each haplotype–event pair, find SNVs within
        ±max_exonic_dist bp on the exonic coordinate axis.

        Linkage is not gated by hap-event significance so that the
        downstream raw SNV-event contingency test (Task 4c) can surface
        associations the EM-based test underestimates (e.g. when read
        truncation dilutes the haplotype-event signal near the SNV).

        Intronic SNVs are included but intron length is not counted toward
        distance (introns collapse to 0 width in exonic coordinates).

        Reports both exonic_distance and genomic_distance.
        """
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
                        'snv_hap': 'A' if snv['h_A'] > 0.5 else 'B',  # h_A == 0.5 defaults to B
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
        """
        Distance between snv_pos and the nearest boundary of [event_start, event_end]
        measured on the exonic coordinate axis (introns have zero width).

        If the SNV falls inside the event interval, distance is 0.
        """
        if event_start <= snv_pos <= event_end:
            return 0

        def to_exonic(pos):
            cum = 0
            for es, ee in exons_sorted:
                if pos < es:
                    return cum        # in an intron before this exon → clamp to exon start
                if pos <= ee:
                    return cum + (pos - es)
                cum += (ee - es)
            return cum

        snv_ex = to_exonic(snv_pos)
        return min(abs(snv_ex - to_exonic(event_start)),
                   abs(snv_ex - to_exonic(event_end)))

    # ------------------------------------------------------------------
    # Task 4c: Raw-read chi-squared test for SNV–event pairs
    # ------------------------------------------------------------------

    def _chi_sq_snv_event_raw(self, snv_event_df, cell_type='Bulk',
                              scotch_read_cell=None, cell_type_df=None,
                              scotch_isoform_df=None):
        """
        For each (SNV, event) pair, use raw BAM reads to build a
        2×2 contingency table:
                      event present | event absent
          ref allele:      rp       |     ra
          alt allele:      ap       |     aa

        Event membership is derived from the SCOTCH isoform assignment of each read
        (not from the BAM directly), to avoid read-truncation noise.
        """
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
        # Build read_name → list-of-isoforms. SCOTCH aux can hold multiple rows
        # for the same canonical read after read-name normalization (PacBio
        # sub-segment counter collapse) or for multi-locus alignments under any
        # platform. A scalar `read → single isoform` dict (.to_dict on a DataFrame
        # with duplicate index) would silently drop all but one isoform.
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

            # SNVs falling inside an exon event have degenerate cross-event
            # geometry: every read covering the SNV also covers the exon, so
            # the event-absent row of the 2x2 contingency table is 0 by
            # construction. Rather than skip these cases, we still collect
            # per-allele read counts and fall back to a binomial test on
            # ref vs alt counts (cis-eQTL standard).
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

                # Collect alleles at the SNV position
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
                        # SNV is inside the exon: by geometry every read
                        # covering the SNV covers the exon, so we skip the
                        # SCOTCH-isoform present/absent split and only
                        # tally ref vs alt allele counts. ra and aa stay 0.
                        if allele == alt_base:
                            ap += 1
                        elif allele == ref_base:
                            rp += 1
                    else:
                        isos = iso_lookup.get(read_name)
                        if not isos:
                            continue
                        # Multi-isoform reads (multi-locus alignment or PacBio
                        # sub-segment collapse) only contribute when every
                        # isoform agrees about event membership; mixed evidence
                        # is genuinely ambiguous for this 2x2 table cell.
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

        # Within-gene FDR on raw 2×2 chi-sq / binomial p-values. NaN raw_p
        # (test_type='insufficient_data' or 'binomial_intra_event' with n<10
        # or 'chi2_cross_event' with sparse table) → NaN adj.
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
        # The gsi pkl is a shared reference generated under the first scotch_target.
        # Fallback order: per-sample updated → meta wnovel → per-sample original → meta.
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
        # Meta pickles are keyed by meta-gene name with lists of per-gene
        # tuples; flatten to the gene-level dict the rest of the code expects.
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
        # variant_align1/ is shared across samples — step1 writes one joint
        # gene-level site_reads.pkl per gene (utils.py process_genes_round1_1
        # merges across samples) and step3 reads the same flat path. step1.5
        # (merge_read_blocks_round1_5) writes the canonical read_blocks.pkl
        # alongside it for step5 obs_* / step3 Knob D consumption.
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

            if self.sample_name_parse:
                scotch_tsv_path = os.path.join(
                    scotch_target,
                    f'samples/{self.sample_name_parse}/auxillary/all_read_isoform_exon_mapping.tsv')
            else:
                scotch_tsv_path = os.path.join(scotch_target, 'auxillary/all_read_isoform_exon_mapping.tsv')

            configs.append({
                'sample_name': sample_name,
                'snv_hap_map_path': os.path.join(snv_hap_dir, 'snv_hap_map.csv'),
                'read_hap_map_path': os.path.join(snv_hap_dir, 'read_hap_map.csv'),
                'summary_statistics_path': os.path.join(summary_dir, 'summary_statistics.csv'),
                'count_dir': count_dir,
                'isoform_agg_balance_path': os.path.join(count_dir, 'all_genes', 'isoform_agg_balance.csv'),
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
