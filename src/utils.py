import pysam
from collections import defaultdict
import pandas as pd
import os
import sys
import pickle
import hashlib
import zlib
import io
import numpy as np
import anndata as ad
from src.inference import (assign_reads_frozen_markers, linkage_agreement, linkage_loglr, run_em, compute_P_het, guarded_switch_flip,
                           assign_phase_blocks, assign_read_blocks,
                           block_aware_alpha_bounds,
                           block_orientation_alpha_bounds)
from src.compat import resolve_scotch_auxiliary_tsv
from scipy import sparse
from scipy.io import mmwrite
from typing import Union, Sequence
import warnings
import logging
from scipy.stats import chi2
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, betaln
import subprocess
import tempfile
import threading
from collections.abc import Iterable
from joblib import Parallel, delayed, load as joblib_load
from src.statistical_test import observed_loglikelihood, run_em_fixed_alpha
from src.compat import _collapse_legacy_merge_suffixes
from statsmodels.stats.multitest import multipletests
import math
import re
from scipy.sparse import csr_matrix


def load_pickle(file):
    if os.path.exists(file):
        with open(file,'rb') as file:
            data=pickle.load(file)
    else:
        data = None
    return data


_READNAME_SUFFIX_RE = re.compile(r'_\d+$')


def canonicalize_read_name(name):
    if not isinstance(name, str):
        return name
    if '/' in name:
        return _READNAME_SUFFIX_RE.sub('', name)
    return name


def warn_if_output_is_symlinked(paths, logger=None):
    out = []
    for p in paths:
        if not p:
            continue
        real = os.path.realpath(p)
        if real != os.path.abspath(p):
            out.append((p, real))
            _log_with_fallback(logger, f'[WARN] output path is (or sits under) a '
                                       f'symlink: {p} -> {real}; files will land '
                                       f'in the target, not where the logs say')
    return out


_PILEUP_FLAG_FILTER = 0x4 | 0x100 | 0x200 | 0x400 | 0x800


def _keep_is_one_bytes(x):
    x = x.strip()
    if x == b'1':
        return True
    try:
        return float(x) == 1.0
    except ValueError:
        return False


def compute_intron_spans(read):
    spans = []
    cigar = read.cigartuples
    if cigar is None:
        return spans
    ref_pos = read.reference_start
    for op, length in cigar:
        if op == 3:
            spans.append((ref_pos, ref_pos + length))
            ref_pos += length
        elif op in (0, 2, 7, 8):
            ref_pos += length

    return spans


_WORKER_LOG_HANDLERS = {}


def log_file_of(logger):
    node = logger
    while node is not None:
        for h in getattr(node, 'handlers', ()):
            base = getattr(h, 'baseFilename', None)
            if base:
                return base
        node = node.parent if getattr(node, 'propagate', True) else None
    return None


def _has_file_handler(logger, path):
    node = logger
    while node is not None:
        for h in getattr(node, 'handlers', ()):
            base = getattr(h, 'baseFilename', None)
            if base and os.path.abspath(base) == path:
                return True
        node = node.parent if getattr(node, 'propagate', True) else None
    return False


def attach_worker_log_handler(logger, log_file):
    if logger is None or not log_file:
        return False
    path = os.path.abspath(str(log_file))
    if _has_file_handler(logger, path):
        return False
    name = getattr(logger, 'name', '')
    prev = _WORKER_LOG_HANDLERS.get(name)
    if prev is not None and prev[0] != path:
        try:
            logger.removeHandler(prev[1])
            prev[1].close()
        except Exception:
            pass
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    _WORKER_LOG_HANDLERS[name] = (path, fh)
    return True


def _log_with_fallback(logger, message):
    if logger is None:
        print(message, flush=True)
        return
    has = getattr(logger, 'hasHandlers', None)


    if has is not None and not has():
        print(message, flush=True)
        return
    logger.info(message)


def _resolve_reference_pickle_path(scotch_target, logger=None):
    ref_dir = os.path.join(scotch_target[0], 'reference')
    candidates = [
        'geneStructureInformationupdated.pkl',
        'metageneStructureInformationwnovel.pkl',
        'geneStructureInformation.pkl',
        'metageneStructureInformation.pkl',
    ]
    for name in candidates:
        path = os.path.join(ref_dir, name)
        if os.path.isfile(path):
            _log_with_fallback(logger, f'Resolved reference pickle: {name}')
            return path
    _log_with_fallback(
        logger,
        f'No reference pickle found under {ref_dir}; checked {", ".join(candidates)}.'
    )
    return None


def _gsi_from_bytes(raw, gsi_path, logger=None):
    gsi = pickle.loads(raw)
    if 'meta' in os.path.basename(gsi_path).lower():
        flat = {}
        for genes_info_list in gsi.values():
            for gene_info, exon_info, isoform_info in genes_info_list:
                flat[gene_info['geneID']] = (gene_info, exon_info, isoform_info)
        _log_with_fallback(logger, f'Flattened meta pickle to {len(flat)} genes')
        gsi = flat
    return gsi


def _digest(raw):
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def _load_asset_bytes(path, expected_digest=None, kind='asset'):
    with open(path, 'rb') as fh:
        raw = fh.read()
    got = _digest(raw)
    if expected_digest is not None and got != expected_digest:
        raise RuntimeError(
            f'{path} changed while step 3 was fanning out ({kind} content '
            f'digest {got[:12]} != {expected_digest[:12]}); the worker would '
            f'have run on a different {kind} than the parent.')
    return raw, got


def _load_gsi_with_digest(gsi_path, logger=None):
    if gsi_path is None:
        return None, None
    try:
        raw, digest = _load_asset_bytes(gsi_path, kind='gene structure')
    except OSError:
        _log_with_fallback(logger, f'Unable to load geneStructureInformation pickle: {gsi_path}')
        return None, None
    return _gsi_from_bytes(raw, gsi_path, logger), digest


def _load_gene_structure_information(gsi_path, logger=None):
    if gsi_path is None:
        return None
    gsi = load_pickle(gsi_path)
    if gsi is None:
        _log_with_fallback(logger, f'Unable to load geneStructureInformation pickle: {gsi_path}')
        return None
    if 'meta' in os.path.basename(gsi_path).lower():
        flat = {}
        for genes_info_list in gsi.values():
            for gene_info, exon_info, isoform_info in genes_info_list:
                flat[gene_info['geneID']] = (gene_info, exon_info, isoform_info)
        _log_with_fallback(logger, f'Flattened meta pickle to {len(flat)} genes')
        gsi = flat
    return gsi


def _resolve_base_reference_pickle_path(scotch_target, logger=None):
    ref_dir = os.path.join(scotch_target[0], 'reference')
    candidates = [
        'geneStructureInformation.pkl',
        'metageneStructureInformation.pkl',
    ]
    for name in candidates:
        path = os.path.join(ref_dir, name)
        if os.path.isfile(path):
            _log_with_fallback(logger, f'[high_artifact_mode] Resolved base reference pickle: {name}')
            return path
    _log_with_fallback(
        logger,
        f'[high_artifact_mode] No base reference pickle under {ref_dir}; checked {", ".join(candidates)}.'
    )
    return None


def build_same_individual_inputs(scotch_targets, bam_paths, sample_names,
                                 out_dir, cell_type_df_path=None,
                                 need_bam=True, logger=None):
    import json as _json
    if cell_type_df_path:
        raise SystemExit(
            '--same_individual does not take --cell_type_df_path: sample '
            'names BECOME the cell types (bulk inputs only — a real '
            'single-cell barcode layer would be destroyed by the pooling).')
    if len(set(sample_names)) != len(sample_names):
        raise SystemExit(f'--same_individual needs distinct sample names, '
                         f'got {sample_names}; pass --sample_names.')
    if len(sample_names) != len(scotch_targets):
        raise SystemExit(f'{len(sample_names)} sample names for '
                         f'{len(scotch_targets)} scotch targets.')
    if bam_paths and len(bam_paths) != len(scotch_targets):
        raise SystemExit(f'{len(bam_paths)} BAMs for '
                         f'{len(scotch_targets)} scotch targets — one each.')
    pooled_dir = os.path.join(out_dir, 'same_individual_scotch')
    aux_dir = os.path.join(pooled_dir, 'auxillary')
    tsv_path = os.path.join(aux_dir, 'all_read_isoform_exon_mapping.tsv')
    ct_path = os.path.join(pooled_dir, 'sample_celltype.csv')
    bam_out = os.path.join(out_dir, 'same_individual_pooled.bam')
    manifest_path = os.path.join(pooled_dir, 'POOL_MANIFEST.json')
    def _sig(path):
        try:
            st = os.stat(path)
            return [os.path.abspath(path), st.st_mtime_ns, st.st_size]
        except OSError:
            return [os.path.abspath(path), None, None]


    manifest = {'samples': list(sample_names),
                'scotch_targets': [_sig(resolve_scotch_auxiliary_tsv(st, None))
                                   for st in scotch_targets],
                'bams': ([_sig(b) for b in bam_paths] if bam_paths else None)}

    def _log(msg):
        print(msg) if logger is None else logger.info(msg)

    fresh = True
    if os.path.exists(manifest_path):
        try:
            fresh = _json.load(open(manifest_path)) != manifest
        except Exception:
            fresh = True
    if fresh:
        os.makedirs(aux_dir, exist_ok=True)
        parts, owner = [], {}
        for name, st in zip(sample_names, scotch_targets):
            mp = resolve_scotch_auxiliary_tsv(st, None)
            df = pd.read_csv(mp, sep='\t', dtype=str)
            dup = set(df['Read']) & set(owner)
            if dup:
                some = sorted(dup)[:3]
                raise SystemExit(
                    f'--same_individual: {len(dup)} read names of {name!r} '
                    f'already appear in {owner[some[0]]!r} (e.g. {some}) — a '
                    f'cross-sample QNAME collision would double-count reads '
                    f'in the pooled EM. Refusing.')
            for r in df['Read']:
                owner[r] = name
            df['Cell'] = name
            if 'CBUMI' in df.columns:
                df['CBUMI'] = name + '_' + df['Umi'].astype(str)
            parts.append(df)
            _log(f'[same_individual] {name}: {len(df)} mapping rows from {st}')
        merged = pd.concat(parts, ignore_index=True)
        merged = merged.sort_values('geneID', kind='stable', ignore_index=True)
        merged.to_csv(tsv_path, sep='\t', index=False)

        gene_col = list(merged.columns).index('geneID')
        with open(tsv_path, 'rb') as fh, open(tsv_path + '.geneidx.tmp', 'w') as ix:
            offset = len(fh.readline())
            cur, start, nrows = None, offset, 0
            for line in fh:
                gid = line.split(b'\t')[gene_col].decode()
                if gid != cur:
                    if cur is not None:
                        ix.write(f'{cur}\t{start}\t{offset - start}\t{nrows}\n')
                    cur, start, nrows = gid, offset, 0
                nrows += 1
                offset += len(line)
            if cur is not None:
                ix.write(f'{cur}\t{start}\t{offset - start}\t{nrows}\n')
        os.replace(tsv_path + '.geneidx.tmp', tsv_path + '.geneidx.tsv')
        ref_dst = os.path.join(pooled_dir, 'reference')
        if os.path.lexists(ref_dst):
            os.remove(ref_dst)
        os.symlink(os.path.abspath(os.path.join(scotch_targets[0], 'reference')),
                   ref_dst)
        pd.DataFrame({'Cell': list(sample_names),
                      'CellType': list(sample_names)}).to_csv(ct_path, index=False)


        for stale in (bam_out, bam_out + '.bai'):
            if os.path.exists(stale):
                os.remove(stale)
        with open(manifest_path + '.tmp', 'w') as f:
            _json.dump(manifest, f, indent=1)
        os.replace(manifest_path + '.tmp', manifest_path)
        _log(f'[same_individual] pooled scotch -> {pooled_dir} '
             f'({len(merged)} rows, {len(sample_names)} samples-as-celltypes)')
    else:
        _log(f'[same_individual] reusing pooled inputs at {pooled_dir} '
             f'(manifest matches)')

    out = {'scotch_target': pooled_dir, 'cell_type_df_path': ct_path,
           'bam_path': None}
    if need_bam:
        if not bam_paths:
            raise SystemExit('--same_individual with this task needs '
                             '--bam_path (one per sample).')


        if (fresh or not os.path.exists(bam_out)
                or not os.path.exists(bam_out + '.bai')):
            _log(f'[same_individual] merging {len(bam_paths)} BAMs -> {bam_out}')
            tmp_bam = bam_out + '.tmp.bam'
            pysam.merge('-f', tmp_bam, *[str(b) for b in bam_paths])
            pysam.index(tmp_bam)


            if os.path.exists(bam_out + '.bai'):
                os.remove(bam_out + '.bai')
            os.replace(tmp_bam, bam_out)
            os.replace(tmp_bam + '.bai', bam_out + '.bai')
        out['bam_path'] = bam_out
    return out


def compute_nascent_leak_intervals(gsi_updated, logger=None):
    leak = {}
    if not gsi_updated:
        return leak
    for geneID, info in gsi_updated.items():
        if isinstance(geneID, str) and geneID.startswith('gene_'):
            continue
        if not isinstance(info, (tuple, list)) or len(info) < 3:
            continue
        geneInfo, exon_positions, exon_isoform_dict = info[0], info[1], info[2]
        if not exon_positions or not isinstance(exon_isoform_dict, dict):
            continue
        try:
            n_exons = len(exon_positions)
        except TypeError:
            continue
        all_idx = set(range(n_exons))
        referenced = set()
        for indices in exon_isoform_dict.values():
            for i in indices:
                try:
                    referenced.add(int(i))
                except (TypeError, ValueError):
                    pass
        novel_idx = all_idx - referenced
        try:
            gene_span = int(geneInfo['geneEnd']) - int(geneInfo['geneStart'])
        except (KeyError, TypeError, ValueError):
            continue
        canon_len = sum(int(exon_positions[i][1]) - int(exon_positions[i][0]) for i in referenced)
        intron_len = gene_span - canon_len
        if intron_len <= 0:
            continue
        if not novel_idx:
            leak[geneID] = (0.0, [])
            continue
        novel_intervals = sorted(
            (int(exon_positions[i][0]), int(exon_positions[i][1])) for i in novel_idx
        )
        novel_len = sum(e - s for s, e in novel_intervals)
        leak[geneID] = (novel_len / intron_len, novel_intervals)
    _log_with_fallback(
        logger,
        f'[high_artifact_mode][KnobB] precomputed nascent_leak_intervals for {len(leak)} ENST genes'
    )
    return leak


def compute_canonical_exons(gsi_base, logger=None):
    out = {}
    if not gsi_base:
        return out
    for geneID, info in gsi_base.items():
        if isinstance(geneID, str) and geneID.startswith('gene_'):
            continue
        if not isinstance(info, (tuple, list)) or len(info) < 2:
            continue
        geneInfo, exon_positions = info[0], info[1]
        if not exon_positions:
            continue
        try:
            chrom = str(geneInfo['geneChr'])
            gstart = int(geneInfo['geneStart'])
            gend = int(geneInfo['geneEnd'])
        except (KeyError, TypeError, ValueError):
            continue
        intervals = sorted((int(s), int(e)) for s, e in exon_positions)
        out[geneID] = (chrom, gstart, gend, intervals)
    _log_with_fallback(
        logger,
        f'[high_artifact_mode][KnobC] precomputed canonical_exons for {len(out)} ENST genes'
    )
    return out


def _overlap_bp(s, e, sorted_intervals):
    if e <= s or not sorted_intervals:
        return 0
    total = 0
    for ints, inte in sorted_intervals:
        if inte <= s:
            continue
        if ints >= e:
            break
        total += min(e, inte) - max(s, ints)
    return total


class BamInputError(FileNotFoundError):
    pass


class KnobCInputError(BamInputError):
    pass


_CHROM_BAM_NAME_RE = re.compile(r'^chr(\d+|[XYM]|MT)$')


def resolve_bam_file(bam_path, chrom=None, cache=None):
    if os.path.isfile(bam_path):
        return bam_path
    if not os.path.isdir(bam_path):
        raise FileNotFoundError(f'BAM path is neither a file nor a directory: {bam_path}')
    if chrom is None:
        raise ValueError(f'chrom is required when bam_path is a directory: {bam_path}')
    chrom_cache = cache.get(bam_path) if cache is not None else None
    if chrom_cache is None:
        chrom_cache = {}


        names = set()
        for fname in os.listdir(bam_path):
            if fname.endswith('.bam'):
                names.add(fname)
            elif fname.endswith('.pgbam'):
                names.add(fname[:-len('.pgbam')] + '.bam')
        for fname in sorted(names):
            parts = fname.replace('.bam', '').split('.')
            for part in parts:
                if _CHROM_BAM_NAME_RE.match(part):
                    chrom_cache.setdefault(part, os.path.join(bam_path, fname))
                    break
        if cache is not None:
            cache[bam_path] = chrom_cache
    bam_file = chrom_cache.get(chrom)
    if bam_file is None:
        raise FileNotFoundError(f'No BAM found for chromosome {chrom} in {bam_path}')
    return bam_file


def check_knob_c_bams(bam_paths, logger=None, chroms=None):
    if not bam_paths:
        raise KnobCInputError('--high_artifact_mode: the Knob C read filter needs '
                              '--bam_path, and none was given')
    out = {}
    for bp in (bam_paths if isinstance(bam_paths, (list, tuple)) else [bam_paths]):
        if bp is None:
            raise KnobCInputError('--high_artifact_mode: a --bam_path entry is None')
        if os.path.isfile(bp):
            with pysam.AlignmentFile(str(bp), 'rb') as bam:
                if not bam.has_index():
                    raise KnobCInputError(f'--high_artifact_mode: {bp} has no index; the '
                                          f'Knob C read filter fetches by gene region')
            out[bp] = 'file'
        elif os.path.isdir(bp):
            cache = {}
            try:
                resolve_bam_file(bp, chrom='chr1', cache=cache)
            except FileNotFoundError:
                pass
            found = cache.get(bp, {})
            if not found:
                raise KnobCInputError(f'--high_artifact_mode: {bp} is a directory but holds no '
                                      f'per-chromosome BAM (expected names carrying chr1..chr22/X/Y/M)')
            if chroms:


                missing = sorted(c for c in set(chroms)
                                 if _CHROM_BAM_NAME_RE.match(str(c)) and c not in found)
                if missing:
                    raise KnobCInputError(f'--high_artifact_mode: {bp} has no BAM for '
                                          f'{len(missing)} chromosome(s) this run needs: '
                                          f'{", ".join(missing[:8])}{"…" if len(missing) > 8 else ""}')


            for chrom, member in sorted(found.items()):
                if not os.path.isfile(member):
                    raise KnobCInputError(f'--high_artifact_mode: {member} is listed (petagene '
                                          f'.pgbam) but not readable from this node -- run on a '
                                          f'compute node with the petagene module loaded')
                with pysam.AlignmentFile(member, 'rb') as bam:
                    if not bam.has_index():
                        raise KnobCInputError(f'--high_artifact_mode: {member} has no index; the '
                                              f'Knob C read filter fetches by gene region')
                    if chrom not in bam.references:
                        raise KnobCInputError(f'--high_artifact_mode: {member} is named for {chrom} '
                                              f'but its header has no such contig')
            out[bp] = f'directory of {len(found)} per-chromosome BAMs'
        else:
            raise KnobCInputError(f'--high_artifact_mode: --bam_path {bp} does not exist')
        _log_with_fallback(logger, f'[high_artifact_mode][KnobC] BAM ok: {bp} ({out[bp]})')
    return out


def compute_knob_c_blacklist(geneID, sample_index, bam_path_list, canonical_exons,
                             read_intronic_pct_max, candidate_reads, logger=None,
                             bam_cache=None):
    stats = {'status': 'ok', 'n_candidates': len(candidate_reads) if candidate_reads else 0,
             'n_found': 0, 'bam': None}
    info = canonical_exons.get(geneID) if canonical_exons else None
    if info is None:
        stats['status'] = 'no_annotation'
        return set(), stats
    chrom, gstart, gend, exon_intervals = info
    if not bam_path_list:
        raise KnobCInputError(f'[KnobC] {geneID}: the read filter needs --bam_path, none given')
    if isinstance(bam_path_list, (list, tuple)):
        if not (0 <= sample_index < len(bam_path_list)):
            raise KnobCInputError(f'[KnobC] {geneID}: sample index {sample_index} but only '
                                  f'{len(bam_path_list)} BAM path(s)')
        bam_path = bam_path_list[sample_index]
    else:
        bam_path = bam_path_list
    if bam_path is None:
        raise KnobCInputError(f'[KnobC] {geneID}: no BAM for sample index {sample_index}')
    try:
        bam_file = resolve_bam_file(bam_path, chrom=chrom, cache=bam_cache)
    except (FileNotFoundError, ValueError) as exc:
        if os.path.isdir(str(bam_path)) and not _CHROM_BAM_NAME_RE.match(str(chrom)):


            stats['status'] = 'no_bam_for_contig'
            return set(), stats
        raise KnobCInputError(f'[KnobC] {geneID}: {exc}') from exc
    stats['bam'] = bam_file
    if not candidate_reads:
        stats['status'] = 'no_reads'
        return set(), stats
    needed = set(candidate_reads)
    blacklist = set()
    seen_qnames = set()
    try:
        bam = pysam.AlignmentFile(bam_file, 'rb')
    except (OSError, ValueError) as exc:
        raise KnobCInputError(f'[KnobC] {geneID}: cannot open {bam_file}: {exc}') from exc
    try:


        try:
            for read in bam.fetch(chrom, gstart, gend):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                qname = canonicalize_read_name(read.query_name)
                if qname not in needed or qname in seen_qnames:
                    continue
                blocks = read.get_blocks()
                if not blocks:
                    continue
                total = sum(e - s for s, e in blocks)
                if total == 0:
                    continue
                exonic = sum(_overlap_bp(s, e, exon_intervals) for s, e in blocks)
                intronic_pct = (total - exonic) / total
                if intronic_pct > read_intronic_pct_max:
                    blacklist.add(qname)
                seen_qnames.add(qname)
        except (ValueError, OSError) as exc:
            raise KnobCInputError(f'[KnobC] {geneID}: cannot read {chrom}:{gstart}-{gend} '
                                  f'from {bam_file}: {exc}') from exc
    finally:
        bam.close()
    stats['n_found'] = len(seen_qnames)
    return blacklist, stats


_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def _model(name):
    return os.path.join(_MODEL_DIR, name)


_CLF_ONT = {'snv_classifier': _model('snv_classifier_ont_cdna_hg002_clfbeta_17feat.joblib'),
            'clf_hard_threshold': 0.05}
_CLF_HIFI = {'snv_classifier': _model('snv_classifier_hifi_isoseq_hg002_clfbeta_17feat.joblib'),
             'clf_hard_threshold': 0.05}
_CLF_MASSEQ = {'snv_classifier': _model('snv_classifier_hifi_masseq_hg002_clfbeta_17feat.joblib'),
               'clf_hard_threshold': 0.05}
_CLF_DRNA = {'snv_classifier': _model('snv_classifier_ont_drna_hg002_clfbeta_17feat.joblib'),
             'clf_hard_threshold': 0.05}


_CLF_NONE = {'snv_classifier': None}


_CLF_FREE = {'snv_classifier': None, 'h_m_init_from': 'linkage', 'het_beta': None}


DECIDED_READ_GATES = {'min_mapq': 20, 'light_assign_by': 'gene_range',
                      'light_min_exonic_bp': 0,
                      'light_ambiguity_ratio': 1.0}

SETTING_OVERLAYS = {'clf': {'het_beta': '2,2', 'h_m_init_from': 'clf'},
                    'clf-free': _CLF_FREE}


_STRETCH_OFF = {'alt_stretch_len': 0}
_STRETCH_7 = {'alt_stretch_len': 7}
_STRETCH_5 = {'alt_stretch_len': 5}


_EDIT_EXEMPT = {'editing_exempt_affinity': 0.95, 'editing_exempt_min_reads': 10}


PLATFORM_PRESETS = {


    'ont-cdna':    {'het_prob_step1': 0.8, 'het_prob_step3': 0.99,
                    'em_max_reads': 20000,
                    'depth': 10, 'n_alt_count': 2, 'min_alt_frac': 0.0,
                    'min_baseq': 0, 'max_baseq': 30, 'min_dist_to_end': 3,
                    'gene_guard_depth': 20,
                    **_CLF_ONT, **_STRETCH_7, **_EDIT_EXEMPT},


    'ont-drna':    {'het_prob_step1': 0.8, 'het_prob_step3': 0.99,
                    'em_max_reads': 20000,
                    'depth': 10, 'n_alt_count': 2, 'min_alt_frac': 0.0,
                    'min_baseq': 0, 'max_baseq': 30, 'min_dist_to_end': 3,
                    'gene_guard_depth': 20,
                    **_CLF_DRNA, **_STRETCH_7, **_EDIT_EXEMPT},


    'hifi-isoseq': {'het_prob_step1': 0.8, 'het_prob_step3': 0.99,
                    'em_max_reads': 20000,
                    'depth': 10, 'n_alt_count': 2, 'min_alt_frac': 0.0,
                    'min_baseq': 0, 'max_baseq': 30, 'min_dist_to_end': 3,
                    'gene_guard_depth': 20,
                    **_CLF_HIFI, **_STRETCH_OFF, **_EDIT_EXEMPT},


    'hifi-masseq': {'het_prob_step1': 0.8, 'het_prob_step3': 0.99,
                    'em_max_reads': 20000,
                    'depth': 10, 'n_alt_count': 2, 'min_alt_frac': 0.0,
                    'min_baseq': 0, 'max_baseq': 30, 'min_dist_to_end': 3,
                    'gene_guard_depth': 20,
                    **_CLF_MASSEQ, **_STRETCH_OFF, **_EDIT_EXEMPT},


    'other':       {'het_prob_step1': 0.8, 'het_prob_step3': 0.99,
                    'em_max_reads': 20000,
                    'depth': 10, 'n_alt_count': 2, 'min_alt_frac': 0.0,
                    'min_baseq': 0, 'max_baseq': 30, 'min_dist_to_end': 3,
                    'gene_guard_depth': 20,
                    **_CLF_NONE, **_STRETCH_5, **_EDIT_EXEMPT},
}


def _flag_given(argv, dest):
    flag = '--' + dest
    return any(a == flag or a.startswith(flag + '=') for a in argv)


def em_cap_subsample_indices(n_reads, cap, seed, gene_id):
    rng = np.random.default_rng(
        [int(seed) if seed is not None else 0,
         zlib.crc32(str(gene_id).encode())])
    return np.sort(rng.choice(n_reads, size=cap, replace=False))


def run_em_capped(df_r, df_pi, cap, gene_id, logger=None, **em_kwargs):
    n = df_r.shape[0]
    if not (cap and cap > 0) or n <= cap:
        return run_em(df_r, df_pi, **em_kwargs)
    sub = em_cap_subsample_indices(n, cap, em_kwargs.get('seed'), gene_id)
    mes = (f'[EM-CAP] {gene_id}: {n} reads > cap {cap} — markers learned on '
           f'{cap} subsampled reads; all reads then assigned by the learned '
           f'markers, alpha re-estimated on all reads')


    _log_with_fallback(logger, mes)
    sub_fit = run_em(df_r.iloc[sub], df_pi.iloc[sub], **em_kwargs)
    if np.sum(sub_fit['h_m'] > 0.5) == 0:
        return None


    full = assign_reads_frozen_markers(
        df_r, df_pi, sub_fit,
        max_iter=em_kwargs.get('max_iter', 50), tol=em_kwargs.get('tol', 1e-3),
        filter_reads=em_kwargs.get('filter_reads', True))


    full['h_A_init_used'] = sub_fit['h_A_init_used']
    full['h_m_init_used'] = sub_fit['h_m_init_used']
    return full


def apply_platform_preset(args, argv=None, logger=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    plat = getattr(args, 'platform', None)
    setting = getattr(args, 'setting', None)
    def _say(m):
        print(m) if logger is None else logger.info(m)


    for k, v in DECIDED_READ_GATES.items():
        if not hasattr(args, k):
            continue
        if _flag_given(argv, k):
            _say(f'[decided] {k}={getattr(args, k)} (user override, decided value '
                 f'was {v})')
            continue
        setattr(args, k, v)
        _say(f'[decided] {k}={v} (decided, all platforms)')
    if not plat:


        _say('[decided] no --platform given: the read gates above are the '
             'decided defaults, and the platform-specific parameters '
             '(classifier, homopolymer gate, depth, het cutoffs) keep whatever '
             'this command line set. That is a complete configuration, but it '
             'was not declared -- pass --platform for a run whose numbers are '
             'reported anywhere.')
        if setting:
            raise ValueError('--setting needs --platform: the settings are the two '
                             'halves of a PLATFORM\'s decided configuration, and '
                             'there is nothing to be a half of without one')
        return args
    if plat not in PLATFORM_PRESETS:
        raise ValueError(f'unknown --platform {plat!r}; choose from '
                         f'{sorted(PLATFORM_PRESETS)}')


    if setting is None:


        setting = 'clf' if PLATFORM_PRESETS[plat].get('snv_classifier') else 'clf-free'
        _say(f'[platform {plat}] setting={setting} (default: the platform '
             f'{"has" if setting == "clf" else "has no"} a classifier)')
    elif setting not in SETTING_OVERLAYS:
        raise ValueError(f'unknown --setting {setting!r}; choose from '
                         f'{sorted(SETTING_OVERLAYS)}')
    elif setting == 'clf' and not PLATFORM_PRESETS[plat].get('snv_classifier'):
        raise ValueError(f'--platform {plat} ships no classifier, so --setting clf '
                         f'cannot run; use --setting clf-free (its default) or pass '
                         f'--snv_classifier <your model> with a platform that has one')
    else:
        _say(f'[platform {plat}] setting={setting} (explicit)')
    resolved = dict(PLATFORM_PRESETS[plat], **SETTING_OVERLAYS[setting])
    for key, val in resolved.items():
        if not hasattr(args, key):
            _say(f'[platform {plat}] {key}={val} NOT applied — this entry '
                 f'point has no such option (preset value ignored)')
            continue
        if _flag_given(argv, key):
            _say(f'[platform {plat}] {key}={getattr(args, key)} (user override, '
                 f'preset was {val})')
        else:
            setattr(args, key, val)
            _say(f'[platform {plat}] {key}={val} (preset)')
    return args


MAPQ_UNAVAILABLE_VALUES = (0, 255)
MAPQ_PROBE_READS = 100_000


MAPQ_POLICY_TASKS = ('light_prep', 'step1', 'step1_5', 'step3')
MAPQ_RECORD_KEYS = ('mapq_verdict', 'mapq_fallback')


def bam_members(bam_path, cache=None):
    if os.path.isfile(bam_path):
        return [bam_path]
    cache = {} if cache is None else cache
    try:
        resolve_bam_file(bam_path, chrom='chr1', cache=cache)
    except FileNotFoundError:
        pass
    found = cache.get(bam_path, {})
    if not found:
        raise FileNotFoundError(f'{bam_path}: not a BAM file and not a directory holding '
                                f'per-chromosome BAMs')
    special = {'X': 0, 'Y': 1, 'M': 2, 'MT': 3}
    def _order(k):
        c = k[3:]
        return (0, int(c), '') if c.isdigit() else (1, special.get(c, 9), c)
    return [found[k] for k in sorted(found, key=_order)]


def probe_mapq(bam_path, n_reads=MAPQ_PROBE_READS):
    members = bam_members(str(bam_path))
    per = []
    sampled = 0
    for j, member in enumerate(members):


        m_budget = max(1, int(math.ceil((n_reads - sampled) / (len(members) - j))))
        m = {'member': os.path.basename(member), 'sampled': 0, 'zero': 0, 'v255': 0,
             '_valid_values': []}
        with pysam.AlignmentFile(member, 'rb') as bam:
            try:
                stats = [st for st in bam.get_index_statistics() if st.mapped > 0]
            except (ValueError, OSError):
                stats = None
            if stats:
                share = max(1, int(math.ceil(m_budget / len(stats))))
                iters = [(bam.fetch(st.contig), share) for st in stats]
            else:
                iters = [(bam.fetch(until_eof=True), m_budget)]
            for it, budget in iters:
                taken = 0
                for aln in it:
                    if taken >= budget or m['sampled'] >= m_budget:
                        break
                    if aln.flag & _PILEUP_FLAG_FILTER:
                        continue
                    q = int(aln.mapping_quality)
                    taken += 1
                    m['sampled'] += 1
                    if q == 0:
                        m['zero'] += 1
                    elif q == 255:
                        m['v255'] += 1
                    else:
                        m['_valid_values'].append(q)
                if m['sampled'] >= m_budget:
                    break
        sampled += m['sampled']
        m['valid'] = len(m['_valid_values'])
        if m['sampled'] == 0:
            m['verdict'], m['kind'] = 'no_reads', None
        elif m['valid']:
            m['verdict'], m['kind'] = 'available', None
        else:
            m['verdict'] = 'unavailable'
            m['kind'] = ('all_zero' if m['v255'] == 0 else
                         'all_255' if m['zero'] == 0 else 'zero_or_255')
        per.append(m)
    zero = sum(m['zero'] for m in per)
    v255 = sum(m['v255'] for m in per)
    valid = [q for m in per for q in m['_valid_values']]
    label = os.path.basename(str(bam_path).rstrip('/'))
    if len(members) > 1 or not os.path.isfile(str(bam_path)):
        label += f' ({len(members)} per-chromosome BAMs)'
    out = {'bam': label, 'sampled': sampled,
           'zero': zero, 'v255': v255, 'valid': len(valid),
           'min_valid': min(valid) if valid else None,
           'max_valid': max(valid) if valid else None,
           'median_valid': float(np.median(valid)) if valid else None,
           'kind': None,


           'members': [{k: v for k, v in m.items() if k != '_valid_values'} for m in per],
           '_member_valid_values': [m['_valid_values'] for m in per]}
    bad = [m for m in per if m['verdict'] == 'unavailable']
    if sampled == 0:
        out['verdict'] = 'no_reads'
    elif bad:
        out['verdict'] = 'unavailable'
        out['kind'] = (', '.join(f"{m['kind']} in {m['member']}" for m in bad)
                       if len(members) > 1 else bad[0]['kind'])
    else:
        out['verdict'] = 'available'
    out['_valid_values'] = valid
    return out


def mapq_gate_survivors(probe, gate):
    vals = probe.get('_valid_values') or []
    return (sum(1 for q in vals if q >= gate)
            + (probe['zero'] if gate <= 0 else 0)
            + (probe['v255'] if gate <= 255 else 0))


def resolve_mapq_policy(args, argv=None, logger=None, n_reads=MAPQ_PROBE_READS):
    argv = list(sys.argv[1:] if argv is None else argv)
    say = lambda m: _log_with_fallback(logger, m)
    if getattr(args, '_mapq_policy_resolved', False):


        return args._mapq_policy
    task = getattr(args, 'task', None)
    bams = getattr(args, 'bam_path', None)
    if task not in MAPQ_POLICY_TASKS or not bams:
        args._mapq_policy = None
        args._mapq_policy_resolved = True
        return None
    if isinstance(bams, str):
        bams = [bams]
    probes = [probe_mapq(b, n_reads=n_reads) for b in bams]
    gate = int(getattr(args, 'min_mapq', 0) or 0)
    for pr in probes:
        n = pr['sampled']
        if n == 0:
            say(f"[mapq] {pr['bam']}: no primary reads sampled -- MAPQ cannot "
                f"be judged; the gate min_mapq={gate} applies as given")
            pr['survivors_at_gate'] = None
            continue
        surv = mapq_gate_survivors(pr, gate)
        pr['survivors_at_gate'] = surv


        starved = [m['member'] for m, vals in zip(pr.get('members', []),
                                                 pr.get('_member_valid_values', []))
                   if m['sampled'] > 0 and m['verdict'] == 'available'
                   and mapq_gate_survivors({'zero': m['zero'], 'v255': m['v255'],
                                            '_valid_values': vals}, gate) == 0]
        say(f"[mapq] {pr['bam']}: sampled {n} primary reads -- "
            f"MAPQ 0: {pr['zero'] / n:.1%}  255: {pr['v255'] / n:.1%}  "
            f"1..254: {pr['valid'] / n:.1%}"
            + (f" (median {pr['median_valid']:.0f}, range "
               f"{pr['min_valid']}-{pr['max_valid']})" if pr['valid'] else "")
            + f"; min_mapq={gate} would drop {(n - surv) / n:.1%} of them "
              f"({n - surv}/{n}; {surv} survive)")


        if pr['verdict'] == 'available' and gate > 0 and (surv == 0 or starved):
            pr['verdict'] = 'unavailable'
            pr['kind'] = 'gate_drops_all' + (f" in {','.join(starved)}" if starved and surv > 0 else '')
    for pr in probes:
        pr.pop('_valid_values', None)
        pr.pop('_member_valid_values', None)
    bad = [pr for pr in probes if pr['verdict'] == 'unavailable']
    policy = {'verdict': 'unavailable' if bad else
              ('available' if any(pr['verdict'] == 'available' for pr in probes)
               else 'no_reads'),
              'auto_fallback': False, 'probe': probes, 'changed': {},
              'kept_user_flags': {}}
    if not bad:
        args._mapq_policy = policy
        args._mapq_policy_resolved = True
        return policy
    say('⚠️ ' + '=' * 70)
    say(f"⚠️ MAPQ UNAVAILABLE in {len(bad)} of {len(probes)} BAM(s):")
    for pr in bad:
        say(f"⚠️   {pr['bam']} ({pr['kind']}): "
            + ("every sampled primary read has MAPQ 0 or 255 -- no read "
               "carries a mapping-quality value; the aligner or single-cell "
               "pipeline that wrote it did not keep one"
               if pr['kind'] != 'gate_drops_all' else
               f"MAPQ values exist ({pr['valid']}/{pr['sampled']} in 1..254) "
               f"but min_mapq={gate} would discard every sampled read, so the "
               f"run would finish with nothing"))

    fallback = {'min_mapq': 0}

    fallback.update({k: v for k, v in _CLF_FREE.items() if hasattr(args, k)})


    owned = set()
    if _flag_given(argv, 'setting') and getattr(args, 'setting', None) in SETTING_OVERLAYS:
        owned = set(SETTING_OVERLAYS[args.setting]) | {'snv_classifier'}
    for k, v in fallback.items():
        if _flag_given(argv, k) or k in owned:
            policy['kept_user_flags'][k] = getattr(args, k)
            say(f"⚠️ [mapq] {k}={getattr(args, k)!r} kept (user override"
                f"{' via --setting' if k in owned and not _flag_given(argv, k) else ''}; "
                f"the automatic fallback would have set {v!r})")
            continue
        old = getattr(args, k)
        setattr(args, k, v)
        policy['changed'][k] = (old, v)
        say(f"⚠️ [mapq] {k}={v!r} (AUTOMATIC: MAPQ unavailable; "
            + (f"was {old!r})" if old != v else "already so, now decided)"))

    if 'snv_classifier' in policy['changed'] and hasattr(args, 'setting') \
            and not _flag_given(argv, 'setting'):
        old = getattr(args, 'setting')
        if old is None and getattr(args, 'platform', None):

            old = 'clf (platform default)' if policy['changed']['snv_classifier'][0] else 'clf-free (platform default)'
        args.setting = 'clf-free'
        policy['changed']['setting'] = (old, 'clf-free')
        say(f"⚠️ [mapq] setting='clf-free' (AUTOMATIC; was {old!r})")
    policy['auto_fallback'] = bool(policy['changed'])
    if policy['kept_user_flags'].get('snv_classifier'):
        say("⚠️ [mapq] the SNV classifier is still ON by your flag, and one of "
            "its 17 inputs (mean_mapq) is now a constant it was never trained "
            "on -- its scores are not trustworthy on this BAM")
    if policy['kept_user_flags'].get('min_mapq', 0) > 0 and \
            all(pr['survivors_at_gate'] == 0 for pr in bad):
        say(f"⚠️ [mapq] min_mapq={policy['kept_user_flags']['min_mapq']} is kept "
            f"by your flag and it discards EVERY sampled read: this run will "
            f"finish with no candidates")
    say("⚠️ [mapq] with the MAPQ gate off, multi-mapping / low-confidence "
        "alignments enter variant calling unfiltered -- report this as a "
        "limitation of the run (the fallback's price)")
    say(f"⚠️ [mapq] this decision is written into the run's parameter record "
        f"(mapq_fallback={policy['auto_fallback']}, mapq_verdict=unavailable)")
    say('⚠️ ' + '=' * 70)
    args._mapq_policy = policy
    args._mapq_policy_resolved = True
    return policy


def mapq_record_fields(policy):
    if policy is None:
        return {}
    return {'mapq_verdict': policy['verdict'],
            'mapq_fallback': bool(policy['auto_fallback'])}


def path_safe_gene_name(geneName):
    if geneName is None:
        return None
    return str(geneName).replace('/', '_').replace('\\', '_')


def base_is_at_read_end(qpos, read_len, min_dist):
    if min_dist <= 0 or not read_len or read_len <= 0:
        return False
    return min(qpos, read_len - 1 - qpos) < min_dist


_WORKER_ASSET_CACHE = {}


def _worker_cached_asset(kind, path, digest, loader):
    key = (kind, path, digest)
    if key not in _WORKER_ASSET_CACHE:
        _WORKER_ASSET_CACHE[key] = loader(path, digest)
    return _WORKER_ASSET_CACHE[key]


RETIRED_ENTRY_POINTS = {
    'variant_call.py': {'initial call': 'step1', 'generate input': 'step2'},
    'haplotyping.py': {'haplotyping': 'step3', 'summary': 'step4'},
}


def refuse_retired_entry_point(script, task=None, argv=None, stream=None):
    import sys as _sys
    stream = _sys.stderr if stream is None else stream
    tasks = RETIRED_ENTRY_POINTS[script]
    step = tasks.get(task)
    argv = list(_sys.argv[1:] if argv is None else argv)

    rest = []
    skip = False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a == '--task':
            skip = True
            continue
        if a.startswith('--task='):
            continue
        rest.append(a)
    lines = [
        f'{script} is retired. Run the pipeline through longallele.py, which '
        f'is the only entry point we run and the only one that receives new '
        f'options.',
        '',
    ]
    if step:
        cmd = ' '.join(['python src/longallele.py', '--task', step] + rest)
        lines += ['The same run, unchanged apart from the entry point:', '',
                  '  ' + cmd, '']
    else:
        opts = ' | '.join(f'--task {v}  (was --task {k!r})'
                          for k, v in tasks.items())
        lines += [f'Equivalent tasks: {opts}', '']
    lines += [
        'Why: this wrapper kept a second copy of the command line, so options '
        'added to longallele.py were silently missing here — including the '
        'heterozygous model the shipped classifier requires. A run could look '
        'normal and be using a different filter than the one we report.',
    ]
    print('\n'.join(lines), file=stream)
    raise SystemExit(2)


GATE_CONFIG_FILE = 'gate_config.json'


def _read_gate_config(path, tries=5, delay=0.05):
    import json, time
    last = None
    for _ in range(tries):
        try:
            with open(path) as fh:
                txt = fh.read()
            if txt.strip():
                return json.loads(txt)
        except (OSError, ValueError) as exc:
            last = exc
        time.sleep(delay)
    if last is not None:
        raise last
    raise ValueError(f'{path} is empty')


def _step3_config_path(output_folder, prefix):
    return os.path.join(output_folder, f"step3_config_{prefix or 'default'}.json")


def ensure_step3_config(output_folder, prefix, params, logger=None,
                        cover_existing=False, claim_wait=5.0):
    import json
    path = _step3_config_path(output_folder, prefix)
    say = lambda m: _log_with_fallback(logger, m)
    if not os.path.exists(path):
        os.makedirs(output_folder, exist_ok=True)
        tmp = path + f'.tmp.{os.getpid()}'
        with open(tmp, 'w') as fh:
            json.dump(params, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
        say(f'[step3] parameter record written: {path}')
        return params
    rec = _read_gate_config(path)
    diff = {k: (rec.get(k), v) for k, v in params.items() if rec.get(k) != v}


    old_mapq = [k for k in MAPQ_RECORD_KEYS if k in diff and k not in rec]
    if old_mapq:
        for k in old_mapq:
            diff.pop(k)
        say(f'[step3] ⚠️ the parameter record at {path} predates the MAPQ '
            f'decision ({old_mapq}); this launch has '
            f'{ {k: params[k] for k in old_mapq} } and cannot verify the '
            f'earlier launches agreed')
    if not diff:
        if old_mapq:


            import time
            claim = path + '.mapq_claim'
            try:
                fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                deadline = time.monotonic() + float(claim_wait)
                while True:
                    rec = _read_gate_config(path)
                    if all(k in rec for k in old_mapq):
                        break
                    if time.monotonic() >= deadline:
                        raise SystemExit(
                            f'⛔ step3: another launch holds {claim} but has not '
                            f'written the MAPQ decision into {path} within '
                            f'{claim_wait}s; not resuming blind. Remove the '
                            f'claim file if that launch is dead.')
                    time.sleep(0.05)
                late = {k: (rec[k], params[k]) for k in old_mapq
                        if rec[k] != params[k]}
                if late:
                    lines = '\n'.join(f'  {k}: recorded={a!r}  this run={b!r}'
                                      for k, (a, b) in late.items())
                    raise SystemExit(
                        f'⛔ step3 refuses to resume into {output_folder} (prefix '
                        f'{prefix!r}): a concurrent launch just established the '
                        f'MAPQ decision differently —\n{lines}')
                say(f'[step3] parameter record established by a concurrent '
                    f'launch, matches: {path} (resume allowed)')
                return rec
            with os.fdopen(fd, 'w') as fh:
                fh.write(f'{os.getpid()}\n')


            fresh = _read_gate_config(path)
            if fresh != rec:
                raise SystemExit(
                    f'⛔ step3 refuses to resume into {output_folder} (prefix '
                    f'{prefix!r}): {path} changed underneath this launch while it '
                    f'was establishing the MAPQ decision (another launch, '
                    f'probably --cover_existing, rewrote it). Re-launch.')
            rec = dict(rec, **{k: params[k] for k in old_mapq})
            tmp = path + f'.tmp.{os.getpid()}'
            with open(tmp, 'w') as fh:
                json.dump(rec, fh, indent=1, sort_keys=True)
            os.replace(tmp, path)
            say(f'[step3] parameter record augmented with {old_mapq}: {path}')
        say(f'[step3] parameter record matches: {path} (resume allowed)')
        return rec
    lines = '\n'.join(f'  {k}: recorded={a!r}  this run={b!r}' for k, (a, b) in diff.items())
    if cover_existing:
        say(f'[step3] --cover_existing: replacing the parameter record at {path}; '
            f'differences:\n{lines}')
        tmp = path + f'.tmp.{os.getpid()}'
        with open(tmp, 'w') as fh:
            json.dump(params, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)


        try:
            os.remove(path + '.mapq_claim')
        except OSError:
            pass
        return params
    raise SystemExit(
        f'⛔ step3 refuses to resume into {output_folder} (prefix {prefix!r}): the '
        f'products there were made with DIFFERENT parameters —\n{lines}\n'
        f'Resuming would skip every gene already present and hand back a mixture '
        f'of two runs with exit code 0. Either clear this prefix\'s step3 outputs '
        f'(and {os.path.basename(path)}) or pass --cover_existing to redo the '
        f'whole set.')


_GATE_UNSET = object()


def _gate_cfg(n_alt, depth, min_alt_frac, het_beta=_GATE_UNSET,
              het_prob_step1=_GATE_UNSET, setting=_GATE_UNSET,
              mapq_policy=_GATE_UNSET):
    cfg = {'n_alt_count': int(n_alt), 'depth': int(depth),
           'min_alt_frac': float(min_alt_frac)}
    if het_beta is not _GATE_UNSET:
        if het_beta is None:
            cfg['het_beta'] = 'fixed'
        elif isinstance(het_beta, (tuple, list)):


            cfg['het_beta'] = ','.join(f'{float(x)!r}' for x in het_beta)
        else:
            cfg['het_beta'] = ','.join(
                f'{float(x)!r}' for x in str(het_beta).split(','))
    if het_prob_step1 is not _GATE_UNSET and het_prob_step1 is not None:
        cfg['het_prob_step1'] = float(het_prob_step1)
    if setting is not _GATE_UNSET and setting is not None:


        cfg['setting'] = str(setting)
    if mapq_policy is not _GATE_UNSET and mapq_policy is not None:


        cfg.update(mapq_record_fields(mapq_policy))
    return cfg


def _gate_diff(recorded, wanted):
    return {k: (recorded[k], v) for k, v in wanted.items()
            if k in recorded and recorded[k] != v and k != 'setting'}


def _gate_het_unknown(rec, cur, say):
    missing = [k for k in ('het_beta', 'het_prob_step1')
               if k in cur and k not in rec]
    if missing:
        say(f'⚠️ the gate record does not say which heterozygous model step1 '
            f'used ({missing}) — it predates that being recorded. This step '
            f'uses het_beta={cur.get("het_beta")!r}, '
            f'het_prob_step1={cur.get("het_prob_step1")!r}. If step1 ran a '
            f'different one, the sites its model discarded are already gone '
            f'and nothing here can bring them back.')


def ensure_gate_config(output_folder, n_alt, depth, min_alt_frac,
                       logger=None, strict=True,
                       het_beta=_GATE_UNSET, het_prob_step1=_GATE_UNSET,
                       setting=_GATE_UNSET, mapq_policy=_GATE_UNSET):
    import json
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, GATE_CONFIG_FILE)
    cfg = _gate_cfg(n_alt, depth, min_alt_frac, het_beta, het_prob_step1,
                    setting, mapq_policy)
    def _say(m, warn=True):
        if logger is None:
            print(m)
        else:
            (logger.warning if warn else logger.info)(m)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, 'w') as fh:
            json.dump(cfg, fh, sort_keys=True)
        _say(f'candidate gate recorded: {cfg}', warn=False)
        return cfg
    rec = _read_gate_config(path)
    diff = _gate_diff(rec, cfg)
    _gate_het_unknown(rec, cfg, _say)
    if not diff:
        _say(f'candidate gate matches the existing record: {cfg}', warn=False)
        return cfg
    msg = (f'candidate-gate conflict in {output_folder} (recorded -> this '
           f'job): {diff}. Sharded step1 jobs must all use the same gate, '
           'otherwise the candidate set is a mix of two definitions. '
           'Re-run this shard with the recorded values, or start a clean '
           'output folder.')
    if strict:
        raise ValueError(msg)
    _say('WARNING: ' + msg)
    return rec


def write_gate_config(output_folder, n_alt, depth, min_alt_frac, logger=None):
    return ensure_gate_config(output_folder, n_alt, depth, min_alt_frac,
                              logger=logger, strict=True)


def check_gate_config(output_folder, n_alt, depth, min_alt_frac,
                      het_beta=_GATE_UNSET, het_prob_step1=_GATE_UNSET,
                      setting=_GATE_UNSET, logger=None, strict=True,
                      mapq_policy=_GATE_UNSET):
    import json
    path = os.path.join(output_folder, GATE_CONFIG_FILE)
    cur = _gate_cfg(n_alt, depth, min_alt_frac, het_beta, het_prob_step1,
                    setting, mapq_policy)
    def _say(m, warn=True):
        if logger is None:
            print(m)
        else:
            (logger.warning if warn else logger.info)(m)
    if not os.path.exists(path):
        _say(f'no {GATE_CONFIG_FILE} in {output_folder} (step1 predates the '
             f'8-28 gate record) — cannot verify step1/step3 gate agreement; '
             f'make sure --n_alt_count/--depth/--min_alt_frac match step1')
        return None
    try:
        rec = _read_gate_config(path)
    except (OSError, ValueError) as exc:
        msg = (f'{path} exists but is unreadable/corrupt ({exc}); the step1 '
               'gate cannot be verified')
        if strict:
            raise ValueError(msg)
        _say('WARNING: ' + msg)
        return None
    diff = _gate_diff(rec, cur)
    _gate_het_unknown(rec, cur, _say)
    if not diff:
        _say(f'gate config matches step1: {cur}', warn=False)
        return True
    msg = ('candidate-gate mismatch between step1 and this step '
           f'(step1 -> now): {diff}. The two steps re-apply the same gate, '
           'so a mismatch silently re-admits or re-filters candidates. '
           'Re-run with step1\'s values, or re-run step1.')
    if strict:
        raise ValueError(msg)
    _say('WARNING: ' + msg)
    return False


def validate_min_alt_frac(v):
    v = float(v)
    if not np.isfinite(v) or not (0.0 <= v <= 1.0):
        raise ValueError(f'min_alt_frac must be a finite fraction in [0, 1], '
                         f'got {v!r}')
    return v


def check_gate_consistency(min_alt_frac, fast_pileup_min_frac, logger=None,
                           fast_pileup_raw_frac=None):
    if not min_alt_frac:
        return True
    def _warn(m):
        print(m) if logger is None else logger.warning(m)
    ok = True

    if fast_pileup_raw_frac:


        if fast_pileup_raw_frac > min_alt_frac:
            _warn(f'WARNING: --fast_pileup_raw_frac {fast_pileup_raw_frac} '
                  f'exceeds --min_alt_frac {min_alt_frac}; the RAW prescreen '
                  f'(non-whitelisted counts, fraction diluted by off-gene '
                  f'reads) may drop sites the AF gate would keep. Lower it.')
            ok = False
    elif fast_pileup_min_frac and fast_pileup_min_frac > min_alt_frac:

        _warn(f'WARNING: --fast_pileup_min_frac {fast_pileup_min_frac} is '
              f'STRICTER than --min_alt_frac {min_alt_frac}; the prescreen '
              f'would drop callable sites. Lower the prescreen gate.')
        ok = False
    return ok


EM_MISSING_CODE = -1
EM_REF_CODE = 0
EM_ALT_CODE = 1
EM_OTHER_CODE = 2

class _PhasabilityPartitionMismatch(ValueError):
    pass


SNV_CLF_FEATURE_COLUMNS = [
    "depth", "alt_count", "het_prob",
    "mean_mapq", "mean_bq_alt", "mean_bq_ref",
    "n_distinct_alt", "alt_pos_on_read_mean", "alt_pos_on_read_std",
    "strand_sor", "del_frac",
    "gc_content_11bp", "homopolymer_len", "is_homopolymer_ge5",
    "creates_homopolymer", "flanking_is_AT", "is_transition",
]


def _coerce_read_snv_codes(df_read_snv):
    r = df_read_snv.to_numpy()
    if np.issubdtype(r.dtype, np.integer):
        arr = r.astype(np.int8, copy=False)
    else:
        arr = np.full(r.shape, EM_MISSING_CODE, dtype=np.int8)
        arr[r == EM_REF_CODE] = EM_REF_CODE
        arr[r == EM_ALT_CODE] = EM_ALT_CODE
        arr[r == EM_OTHER_CODE] = EM_OTHER_CODE
        arr[r == 'ref'] = EM_REF_CODE
        arr[r == 'alt'] = EM_ALT_CODE
        arr[r == 'other'] = EM_OTHER_CODE
    return pd.DataFrame(arr, index=df_read_snv.index, columns=df_read_snv.columns, dtype=np.int8)


def _save_em_input_npz(path, r_code, pi_arr, read_names, column_names):
    dirpath = os.path.dirname(path) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp.npz")
    os.close(tmp_fd)
    try:
        np.savez_compressed(
            tmp_path,
            r_code=np.asarray(r_code, dtype=np.int8),
            pi_arr=np.asarray(pi_arr, dtype=np.float32),
            read_names=np.asarray(read_names, dtype=str),
            column_names=np.asarray(column_names, dtype=str),
        )
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _atomic_to_csv(df, path, **kwargs):
    dirpath = os.path.dirname(path) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp.csv")
    os.close(tmp_fd)
    try:
        df.to_csv(tmp_path, **kwargs)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _atomic_pickle_dump(obj, path):
    dirpath = os.path.dirname(path) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp.pkl")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _load_em_input(npz_path, read_snv_csv_path, read_pi_csv_path):
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            read_names = data['read_names'].astype(str)
            column_names = data['column_names'].astype(str)
            df_read_snv = pd.DataFrame(data['r_code'].astype(np.int8), index=read_names, columns=column_names)
            df_read_pi = pd.DataFrame(data['pi_arr'].astype(np.float32), index=read_names, columns=column_names)
        return df_read_snv, df_read_pi
    df_read_snv = pd.read_csv(read_snv_csv_path, index_col=0, low_memory=False)
    df_read_pi = pd.read_csv(read_pi_csv_path, index_col=0, low_memory=False)
    df_read_snv = _coerce_read_snv_codes(df_read_snv)
    df_read_pi = df_read_pi.astype(np.float32)
    return df_read_snv, df_read_pi


def _order_genes_heavy_first(gene_ids, em_dir):
    def weight(gene_id):
        total = 0
        for suffix in ('_read_matrices.npz', '_read_pi.csv', '_read_snv.csv'):
            try:
                total += os.path.getsize(os.path.join(em_dir, gene_id + suffix))
            except OSError:
                pass
        return total


    return sorted(gene_ids, key=lambda g: (-weight(g), g))


def _list_gene_ids_from_em_input(directory):
    gene_ids = set()
    for filename in os.listdir(directory):
        if filename.endswith('_read_matrices.npz'):
            gene_ids.add(filename.rsplit('_read_matrices.npz', 1)[0])
        elif filename.endswith('_read_pi.csv'):
            gene_ids.add(filename.rsplit('_read_pi.csv', 1)[0])
    return sorted(gene_ids)


class VariantCaller:
    def __init__(self, scotch_target:Union[str, Sequence[str]], bam_path:Union[str, Sequence[str]],
                 ref_fasta_path = None, ref_pickle_path = None, target = None, sample_names = None,
                 n_jobs = 1, samtools_threads = 1, job_index = 0, depth = 20, n_alt = 10,
                 min_mapq = 20, min_baseq = 5, min_dist_to_end = 3,
                 sample_name_parse = None,
                 gene_subset = None, het_prefilter_threshold = 0.8 / 0.99,
                 het_beta = None,
                 given_snv = None,
                 fast_pileup = None, fast_pileup_min_frac = 0.05,
                 fast_pileup_raw_frac = 0.02,
                 pileup_engine = 'walk',
                 marker_span_from_reads = False,
                 min_alt_frac = 0.0,
                 max_baseq = None,
                 gene_guard_depth = None,
                 n_workers = 1,
                 logger = None):
        self.logger = logger


        self.min_alt_frac = validate_min_alt_frac(min_alt_frac)
        self.n_workers = n_workers

        self.sample_name_parse = sample_name_parse
        self.bam_path = bam_path
        self.target = target
        self.scotch_target = self._ensure_list(scotch_target if scotch_target is not None else target)
        self.sample_names = [os.path.basename(st) if sample_names is None else sample_names for st in self.scotch_target]
        self.n_samples = len(self.scotch_target)
        self.bam_path = self._ensure_list(bam_path)
        self.read_isoform_mapping_path = [resolve_scotch_auxiliary_tsv(st, self.sample_name_parse) for st in self.scotch_target]
        self.variant_align_folder_path1 = os.path.join(self.target, "variant_align1")
        self.bam_by_gene_folder_1 = os.path.join(self.variant_align_folder_path1, 'bam_by_gene')
        self.reads_by_gene_folder_1 = os.path.join(self.variant_align_folder_path1, 'reads_by_gene')
        self.variants_by_gene_folder_1 = os.path.join(self.variant_align_folder_path1, 'variants_by_gene')
        self.em_input = os.path.join(self.target, 'em_input')
        self.ref_pickle_path = ref_pickle_path
        if ref_pickle_path is not None:
            self.geneStructureInformation = _load_gene_structure_information(ref_pickle_path, self.logger)
        else:
            gsi_path = _resolve_reference_pickle_path(self.scotch_target, self.logger)
            self.geneStructureInformation = _load_gene_structure_information(gsi_path, self.logger)
        self.ref_fasta_path = ref_fasta_path
        self.ref_fasta = pysam.FastaFile(self.ref_fasta_path)
        self.n_jobs = n_jobs
        self.samtools_threads = samtools_threads
        self.job_index = job_index
        self.depth = depth


        self.gene_guard_depth = (int(gene_guard_depth)
                                 if gene_guard_depth is not None else depth)


        self.max_baseq = int(max_baseq) if max_baseq is not None else None
        self.n_alt = n_alt
        self.min_mapq = min_mapq
        self.min_baseq = min_baseq
        self.min_dist_to_end = min_dist_to_end
        self.gene_subset = set(gene_subset) if gene_subset is not None else None


        self.given_snv = given_snv
        self.het_prefilter_threshold = het_prefilter_threshold


        self.het_beta = het_beta


        self.fast_pileup_min_frac = float(fast_pileup_min_frac)
        if not 0.0 <= self.fast_pileup_min_frac <= 1.0:
            raise ValueError(f'fast_pileup_min_frac must be in [0, 1], '
                             f'got {self.fast_pileup_min_frac}')


        self.fast_pileup_raw_frac = float(fast_pileup_raw_frac)
        if not 0.0 <= self.fast_pileup_raw_frac <= 1.0:
            raise ValueError(f'fast_pileup_raw_frac must be in [0, 1], '
                             f'got {self.fast_pileup_raw_frac}')


        if pileup_engine not in ('column', 'walk'):
            raise ValueError(f"pileup_engine must be 'column' or 'walk', "
                             f"got {pileup_engine!r}")
        self.pileup_engine = pileup_engine
        if fast_pileup is None:
            self.fast_pileup = self.fast_pileup_min_frac > 0
        else:
            self.fast_pileup = bool(fast_pileup) and \
                self.fast_pileup_min_frac > 0


        self.marker_span_from_reads = bool(marker_span_from_reads)
        if self.marker_span_from_reads and not self.fast_pileup:
            raise ValueError('--marker_span_from_reads needs the fast (prescreen) '
                             'step1 path; it is not implemented on the legacy '
                             'full-interval scan')
        self._bam_path_cache = {}


        self.gene_index = None
        self._geneidx_colpos = None
    @staticmethod
    def _ensure_list(x):
        if isinstance(x, str):
            return [x]
        if isinstance(x, Iterable):
            return list(x)
        raise TypeError("Expected str or iterable of str")
    @staticmethod
    def heterozygous_prob(depth, n_alt, e=0.01):
        priors = (1 / 3, 1 / 3, 1 / 3)
        log_binom = math.lgamma(depth + 1) - math.lgamma(n_alt + 1) - math.lgamma(depth - n_alt + 1)
        log_p_het = log_binom + depth * math.log(0.5)
        log_p_hom_ref = log_binom + n_alt * math.log(e) + (depth - n_alt) * math.log(1 - e)
        log_p_hom_alt = log_binom + n_alt * math.log(1 - e) + (depth - n_alt) * math.log(e)
        log_post = [math.log(priors[0]) + log_p_het, math.log(priors[1]) + log_p_hom_ref, math.log(priors[2]) + log_p_hom_alt]
        m = max(log_post)
        den = sum(math.exp(x - m) for x in log_post)
        return math.exp(log_post[0] - m) / den
    @staticmethod
    def heterozygous_prob_vec(depth, n_alt, e=0.01, priors=None, het_beta=None):
        if priors is None:
            priors = (1 / 3, 1 / 3, 1 / 3)
        depth = np.asarray(depth, dtype=np.int64)
        n_alt = np.asarray(n_alt, dtype=np.int64)
        log_binom = gammaln(depth + 1) - gammaln(n_alt + 1) - gammaln(depth - n_alt + 1)
        if het_beta is None:
            log_p_het = log_binom + depth * np.log(0.5)
        else:
            _a, _b = float(het_beta[0]), float(het_beta[1])
            log_p_het = (log_binom
                         + betaln(n_alt + _a, depth - n_alt + _b)
                         - betaln(_a, _b))
        log_p_hom_ref = log_binom + n_alt * np.log(e) + (depth - n_alt) * np.log(1 - e)
        log_p_hom_alt = log_binom + n_alt * np.log(1 - e) + (depth - n_alt) * np.log(e)
        log_post = np.stack([
            np.log(priors[0]) + log_p_het,
            np.log(priors[1]) + log_p_hom_ref,
            np.log(priors[2]) + log_p_hom_alt,
        ])
        m = np.max(log_post, axis=0)
        den = np.exp(log_post - m).sum(axis=0)
        return np.exp(log_post[0] - m) / den
    @staticmethod
    def _extract_gene_id_from_isoform_filename(filename):
        match = re.search(r'_(ENSG[^_]+)_(?:isoform_agg(?:_balance|_unbalance|_extrap|_pmax)?)\.csv$', os.path.basename(filename))
        if match is None:
            raise ValueError(f'Could not extract geneID from filename: {filename}')
        return match.group(1)
    def _read_mapping(self):
        mapping_df_list = []
        for read_isoform_mapping_path in self.read_isoform_mapping_path:
            pieces = defaultdict(list)
            for chunk in pd.read_csv(read_isoform_mapping_path, sep='\t', chunksize=100000):
                chunk = chunk[chunk['Keep'] == 1][['Read', 'geneName', 'geneID', 'geneChr', 'Cell', 'Umi']]
                chunk['Read'] = chunk['Read'].map(canonicalize_read_name)
                for gene_id, sub_df in chunk.groupby('geneID', sort=False):
                    pieces[gene_id].append(sub_df)
                del chunk
            mapping_df_dict = {
                gene_id: pd.concat(parts, ignore_index=True)
                for gene_id, parts in pieces.items()
            }
            mapping_df_list.append(mapping_df_dict)
            del pieces
        return mapping_df_list
    def _load_gene_index(self):
        gene_index_list = []
        colpos_list = []
        for tsv_path in self.read_isoform_mapping_path:
            idx_path = tsv_path + '.geneidx.tsv'
            if not os.path.exists(idx_path):
                return None, None
            idx = {}
            with open(idx_path) as f:
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 4:
                        continue
                    gid, off, length, nreads = parts[0], parts[1], parts[2], parts[3]
                    idx[gid] = (int(off), int(length), int(nreads))
            with open(tsv_path) as f:
                header = f.readline().rstrip('\r\n').split('\t')
            cols = {name: i for i, name in enumerate(header)}
            if 'Read' not in cols or 'Keep' not in cols:
                return None, None
            colpos_list.append((cols['Read'], cols['Keep']))
            gene_index_list.append(idx)
        return gene_index_list, colpos_list
    def _gene_reads_ordered(self, sample_index, geneID):
        idx = self.gene_index[sample_index].get(geneID)
        if idx is None:
            return []
        offset, length, _ = idx
        read_i, keep_i = self._geneidx_colpos[sample_index]
        ncols = max(read_i, keep_i) + 1
        with open(self.read_isoform_mapping_path[sample_index], 'rb') as f:
            f.seek(offset)
            block = f.read(length)
        ordered = {}
        for line in block.split(b'\n'):
            if not line:
                continue
            fields = line.split(b'\t')
            if len(fields) < ncols:
                raise ValueError(
                    f'malformed mapping block for {geneID} in '
                    f'{self.read_isoform_mapping_path[sample_index]}: expected '
                    f'>= {ncols} columns, got {len(fields)}')
            if _keep_is_one_bytes(fields[keep_i]):
                ordered.setdefault(
                    canonicalize_read_name(fields[read_i].decode()), None)
        return list(ordered)

    def _reads_for_gene_block(self, sample_index, geneID):
        idx = self.gene_index[sample_index].get(geneID)
        if idx is None:
            return set()
        offset, length, _ = idx
        read_i, keep_i = self._geneidx_colpos[sample_index]
        ncols = max(read_i, keep_i) + 1
        with open(self.read_isoform_mapping_path[sample_index], 'rb') as f:
            f.seek(offset)
            block = f.read(length)
        reads = set()
        for line in block.split(b'\n'):
            if not line:
                continue
            fields = line.split(b'\t')
            if len(fields) < ncols:


                raise ValueError(
                    f'malformed mapping block for {geneID} in '
                    f'{self.read_isoform_mapping_path[sample_index]}: expected '
                    f'>= {ncols} columns, got {len(fields)}')
            if _keep_is_one_bytes(fields[keep_i]):
                reads.add(canonicalize_read_name(fields[read_i].decode()))
        return reads
    def _get_bam_file_path(self, bam_path, chrom=None):

        return resolve_bam_file(bam_path, chrom=chrom, cache=self._bam_path_cache)

    def _note_unresolvable_contig(self, step, sample_index, contig, what):
        seen = getattr(self, '_unresolvable_contigs', None)
        if seen is None:
            seen = self._unresolvable_contigs = set()
        key = (sample_index, str(contig))
        if key in seen:
            return
        seen.add(key)
        _log_with_fallback(self.logger,
                           f'⚠️ [{step}] sample {sample_index}: no BAM for contig {contig} in the '
                           f'per-chromosome directory -- skipping {what} (first: this line is '
                           f'printed once per contig; only chr1-22/X/Y/M are resolved from a '
                           f'directory)')
    def _read_bam_single(self, bam_path, chrom = None):
        bam_file = self._get_bam_file_path(bam_path, chrom=chrom)
        bamFilePysam = pysam.Samfile(bam_file, "rb")
        return bamFilePysam
    def _read_bam(self, chrom=None, bam_path: list = None):
        bam_path_list = self.bam_path if bam_path is None else bam_path
        bamFilePysam_list = []
        for bp in bam_path_list:
            bamFilePysam = self._read_bam_single(bp, chrom=chrom)
            bamFilePysam_list.append(bamFilePysam)
        return bamFilePysam_list
    def _af_gate_ok(self, df):
        if self.min_alt_frac <= 0:
            return pd.Series(True, index=df.index)
        return df['alt_count'] >= np.ceil(self.min_alt_frac * df['depth'])

    @staticmethod
    def _parse_pileup_bases(bases: str):
        i, n = 0, len(bases)
        ref_count = 0
        alt_count = 0
        while i < n:
            c = bases[i]
            if c == '^':
                i += 2
                continue
            if c == '$':
                i += 1
                continue
            if c in '+-':
                i += 1
                j = i
                while j < n and bases[j].isdigit():
                    j += 1
                if j == i:
                    i += 1
                    continue
                length = int(bases[i:j])
                i = j + length
                continue
            if c == '*':
                i += 1
                continue
            if c in '.,':
                ref_count += 1
                i += 1
                continue
            if c in 'ACGTNacgtn':
                alt_count += 1
                i += 1
                continue

            i += 1
        return ref_count, alt_count
    def _samtools_snv_counts(self, bam_path_gene, ref_fasta, chrom, start, end):
        region_str = f"{chrom}:{start + 1}-{end}"
        cmd = ["samtools", "mpileup", "-f", ref_fasta, "-q", "0", "-Q", "0", "-r", region_str, bam_path_gene]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        rows = []
        for line in proc.stdout:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 5:
                continue
            chrom, pos1, ref, depth_raw, bases = parts[0], int(parts[1]), parts[2].upper(), int(parts[3]), parts[4]
            ref_count, alt_count = self._parse_pileup_bases(bases)
            eff_depth = ref_count + alt_count
            if eff_depth == 0 or alt_count == 0:
                continue
            alt_frac = alt_count / eff_depth
            rows.append((chrom, pos1 - 1, ref, eff_depth, alt_count, alt_frac))
        proc.wait()
        df = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "depth", "alt_count", "alt_frac"])
        return df
    def _reads_set_for_gene(self, geneID):

        geneInfo, _, _ = self.geneStructureInformation[geneID]
        reads_set_list = []
        if self.gene_index is not None:


            for sample_index in range(self.n_samples):
                reads_set_list.append(self._reads_for_gene_block(sample_index, geneID))
        else:
            for mapping_df_dict in self.mapping_df_list:
                mapping_df_gene = mapping_df_dict.get(geneID)
                if mapping_df_gene is not None:
                    reads_set = set(mapping_df_gene['Read'].tolist())
                else:
                    reads_set = set()
                reads_set_list.append(reads_set)
        return reads_set_list, geneInfo
    def _write_readnames_file(self, geneID):
        reads_set_list, geneInfo = self._reads_set_for_gene(geneID)
        if all(len(s) == 0 for s in reads_set_list):
            readnames_txt_list = None
        else:
            readnames_txt_list = []
            for i in range(self.n_samples):
                readnames_txt = os.path.join(self.reads_by_gene_folder_1, f"{geneID}_readnames_{i}.txt")
                readnames_txt_list.append(readnames_txt)
                os.makedirs(os.path.dirname(readnames_txt), exist_ok=True)
                with open(readnames_txt, "w") as f:
                    for r in reads_set_list[i]:
                        f.write(r + "\n")
        return readnames_txt_list, geneInfo
    @staticmethod
    def merge_bams(bam_list, bam_out):
        if len(bam_list) == 0:
            out = 0
        elif len(bam_list) == 1:
            src = bam_list[0]
            os.replace(src, bam_out)
            if os.path.exists(src + ".bai"):
                os.replace(src + ".bai", bam_out + ".bai")
            elif not os.path.exists(bam_out + ".bai"):
                subprocess.run(["samtools", "index", bam_out], check=True)
            out = 1
        else:
            subprocess.run(["samtools", "merge", "-f", "-o", bam_out, *bam_list], check=True)
            subprocess.run(["samtools", "index", bam_out], check=True)
            out = 1
        for b in bam_list:
            if os.path.exists(b): os.remove(b)
            if os.path.exists(b + ".bai"): os.remove(b + ".bai")
        return out

    def _subset_bam_by_gene_single_file(self, geneID, geneInfo, readnames_txt, bam_in, bam_out):
        if readnames_txt is None:
            return None
        bam_path = self._get_bam_file_path(bam_in, chrom=geneInfo['geneChr'])
        region = f"{geneInfo['geneChr']}:{geneInfo['geneStart']}-{geneInfo['geneEnd']}"
        cmd = ["samtools", "view", "-h", "-b", "-@", str(self.samtools_threads), "-N", readnames_txt, bam_path, region]
        print(f"get bam file for gene {geneID}")
        with open(bam_out, "wb") as outfh:
            subprocess.run(cmd, check=True, stdout=outfh)
        return bam_out

    def _subset_bam_by_gene(self, geneID, bam_out):
        readnames_txt_list, geneInfo = self._write_readnames_file(geneID)
        if readnames_txt_list is None:
            return 0
        bam_out_list = []
        for i, readnames_txt in enumerate(readnames_txt_list):
            out = self._subset_bam_by_gene_single_file(geneID, geneInfo, readnames_txt_list[i], self.bam_path[i],
                                                       bam_out.replace('.bam', f'_{i}.bam'))
            if out is not None:
                bam_out_list.append(out)
        success = self.merge_bams(bam_out_list, bam_out)
        return success
    def _prescreen_positions(self, bam, chrom, start, end, ref_seq, fetch_start,
                             allowed_reads=None, min_frac=None):
        min_mapq = self.min_mapq
        canon_cache = {}

        def _keep(r):
            if (r.is_unmapped or r.is_secondary or r.is_duplicate
                    or r.is_qcfail or r.is_supplementary
                    or r.mapping_quality < min_mapq):
                return False
            if allowed_reads is None:
                return True
            qn = r.query_name
            cn = canon_cache.get(qn)
            if cn is None:
                cn = canonicalize_read_name(qn)
                canon_cache[qn] = cn
            return cn in allowed_reads

        cov = bam.count_coverage(chrom, start, end + 1,
                                 quality_threshold=self.min_baseq,
                                 read_callback=_keep)
        acgt = np.array(cov, dtype=np.int64)
        total = acgt.sum(axis=0)
        base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        L = acgt.shape[1]
        ref_slice = ref_seq[start - fetch_start:start - fetch_start + L]
        ref_rows = np.array([base_idx.get(b, -1) for b in ref_slice],
                            dtype=np.int64)
        ref_counts = np.where(ref_rows >= 0,
                              acgt[np.maximum(ref_rows, 0),
                                   np.arange(L)], 0)
        alt = total - ref_counts
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(total > 0, alt / np.maximum(total, 1), 0.0)
        mask = (total >= self.depth) & (alt > self.n_alt) & \
               (frac >= (self.fast_pileup_min_frac if min_frac is None
                         else float(min_frac))) & (ref_rows >= 0)


        if self.min_alt_frac > 0 and allowed_reads is not None:
            mask &= alt >= np.ceil(self.min_alt_frac * total)
        return [int(p) for p in (np.nonzero(mask)[0] + start)]

    @staticmethod
    def _pileup_cols_at(bam, chrom, positions, **kw):
        kw.setdefault('max_depth', 1_000_000)
        kw.setdefault('flag_filter', _PILEUP_FLAG_FILTER)
        for pos in positions:
            for col in bam.pileup(chrom, pos, pos + 1, truncate=True, **kw):
                yield col

    def _sweep_walk(self, bam, geneChr, geneStart, geneEnd, prescreened,
                    ref_seq, fetch_start, allowed_reads, canon_cache):
        cand = np.asarray(prescreened, dtype=np.int64)
        nC = cand.size
        ref_c = np.zeros(nC, dtype=np.int64)
        alt_c = np.zeros(nC, dtype=np.int64)
        dels = np.zeros(nC, dtype=np.int64)
        tuples = [set() for _ in range(nC)]
        ref_bases = [ref_seq[int(p) - fetch_start] for p in cand]
        n_seen = n_match = 0
        min_baseq = self.min_baseq
        min_dist = self.min_dist_to_end
        ACGT = {"A", "C", "G", "T"}
        for aln in bam.fetch(geneChr, geneStart, geneEnd + 1):
            if (aln.is_unmapped or aln.is_secondary or aln.is_duplicate
                    or aln.is_qcfail or aln.is_supplementary):
                continue
            if aln.is_paired:


                raise ValueError(
                    'pileup_engine=walk supports unpaired (long-read) BAMs '
                    'only: paired record encountered '
                    f'({aln.query_name!r}) — rerun with the default '
                    'column engine')
            if aln.mapping_quality < self.min_mapq:
                continue
            qn = aln.query_name
            cn = canon_cache.get(qn)
            if cn is None:
                cn = canonicalize_read_name(qn)
                canon_cache[qn] = cn
            n_seen += 1
            if allowed_reads is not None and cn not in allowed_reads:
                continue
            n_match += 1
            cig = aln.cigartuples
            if not cig:
                continue
            seq = aln.query_sequence
            quals = aln.query_qualities
            mapq = int(aln.mapping_quality)
            is_reverse = 1 if aln.is_reverse else 0
            read_len = aln.query_length or aln.infer_read_length() or 0
            r = aln.reference_start
            q = 0
            for op, n in cig:
                if op in (0, 7, 8):
                    lo = int(np.searchsorted(cand, r, side="left"))
                    hi = int(np.searchsorted(cand, r + n, side="left"))
                    for idx in range(lo, hi):
                        pos = int(cand[idx])
                        qpos = q + (pos - r)
                        if seq is None or qpos >= len(seq):
                            continue
                        base = seq[qpos].upper()
                        if quals is None:
                            baseq = None
                        else:
                            baseq = int(quals[qpos]) if qpos < len(quals) else 0
                            if baseq < min_baseq:
                                continue
                        rb = ref_bases[idx]
                        if base_is_at_read_end(qpos, read_len, min_dist):
                            continue
                        if base in ACGT:
                            if base == rb:
                                ref_c[idx] += 1
                            else:
                                alt_c[idx] += 1
                        tuples[idx].add((cn, qpos, base, baseq, mapq,
                                         read_len, is_reverse))
                    r += n
                    q += n
                elif op in (2, 3):


                    lo = int(np.searchsorted(cand, r, side="left"))
                    hi = int(np.searchsorted(cand, r + n, side="left"))
                    for idx in range(lo, hi):
                        dels[idx] += 1
                    r += n
                elif op in (1, 4):
                    q += n

        counts, sweeps_reads, sweep_dels = {}, {}, {}
        for idx in range(nC):
            eff = int(ref_c[idx] + alt_c[idx])
            if eff == 0 or alt_c[idx] == 0:
                continue
            pos = int(cand[idx])
            counts[pos] = (ref_bases[idx], int(ref_c[idx]), int(alt_c[idx]))
            sweeps_reads[pos] = tuples[idx]
            sweep_dels[pos] = int(dels[idx])
        return counts, sweeps_reads, sweep_dels, n_seen, n_match

    def _read_span_for_gene(self, bam, geneChr, geneStart, geneEnd,
                            allowed_reads, canon_cache):
        lo, hi = int(geneStart), int(geneEnd)
        min_mapq = self.min_mapq
        for aln in bam.fetch(geneChr, geneStart, geneEnd + 1):
            if (aln.is_unmapped or aln.is_secondary or aln.is_duplicate
                    or aln.is_qcfail or aln.is_supplementary
                    or aln.mapping_quality < min_mapq):
                continue
            qn = aln.query_name
            cn = canon_cache.get(qn)
            if cn is None:
                cn = canonicalize_read_name(qn)
                canon_cache[qn] = cn
            if cn not in allowed_reads:
                continue
            if aln.reference_start < lo:
                lo = int(aln.reference_start)
            if aln.reference_end is not None and aln.reference_end - 1 > hi:
                hi = int(aln.reference_end) - 1
        return lo, hi

    def _call_snvs_fast(self, geneID, bam_path_gene, allowed_reads=None,
                        sample_index=0):
        geneInfo, exonInfo, _ = self.geneStructureInformation[geneID]
        geneChr, geneStart, geneEnd = (geneInfo['geneChr'],
                                       geneInfo['geneStart'],
                                       geneInfo['geneEnd'])
        canon_cache = {}
        seq_cache = {}
        n_pileup_seen = 0
        n_allowed_match = 0
        bam = self._read_bam(bam_path=[bam_path_gene])


        call_start, call_end = geneStart, geneEnd
        if self.marker_span_from_reads:
            if allowed_reads is None:
                bam[0].close()
                raise ValueError('marker_span_from_reads needs the per-gene read '
                                 'whitelist (light_upstream); none was given')
            call_start, call_end = self._read_span_for_gene(
                bam[0], geneChr, geneStart, geneEnd, allowed_reads, canon_cache)
        ref_len = self.ref_fasta.get_reference_length(geneChr)
        fetch_start = max(0, call_start)
        fetch_end = min(ref_len, call_end + 1)
        ref_seq = self.ref_fasta.fetch(geneChr, fetch_start, fetch_end).upper()
        _het_prefilter_priors = (0.6, 0.1, 0.3)
        _het_prefilter_e = 0.01
        _het_prefilter_threshold = self.het_prefilter_threshold

        _given_fast = self._given_sites_for(geneID, sample_index, geneChr,
                                            call_start, call_end)
        try:
            if _given_fast is not None:


                prescreened = sorted(_given_fast)
            elif self.fast_pileup_raw_frac > 0:
                prescreened = self._prescreen_positions(
                    bam[0], geneChr, fetch_start, fetch_end - 1, ref_seq,
                    fetch_start, allowed_reads=None,
                    min_frac=self.fast_pileup_raw_frac)
            else:
                prescreened = self._prescreen_positions(
                    bam[0], geneChr, fetch_start, fetch_end - 1, ref_seq,
                    fetch_start, allowed_reads=allowed_reads,
                    min_frac=self.fast_pileup_min_frac)
            if not prescreened:
                return None, None
            if self.pileup_engine == 'walk':
                (counts, sweeps_reads, sweep_dels,
                 n_pileup_seen, n_allowed_match) = self._sweep_walk(
                    bam[0], geneChr, call_start, call_end, prescreened,
                    ref_seq, fetch_start, allowed_reads, canon_cache)
                col_iter = iter(())
            else:
                counts = {}
                sweeps_reads = {}
                sweep_dels = {}
                col_iter = self._pileup_cols_at(
                    bam[0], geneChr, prescreened,
                    stepper="samtools", min_base_quality=0,
                    min_mapping_quality=self.min_mapq)
            for col in col_iter:
                pos0 = col.reference_pos
                idx = pos0 - fetch_start
                if idx < 0 or idx >= len(ref_seq):
                    continue
                ref_base = ref_seq[idx]
                ref_count = alt_count = del_count = 0
                site_reads = set()
                for pr in col.pileups:
                    aln = pr.alignment
                    qn = aln.query_name
                    cn = canon_cache.get(qn)
                    if cn is None:
                        cn = canonicalize_read_name(qn)
                        canon_cache[qn] = cn
                    n_pileup_seen += 1
                    if allowed_reads is not None and cn not in allowed_reads:
                        continue
                    n_allowed_match += 1
                    mapq = int(aln.mapping_quality) \
                        if aln.mapping_quality is not None else 0
                    is_reverse = 1 if aln.is_reverse else 0
                    read_len = aln.query_length or aln.infer_read_length() or 0
                    if pr.is_del:
                        del_count += 1
                        continue
                    if pr.is_refskip or pr.query_position is None:
                        continue
                    seq_key = (qn, aln.flag, aln.reference_start,
                               aln.reference_end)
                    seq = seq_cache.get(seq_key)
                    if seq is None:
                        seq = aln.query_sequence
                        seq_cache[seq_key] = seq
                    qpos = pr.query_position
                    if seq is None or qpos >= len(seq):
                        continue
                    base = seq[qpos].upper()
                    quals = aln.query_qualities
                    if quals is None:
                        baseq = None
                    else:
                        baseq = int(quals[qpos]) if qpos < len(quals) else 0
                        if baseq < self.min_baseq:
                            continue
                    if base_is_at_read_end(qpos, read_len,
                                           self.min_dist_to_end):
                        continue
                    if base in {"A", "C", "G", "T"}:
                        if base == ref_base:
                            ref_count += 1
                        else:
                            alt_count += 1
                    site_reads.add((cn, qpos, base, baseq, mapq, read_len,
                                    is_reverse))
                eff_depth = ref_count + alt_count


                _keep0 = self.given_snv is not None
                if eff_depth == 0 or (alt_count == 0 and not _keep0):
                    continue
                counts[pos0] = (ref_base, ref_count, alt_count)
                sweeps_reads[pos0] = site_reads
                sweep_dels[pos0] = del_count
        finally:
            bam[0].close()

        if allowed_reads and n_pileup_seen > 0 and n_allowed_match == 0:
            msg = (f'gene {geneID}: {n_pileup_seen} pileup read-positions '
                   f'covered but 0 matched the {len(allowed_reads)} mapping '
                   f'reads — read-name mismatch (canonicalize_read_name vs '
                   f'BAM query_name format?). Gene yields 0 SNVs.')
            if self.logger is not None:
                self.logger.warning(msg)
            else:
                print('WARNING: ' + msg)

        rows = [(geneChr, pos0, rb, rc + ac, ac, ac / (rc + ac))
                for pos0, (rb, rc, ac) in sorted(counts.items())]
        snv_df = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "depth",
                                             "alt_count", "alt_frac"])
        snv_df = snv_df[(snv_df["pos"] >= call_start)
                        & (snv_df["pos"] <= call_end)].copy()
        if _given_fast is not None:
            snv_df = self._restrict_to_given(
                snv_df, _given_fast, geneID, sample_index,
                allele_map=self._given_ref_alt(sample_index))
            if snv_df is None:
                return None, None
        else:
            snv_df = snv_df[(snv_df["depth"] >= self.depth)
                            & (snv_df["alt_count"] > self.n_alt)
                            & self._af_gate_ok(snv_df)].reset_index(drop=True)
            if snv_df.empty:
                return None, None
            snv_df['het_prob'] = self.heterozygous_prob_vec(
                snv_df['depth'].values, snv_df['alt_count'].values,
                e=_het_prefilter_e, priors=_het_prefilter_priors,
                het_beta=self.het_beta)
            if _het_prefilter_threshold >= 0:
                snv_df = snv_df[snv_df['het_prob'] >= _het_prefilter_threshold] \
                    .reset_index(drop=True)
                if snv_df.empty:
                    return None, None

        site_reads = {
            (row.chrom, int(row.pos)): sweeps_reads.get(int(row.pos), set())
            for row in snv_df.itertuples(index=False)
        }
        n_empty = sum(1 for v in site_reads.values() if not v)
        if n_empty > 0:
            tail = ('ALL sites empty -> step3 0-output; do NOT trust this '
                    'gene.' if n_empty == len(site_reads)
                    else 'Investigate before trusting these sites.')
            msg = (f'gene {geneID}: {n_empty}/{len(site_reads)} candidate SNV '
                   f'sites collected 0 site_reads in the fast sweep despite '
                   f'passing depth/alt/het filters. Empty site_reads -> step2 '
                   f'depth=0. ' + tail)
            if self.logger is not None:
                self.logger.warning(msg)
            else:
                print('WARNING: ' + msg)
        site_reads['__del_counts__'] = {
            (row.chrom, int(row.pos)): sweep_dels.get(int(row.pos), 0)
            for row in snv_df.itertuples(index=False)
        }
        return snv_df, site_reads

    def _restrict_to_given(self, snv_df, given, geneID, sample_index,
                           allele_map=None):
        covered = snv_df[snv_df["pos"].astype(int).isin(given)]
        usable = covered[covered["alt_count"] > 0].reset_index(drop=True)
        n_ref_mismatch = 0
        if allele_map:


            keep = []
            for r in usable.itertuples(index=False):
                want = allele_map.get(int(r.pos))
                if want is None or want[0] in (None, '', 'NAN'):
                    keep.append(True)
                    continue
                ok = str(r.ref).upper() == want[0]
                n_ref_mismatch += (not ok)
                keep.append(ok)
            if n_ref_mismatch:
                m = (f'[genotype] {geneID} sample{sample_index}: {n_ref_mismatch} '
                     f'supplied site(s) disagree with the reference base at that '
                     f'position -- the list and this reference build are not the '
                     f'same coordinates; dropped')
                print(m) if self.logger is None else self.logger.info(m)
            usable = usable[pd.Series(keep, index=usable.index)].reset_index(drop=True)
        msg = (f'[genotype] {geneID} sample{sample_index}: given {len(given)}, '
               f'covered {len(covered)}, usable {len(usable)} '
               f'(usable = the ceiling; covered-but-no-ALT sites are not detectable '
               f'heterozygous sites here)'
               + (f'; {n_ref_mismatch} dropped on a reference-base mismatch'
                  if n_ref_mismatch else ''))
        print(msg) if self.logger is None else self.logger.info(msg)
        if usable.empty:
            return None
        usable = usable.copy()
        usable['het_prob'] = np.nan
        return usable

    def _given_sites_for(self, geneID, sample_index, geneChr, geneStart, geneEnd):
        if self.given_snv is None:
            return None
        df = self.given_snv[sample_index]
        m = ((df['chrom'].astype(str) == str(geneChr))
             & (df['pos'].astype(int) >= int(geneStart))
             & (df['pos'].astype(int) <= int(geneEnd)))
        return set(df.loc[m, 'pos'].astype(int).tolist())

    def _given_ref_alt(self, sample_index):
        if self.given_snv is None:
            return {}
        df = self.given_snv[sample_index]
        if 'alt' not in df.columns:
            return {}
        out = {}
        for r in df.itertuples(index=False):
            ref = str(getattr(r, 'ref', '')).upper() if 'ref' in df.columns else None
            out[int(r.pos)] = (ref, str(r.alt).upper())
        return out

    def _call_snvs_for_gene_interval(self, geneID, bam_path_gene, allowed_reads=None,
                                     sample_index=0):
        geneInfo, exonInfo, _ = self.geneStructureInformation[geneID]
        geneChr, geneStart, geneEnd = geneInfo['geneChr'], geneInfo['geneStart'], geneInfo['geneEnd']
        ref_len = self.ref_fasta.get_reference_length(geneChr)
        fetch_start = max(0, geneStart)
        fetch_end = min(ref_len, geneEnd + 1)
        ref_seq = self.ref_fasta.fetch(geneChr, fetch_start, fetch_end).upper()
        _het_prefilter_priors = (0.6, 0.1, 0.3)
        _het_prefilter_e = 0.01
        _het_prefilter_threshold = self.het_prefilter_threshold


        canon_cache = {}


        seq_cache = {}


        n_pileup_seen = 0
        n_allowed_match = 0

        if self.fast_pileup:
            return self._call_snvs_fast(geneID, bam_path_gene,
                                        allowed_reads=allowed_reads,
                                        sample_index=sample_index)


        rows = []
        bam = self._read_bam(bam_path=[bam_path_gene])
        try:
            for col in bam[0].pileup(
                geneChr, geneStart, geneEnd + 1,
                stepper="samtools", min_base_quality=self.min_baseq,
                min_mapping_quality=self.min_mapq, truncate=True,
                flag_filter=_PILEUP_FLAG_FILTER,
            ):
                pos0 = col.reference_pos
                if pos0 < geneStart or pos0 > geneEnd:
                    continue
                idx = pos0 - fetch_start
                if idx < 0 or idx >= len(ref_seq):
                    continue
                ref_base = ref_seq[idx]
                ref_count = 0
                alt_count = 0
                for pr in col.pileups:
                    aln = pr.alignment
                    qn = aln.query_name
                    cn = canon_cache.get(qn)
                    if cn is None:
                        cn = canonicalize_read_name(qn)
                        canon_cache[qn] = cn
                    n_pileup_seen += 1
                    if allowed_reads is not None and cn not in allowed_reads:
                        continue
                    n_allowed_match += 1
                    if pr.is_del or pr.is_refskip or pr.query_position is None:
                        continue


                    seq_key = (qn, aln.flag, aln.reference_start, aln.reference_end)
                    seq = seq_cache.get(seq_key)
                    if seq is None:
                        seq = aln.query_sequence
                        seq_cache[seq_key] = seq
                    qpos = pr.query_position
                    if seq is None or qpos >= len(seq):
                        continue
                    base = seq[qpos].upper()
                    if self.min_dist_to_end > 0:
                        read_len = aln.query_length or aln.infer_read_length() or 0
                        if base_is_at_read_end(qpos, read_len,
                                               self.min_dist_to_end):
                            continue
                    if base in {"A", "C", "G", "T"}:
                        if base == ref_base:
                            ref_count += 1
                        else:
                            alt_count += 1
                eff_depth = ref_count + alt_count


                _keep0 = self.given_snv is not None
                if eff_depth == 0 or (alt_count == 0 and not _keep0):
                    continue
                rows.append((geneChr, pos0, ref_base, eff_depth, alt_count, alt_count / eff_depth))
        finally:
            bam[0].close()

        if allowed_reads and n_pileup_seen > 0 and n_allowed_match == 0:
            msg = (f'gene {geneID}: {n_pileup_seen} pileup read-positions covered but '
                   f'0 matched the {len(allowed_reads)} mapping reads — read-name '
                   f'mismatch (canonicalize_read_name vs BAM query_name format?). '
                   f'Gene yields 0 SNVs.')
            if self.logger is not None:
                self.logger.warning(msg)
            else:
                print('WARNING: ' + msg)

        snv_df = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "depth", "alt_count", "alt_frac"])
        snv_df = snv_df[(snv_df["pos"] >= geneStart) & (snv_df["pos"] <= geneEnd)].copy()
        _given = self._given_sites_for(geneID, sample_index, geneChr, geneStart, geneEnd)
        if _given is not None:


            snv_df = self._restrict_to_given(
                snv_df, _given, geneID, sample_index,
                allele_map=self._given_ref_alt(sample_index))
            if snv_df is None:
                return None, None
        else:

            snv_df = snv_df[(snv_df["depth"] >= self.depth) & (snv_df["alt_count"] > self.n_alt)
                            & self._af_gate_ok(snv_df)].copy()
            snv_df = snv_df.reset_index(drop=True)
            if snv_df.empty:
                return None, None
            snv_df['het_prob'] = self.heterozygous_prob_vec(
                snv_df['depth'].values, snv_df['alt_count'].values,
                e=_het_prefilter_e, priors=_het_prefilter_priors,
                het_beta=self.het_beta)
            if _het_prefilter_threshold >= 0:
                snv_df = snv_df[snv_df['het_prob'] >= _het_prefilter_threshold].reset_index(drop=True)
                if snv_df.empty:
                    return None, None
        candidate_positions = set(snv_df['pos'].astype(int).tolist())


        site_reads_all = {}
        site_del_counts = {}
        bam = self._read_bam(bam_path=[bam_path_gene])
        try:
            for col in bam[0].pileup(
                geneChr, geneStart, geneEnd + 1,
                stepper="samtools", min_base_quality=0,
                min_mapping_quality=self.min_mapq, truncate=True,
                flag_filter=_PILEUP_FLAG_FILTER,
            ):
                pos0 = col.reference_pos
                if pos0 not in candidate_positions:
                    continue
                idx = pos0 - fetch_start
                ref_base = ref_seq[idx]
                del_count = 0
                site_reads = set()
                for pr in col.pileups:
                    aln = pr.alignment
                    qn = aln.query_name
                    canonical_qname = canon_cache.get(qn)
                    if canonical_qname is None:
                        canonical_qname = canonicalize_read_name(qn)
                        canon_cache[qn] = canonical_qname
                    if allowed_reads is not None and canonical_qname not in allowed_reads:
                        continue
                    mapq = int(aln.mapping_quality) if aln.mapping_quality is not None else 0
                    is_reverse = 1 if aln.is_reverse else 0
                    read_len = aln.query_length or aln.infer_read_length() or 0
                    if pr.is_del:
                        del_count += 1
                        continue
                    if pr.is_refskip or pr.query_position is None:
                        continue
                    seq = aln.query_sequence
                    qpos = pr.query_position
                    if seq is None or qpos >= len(seq):
                        continue
                    quals = aln.query_qualities
                    base = seq[qpos].upper()
                    if quals is None:


                        baseq = None
                    else:
                        baseq = int(quals[qpos]) if qpos < len(quals) else 0
                        if baseq < self.min_baseq:
                            continue
                    if base_is_at_read_end(qpos, read_len,
                                           self.min_dist_to_end):
                        continue
                    site_reads.add((canonical_qname, qpos, base, baseq, mapq, read_len, is_reverse))
                site_reads_all[(geneChr, pos0)] = site_reads
                site_del_counts[(geneChr, pos0)] = del_count
        finally:
            bam[0].close()

        site_reads = {
            (row.chrom, int(row.pos)): site_reads_all.get((row.chrom, int(row.pos)), set())
            for row in snv_df.itertuples(index=False)
        }


        n_empty = sum(1 for v in site_reads.values() if not v)
        if n_empty > 0:
            tail = ('ALL sites empty -> step3 0-output; do NOT trust this gene.'
                    if n_empty == len(site_reads)
                    else 'Investigate before trusting these sites.')
            msg = (f'gene {geneID}: {n_empty}/{len(site_reads)} candidate SNV sites '
                   f'collected 0 site_reads in Pass 2 despite passing Pass 1 '
                   f'depth/alt/het filters (base-quality asymmetry? read-name '
                   f'mismatch?). Empty site_reads -> step2 depth=0. ' + tail)
            if self.logger is not None:
                self.logger.warning(msg)
            else:
                print('WARNING: ' + msg)
        site_reads['__del_counts__'] = {
            (row.chrom, int(row.pos)): site_del_counts.get((row.chrom, int(row.pos)), 0)
            for row in snv_df.itertuples(index=False)
        }
        return snv_df, site_reads
    def process_genes_round1_1(self):


        self.gene_index, self._geneidx_colpos = self._load_gene_index()
        if self.gene_index is not None:
            self.mapping_df_list = None
            mes = 'using per-gene index (seek mode); skipping full mapping load'
            print(mes) if self.logger is None else self.logger.info(mes)
        else:
            self.mapping_df_list = self._read_mapping()
        os.makedirs(self.variants_by_gene_folder_1, exist_ok=True)
        geneIDs = list(self.geneStructureInformation.keys())
        if self.gene_subset is not None:
            geneIDs = [g for g in geneIDs if g in self.gene_subset]

        def _gene_n_reads(g):
            if self.gene_index is not None:
                return sum(idx.get(g, (0, 0, 0))[2] for idx in self.gene_index)
            return sum(len(d.get(g, [])) for d in self.mapping_df_list)

        def _gene_max_sample_reads(g):
            if self.gene_index is not None:
                return max((idx.get(g, (0, 0, 0))[2]
                            for idx in self.gene_index), default=0)
            return max((len(d.get(g, [])) for d in self.mapping_df_list),
                       default=0)


        n_before = len(geneIDs)
        if self.given_snv is not None:


            mes = (f'sub-depth gene guard: DISABLED for genotype mode (its exactness '
                   f'assumes the site depth gate, which is skipped here); all '
                   f'{n_before} gene(s) scanned')
            print(mes) if self.logger is None else self.logger.info(mes)
        else:
            sub_depth = {g for g in geneIDs
                         if _gene_max_sample_reads(g) < self.gene_guard_depth}
            geneIDs = [g for g in geneIDs if g not in sub_depth]
            mes = (f'sub-depth gene guard: skipped {len(sub_depth)}/{n_before} '
                   f'genes with < {self.gene_guard_depth} whitelisted reads (candidate-'
                   f'impossible; exact skip)')
            print(mes) if self.logger is None else self.logger.info(mes)


        def _gene_cost(g):
            gInfo = self.geneStructureInformation[g][0]
            gene_len = max(1, gInfo['geneEnd'] - gInfo['geneStart'])
            return gene_len * max(1, _gene_n_reads(g))
        gene_costs = [(g, _gene_cost(g)) for g in geneIDs]
        gene_costs.sort(key=lambda x: -x[1])

        chunk_costs = [0] * self.n_jobs
        chunk_genes = [[] for _ in range(self.n_jobs)]
        for g, cost in gene_costs:
            min_idx = chunk_costs.index(min(chunk_costs))
            chunk_genes[min_idx].append(g)
            chunk_costs[min_idx] += cost

        for i in range(self.n_jobs):
            chunk_genes[i].sort(key=lambda g: (
                self.geneStructureInformation[g][0]['geneChr'],
                self.geneStructureInformation[g][0]['geneStart']))
        geneIDs_job = chunk_genes[self.job_index]


        existed = [f.split('_')[0] for f in os.listdir(self.variants_by_gene_folder_1)
                   if f.endswith('_site_reads.pkl')]
        geneIDs_job = [geneid for geneid in geneIDs_job if geneid not in existed]
        mes = f'process {len(geneIDs_job)} genes for job index {self.job_index} (chunk cost={chunk_costs[self.job_index]:.0f}):'
        print(mes) if self.logger is None else self.logger.info(mes)
        for geneID in geneIDs_job:
            snv_path = os.path.join(self.variants_by_gene_folder_1, f"{geneID}_snvs.csv")
            site_reads_path = os.path.join(self.variants_by_gene_folder_1, f"{geneID}_site_reads.pkl")
            snv_exists = os.path.exists(snv_path)
            site_reads_exists = os.path.exists(site_reads_path)
            if snv_exists and site_reads_exists:
                continue
            if snv_exists != site_reads_exists:
                if snv_exists:
                    os.remove(snv_path)
                if site_reads_exists:
                    os.remove(site_reads_path)
            print(f'process {geneID}')
            reads_set_list, geneInfo = self._reads_set_for_gene(geneID)
            if all(len(s) == 0 for s in reads_set_list):
                continue
            print('call snv round')
            gene_chr = geneInfo['geneChr']
            per_sample_snvs = []
            merged_site_reads = defaultdict(set)
            merged_del_counts = defaultdict(int)
            for sample_index, allowed_reads in enumerate(reads_set_list):
                if len(allowed_reads) == 0:
                    continue
                try:
                    bam_path_gene = self._get_bam_file_path(self.bam_path[sample_index], chrom=gene_chr)
                except FileNotFoundError as exc:


                    if _CHROM_BAM_NAME_RE.match(str(gene_chr)):
                        raise BamInputError(f'step1 {geneID} (sample {sample_index}): {exc} -- '
                                            f'no gene is skipped silently; restrict '
                                            f'--gene_subset_path if that chromosome is '
                                            f'intentionally absent') from exc
                    self._note_unresolvable_contig('step1', sample_index, gene_chr, geneID)
                    continue
                snv_df_sample, site_reads_sample = self._call_snvs_for_gene_interval(
                    geneID,
                    bam_path_gene,
                    allowed_reads=allowed_reads,
                    sample_index=sample_index,
                )
                if snv_df_sample is not None:
                    per_sample_snvs.append(snv_df_sample)
                if site_reads_sample is not None:
                    sample_del_counts = site_reads_sample.pop('__del_counts__', {})
                    for site_key, count in sample_del_counts.items():
                        merged_del_counts[site_key] += count
                    for site, sr in site_reads_sample.items():
                        merged_site_reads[site].update(sr)
            if per_sample_snvs:
                if len(per_sample_snvs) == 1:
                    snv_df = per_sample_snvs[0].copy()
                else:
                    snv_df = pd.concat(per_sample_snvs, ignore_index=True)
                    snv_df = (
                        snv_df.groupby(["chrom", "pos", "ref"], as_index=False)
                        .agg({"depth": "sum", "alt_count": "sum"})
                    )
                    snv_df["alt_frac"] = snv_df["alt_count"] / snv_df["depth"]
                    snv_df["het_prob"] = self.heterozygous_prob_vec(
                        snv_df["depth"].values, snv_df["alt_count"].values,
                        e=0.01, priors=(0.6, 0.1, 0.3),
                        het_beta=self.het_beta
                    )
                    snv_df = snv_df[["chrom", "pos", "ref", "depth", "alt_count", "alt_frac", "het_prob"]]
                    snv_df = snv_df.sort_values(["chrom", "pos"]).reset_index(drop=True)
                site_reads_dict = {
                    (chrom, pos): merged_site_reads[(chrom, pos)]
                    for chrom, pos in snv_df[["chrom", "pos"]].itertuples(index=False, name=None)
                }
                site_reads_dict['__del_counts__'] = {
                    (chrom, pos): merged_del_counts.get((chrom, pos), 0)
                    for chrom, pos in snv_df[["chrom", "pos"]].itertuples(index=False, name=None)
                }
            else:
                snv_df, site_reads_dict = None, None
            if snv_df is not None:
                print(f'save snv calls for gene {geneID}')
                _atomic_to_csv(snv_df, snv_path)
                _atomic_pickle_dump(site_reads_dict, site_reads_path)

    def process_read_blocks_round1_5(self):
        if self.n_jobs != self.n_samples:
            raise ValueError(
                f'step1.5 expects --n_jobs ({self.n_jobs}) == n_samples '
                f'({self.n_samples}); each task processes one BAM.')
        if self.job_index < 0 or self.job_index >= self.n_samples:
            raise ValueError(
                f'step1.5 --job_index {self.job_index} out of range for '
                f'{self.n_samples} samples.')

        self.mapping_df_list = self._read_mapping()
        sample_index = self.job_index
        bam_path_root = self.bam_path[sample_index]
        mapping_df = self.mapping_df_list[sample_index]
        allowed_reads_by_gene = {g: set(df['Read'].tolist())
                                 for g, df in mapping_df.items() if not df.empty}

        os.makedirs(self.variants_by_gene_folder_1, exist_ok=True)
        geneIDs = list(self.geneStructureInformation.keys())
        if self.gene_subset is not None:
            geneIDs = [g for g in geneIDs if g in self.gene_subset]

        by_chrom = defaultdict(list)
        for g in geneIDs:
            info = self.geneStructureInformation[g][0]
            by_chrom[info['geneChr']].append(g)

        n_genes_written = 0
        n_genes_skipped = 0
        for chrom, gene_list in by_chrom.items():
            try:
                bam_path_chrom = self._get_bam_file_path(bam_path_root, chrom=chrom)
            except FileNotFoundError as exc:


                if _CHROM_BAM_NAME_RE.match(str(chrom)):
                    raise BamInputError(f'step1.5 sample {sample_index}: {exc} -- '
                                        f'{len(gene_list)} gene(s) on {chrom} would have been '
                                        f'skipped silently') from exc
                self._note_unresolvable_contig('step1.5', sample_index, chrom,
                                               f'{len(gene_list)} gene(s)')
                continue
            try:
                bam = pysam.AlignmentFile(bam_path_chrom, 'rb')
            except (OSError, ValueError) as exc:
                mes = (f'[step1.5] sample {sample_index}: failed to open BAM '
                       f'{bam_path_chrom} for chrom {chrom}: {exc}')
                print(mes) if self.logger is None else self.logger.warning(mes)
                continue
            try:
                for geneID in gene_list:
                    info = self.geneStructureInformation[geneID][0]
                    out_path = os.path.join(
                        self.variants_by_gene_folder_1,
                        f'{geneID}_read_blocks_{sample_index}.pkl')
                    if os.path.exists(out_path):
                        n_genes_skipped += 1
                        continue
                    allowed = allowed_reads_by_gene.get(geneID)
                    if not allowed:


                        _atomic_pickle_dump({}, out_path)
                        n_genes_written += 1
                        continue
                    per_gene = {}
                    try:
                        fetched = bam.fetch(chrom, int(info['geneStart']),
                                            int(info['geneEnd']) + 1)
                    except (ValueError, OSError) as exc:
                        mes = (f'[step1.5] {geneID}: BAM fetch failed ({exc}); '
                               f'writing empty pkl')
                        print(mes) if self.logger is None else self.logger.warning(mes)
                        _atomic_pickle_dump({}, out_path)
                        n_genes_written += 1
                        continue
                    for read in fetched:
                        if read.is_unmapped or read.is_secondary or read.is_supplementary:
                            continue
                        qn = canonicalize_read_name(read.query_name)
                        if qn not in allowed or qn in per_gene:
                            continue
                        blocks = read.get_blocks()
                        if not blocks:
                            continue
                        per_gene[qn] = (blocks, compute_intron_spans(read))
                    _atomic_pickle_dump(per_gene, out_path)
                    n_genes_written += 1
            finally:
                bam.close()
        mes = (f'[step1.5] sample {sample_index} (BAM {bam_path_root}): '
               f'wrote {n_genes_written} gene pkls, '
               f'skipped {n_genes_skipped} already-existing')
        print(mes) if self.logger is None else self.logger.info(mes)

    def merge_read_blocks_round1_5(self):
        if not os.path.isdir(self.variants_by_gene_folder_1):
            return
        intermediate_re = re.compile(r'^(.+)_read_blocks_(\d+)\.pkl$')
        groups = defaultdict(list)
        for fname in os.listdir(self.variants_by_gene_folder_1):
            m = intermediate_re.match(fname)
            if not m:
                continue
            gene_id, sidx = m.group(1), int(m.group(2))
            groups[gene_id].append((sidx, os.path.join(
                self.variants_by_gene_folder_1, fname)))
        if not groups:
            mes = '[step1.5 merge] no intermediate _read_blocks_*.pkl files found'
            print(mes) if self.logger is None else self.logger.info(mes)
            return

        n_merged = 0
        for gene_id, parts in groups.items():
            out_path = os.path.join(self.variants_by_gene_folder_1,
                                    f'{gene_id}_read_blocks.pkl')
            merged = {}
            for sidx, path in sorted(parts):
                try:
                    sample_dict = load_pickle(path)
                except (OSError, EOFError, pickle.UnpicklingError) as exc:
                    mes = f'[step1.5 merge] {gene_id} sample {sidx}: load failed ({exc}); skipping'
                    print(mes) if self.logger is None else self.logger.warning(mes)
                    continue
                if not isinstance(sample_dict, dict):
                    continue
                before = len(merged)
                merged.update(sample_dict)
                collisions = before + len(sample_dict) - len(merged)
                if collisions > 0:
                    mes = (f'[step1.5 merge] {gene_id} sample {sidx}: '
                           f'{collisions} read-name collisions with prior samples '
                           f'(last-writer-wins). Check for duplicate-BAM / '
                           f'replicate-input cohort setup.')
                    print(mes) if self.logger is None else self.logger.warning(mes)
            _atomic_pickle_dump(merged, out_path)
            for _, path in parts:
                try:
                    os.remove(path)
                except OSError:
                    pass
            n_merged += 1
        mes = (f'[step1.5 merge] merged {n_merged} genes into canonical '
               f'_read_blocks.pkl; intermediate _read_blocks_<N>.pkl files removed')
        print(mes) if self.logger is None else self.logger.info(mes)

    def _load_round_outputs(self, geneID):
        p1 = os.path.join(self.variants_by_gene_folder_1, f"{geneID}_snvs.csv")
        r1 = os.path.join(self.variants_by_gene_folder_1, f"{geneID}_site_reads.pkl")
        snv1 = pd.read_csv(p1, index_col=0) if os.path.exists(p1) else None
        site_reads1 = load_pickle(r1)
        return snv1, site_reads1
    def _build_em_input_gene(self, geneID, merged_df_snv, site_reads, sample_index):
        geneInfo, _, _ = self.geneStructureInformation[geneID]


        if self.mapping_df_list is None:
            reads = self._gene_reads_ordered(sample_index, geneID)
        else:
            mapping_df_gene = self.mapping_df_list[sample_index].get(geneID)
            if mapping_df_gene is not None:
                reads = mapping_df_gene['Read'].drop_duplicates().tolist()
            else:
                reads = []
        read2row = {r: i for i, r in enumerate(reads)}
        sites = sorted({(str(r.chrom), int(r.pos), str(r.ref))
                        for r in merged_df_snv.itertuples(index=False)},
                       key=lambda x: (x[0], x[1]))
        nR, nS = len(reads), len(sites)

        r_code = np.full((nR, nS), -1, dtype=np.int8)
        pi_arr = np.full((nR, nS), np.nan, dtype=np.float32)
        alt_for_site = {}
        depth_alt = np.zeros((nS, 2), dtype=np.int32)
        ACGT = {"A", "C", "G", "T"}
        _cap = self.max_baseq
        def q_to_pi(q):
            try:
                qv = float(q)
            except (TypeError, ValueError):
                return 0.25
            if not np.isfinite(qv):
                return 0.25


            if _cap is not None and qv > _cap:
                qv = _cap
            return float(np.clip(10.0 ** (-(qv / 10.0)), 0.0, 1.0))
        for j, (chrom, pos0, refb) in enumerate(sites):
            tuples = site_reads.get((chrom, pos0), set())
            if not tuples:
                alt_for_site[(chrom, pos0)] = None
                continue

            pis_by_base = defaultdict(list)
            for tup in tuples:
                _rn, _qpos, base, q = tup[0], tup[1], tup[2], tup[3]
                if not isinstance(base, str):
                    continue
                b = base.upper()
                if b in ACGT and b != refb:
                    pi = q_to_pi(q)
                    if np.isfinite(pi):
                        pis_by_base[b].append(pi)
                    else:
                        pis_by_base[b].append(np.nan)
            if not pis_by_base:
                alt = None
            else:
                counts = {b: sum(1 for v in vs if True) for b, vs in pis_by_base.items()}
                max_ct = max(counts.values())
                cands = [b for b, ct in counts.items() if ct == max_ct]
                if len(cands) == 1:
                    alt = cands[0]
                else:

                    def mean_pi(b):
                        vals = [v for v in pis_by_base[b] if np.isfinite(v)]
                        return (np.mean(vals) if vals else np.inf)
                    alt = min(cands, key=lambda b: (mean_pi(b), b))
            alt_for_site[(chrom, pos0)] = alt


            def _rec_order(t):
                bq = t[3] if isinstance(t[3], (int, float)) and t[3] is not None else -1
                return (str(t[0]), -int(t[4]), -float(bq), int(t[1]), str(t[2]))
            for tup in sorted(tuples, key=_rec_order):
                rn, qpos, base, q = tup[0], tup[1], tup[2], tup[3]
                i = read2row.get(rn)
                if i is None:
                    continue
                if r_code[i, j] != -1:
                    continue
                b = base.upper() if isinstance(base, str) else None
                pi = q_to_pi(q)
                if b is None or b not in ACGT:
                    code = -1
                    pi = np.nan
                elif b == refb:
                    code = 0
                    depth_alt[j, 0] += 1
                elif alt is not None and b == alt:
                    code = 1
                    depth_alt[j, 0] += 1
                    depth_alt[j, 1] += 1
                else:
                    code = 2
                    depth_alt[j, 0] += 1
                r_code[i, j] = code
                pi_arr[i, j] = pi

        rows = []
        for j, (chrom, pos0, refb) in enumerate(sites):
            d, a = int(depth_alt[j, 0]), int(depth_alt[j, 1])
            rows.append({
                "chrom": chrom,
                "pos": pos0,
                "ref": refb,
                "alt": alt_for_site.get((chrom, pos0)),
                "depth": d,
                "alt_count": a,
                "alt_frac": (a / d if d > 0 else 0.0)})
        snv_df_final = pd.DataFrame(rows).sort_values(["chrom", "pos"]).reset_index(drop=True)
        cols = [f"{c}_{p}_{r}" for (c, p, r) in sites]
        return snv_df_final, r_code, pi_arr, reads, cols
    def process_genes_final(self):


        self.gene_index, self._geneidx_colpos = self._load_gene_index()
        if self.gene_index is not None:
            self.mapping_df_list = None
            mes = 'step2: using per-gene index (seek mode); skipping full mapping load'
            print(mes) if self.logger is None else self.logger.info(mes)
        else:
            self.mapping_df_list = self._read_mapping()
        if self.n_samples == 1:
            em_dirs = [self.em_input]
        else:
            em_dirs = [os.path.join(self.em_input, sn) for sn in self.sample_names]
        for d in em_dirs:
            os.makedirs(d, exist_ok=True)
        geneIDs = list(self.geneStructureInformation.keys())
        if self.gene_subset is not None:
            geneIDs = [g for g in geneIDs if g in self.gene_subset]
        geneIDs = sorted(geneIDs, key=lambda g: (
            self.geneStructureInformation[g][0]['geneChr'],
            self.geneStructureInformation[g][0]['geneStart']))
        geneIDs_job = np.array_split(geneIDs, self.n_jobs)[self.job_index]

        def _one_gene(geneID):

            print(f'process gene {geneID}')
            idx_iter = [0] if self.n_samples == 1 else range(self.n_samples)
            need = []
            for i in idx_iter:
                outdir = em_dirs[i]
                f_pile = os.path.join(outdir, f"{geneID}_pileup.csv")
                f_npz = os.path.join(outdir, f"{geneID}_read_matrices.npz")
                f_r = os.path.join(outdir, f"{geneID}_read_snv.csv")
                f_pi = os.path.join(outdir, f"{geneID}_read_pi.csv")
                has_em_input = os.path.exists(f_npz) or (os.path.exists(f_r) and os.path.exists(f_pi))
                if not (os.path.exists(f_pile) and has_em_input):
                    need.append((i, f_pile, f_npz))
            if not need:


                print('input files already exist')
                return
            try:
                snv1, site_reads1 = self._load_round_outputs(geneID)
                if snv1 is None:
                    return
                for ind, f_pile, f_npz in need:
                    snv_df_final, r_code, pi_arr, reads, cols = self._build_em_input_gene(geneID, snv1, site_reads1, ind)
                    _atomic_to_csv(snv_df_final, f_pile, index=False)
                    _save_em_input_npz(f_npz, r_code, pi_arr, reads, cols)
            except Exception as e:
                print(f"Error processing {geneID}: {e}")


        if self.n_workers != 1 and len(geneIDs_job) > 1:
            Parallel(n_jobs=self.n_workers, prefer='threads')(
                delayed(_one_gene)(g) for g in geneIDs_job)
        else:
            for geneID in geneIDs_job:
                _one_gene(geneID)


class Haplotyping:
    def __init__(self, scotch_target:Union[str, Sequence[str]],
                 bam_path = None,
                 target = None, sample_names = None,
                 max_iter=50, tol=1e-3, verbose=False, seed = 42,
                 mtx = True, csv = False, n_jobs = 1, job_index = 0, n_alt = 10, depth = 20,
                 heterozygous_filter = -1, alt_stretch_filter = 20, alt_cluster_filter = 20,
                 repeat_filter_kmer=1,
                 var_cluster_window = 20, var_cluster_n = 3,
                 sample_name_parse = None, prefix = 'LongAllele',
                 em_snv_filter = True, em_max_reads=20000, snv_confidence = None, snv_classifier = None,
                 phase_block_min_shared = 1, phase_block_min_agreement = 0.0,
                 het_beta = None, shrink_denominator = 'gene_reads',
                 heterozygous_coverage_factor = 1.0,
                 clf_hard_threshold = 0.005, clf_init = False,
                 gap_tau = 0.10, clf_pruning_threshold = 0.1, clf_pruning_frac = 1.0,
                 rna_editing_db = None,
                 chi_min_frac = 0.10, chi_group_novel = False,
                 cell_type_df_path = None, ref_pickle_path = None, cover_existing = False, n_workers = -1,
                 ref_fasta_path=None, gene_subset = None, logger = None,
                 job_array_by_sample = False,
                 high_artifact_mode = False, novel_exon_pct_max = 0.25,
                 read_intronic_pct_max = 0.60, read_sj_min = 0,
                 gsi_base_pkl_path = None,
                 skip_ase_test = False, skip_astu_test = False,
                 step3_backend = 'threads',
                 min_alt_frac = 0.0,
                 editing_exempt_affinity = 2.0, editing_exempt_min_reads = 10,
                 em_init_method = 'signed', phase_flip = True,
                 max_baseq = None, alt_stretch_len = 5,
                 init_link_min_agreement = 0.0, init_link_min_shared = 3,
                 h_m_init_from = 'clf'):

        self.logger = logger
        self._log_file = log_file_of(logger)

        self.min_alt_frac = validate_min_alt_frac(min_alt_frac)


        self.editing_exempt_affinity = float(editing_exempt_affinity)
        if not np.isfinite(self.editing_exempt_affinity):
            raise ValueError('editing_exempt_affinity must be finite '
                             '(>1 disables the exemption)')
        self.editing_exempt_min_reads = int(editing_exempt_min_reads)
        if self.editing_exempt_min_reads < 1:
            raise ValueError('editing_exempt_min_reads must be >= 1')
        if step3_backend not in ('threads', 'loky'):
            raise ValueError(f"step3_backend must be 'threads' or 'loky', "
                             f"got {step3_backend!r}")
        self.step3_backend = step3_backend
        if em_init_method not in ('signed', 'concurrence'):
            raise ValueError(f"em_init_method must be 'signed' or "
                             f"'concurrence', got {em_init_method!r}")
        self.em_init_method = em_init_method
        self.phase_flip = phase_flip
        self.mapping_df_dict_list = None
        self._hap_gene_index = None
        self._hap_gene_cols = None
        self._block_df_cache = {}


        self.skip_ase_test = skip_ase_test
        self.skip_astu_test = skip_astu_test
        self.n_workers = n_workers
        self.job_array_by_sample = job_array_by_sample
        self.cover_existing = cover_existing
        self.sample_name_parse = sample_name_parse
        self.em_snv_filter = em_snv_filter
        self.em_max_reads = em_max_reads
        self.clf_hard_threshold = clf_hard_threshold
        self.clf_init = clf_init
        self.gap_tau = gap_tau
        self.clf_pruning_threshold = clf_pruning_threshold
        self.clf_pruning_frac = clf_pruning_frac
        self.var_cluster_window = var_cluster_window
        self.var_cluster_n = var_cluster_n
        self.alt_stretch_filter = alt_stretch_filter


        self.alt_stretch_len = int(alt_stretch_len)


        self.init_link_min_agreement = float(init_link_min_agreement)
        self.init_link_min_shared = int(init_link_min_shared)
        if h_m_init_from not in ('clf', 'linkage', 'linkage_lr', 'none'):
            raise ValueError("h_m_init_from must be 'clf', 'linkage', "
                             f"'linkage_lr' or 'none', got {h_m_init_from!r}")
        self.h_m_init_from = h_m_init_from
        self.alt_cluster_filter = alt_cluster_filter
        self.repeat_filter_kmer = repeat_filter_kmer
        self.ref_fasta_path = ref_fasta_path
        self.fasta_handle = pysam.FastaFile(ref_fasta_path) if ref_fasta_path is not None else None


        self._fasta_lock = threading.Lock()
        self.n_jobs = n_jobs
        self.job_index = job_index
        self.target = target
        self.scotch_target = self._ensure_list(scotch_target if scotch_target is not None else target)
        self.n_samples = len(self.scotch_target)
        if sample_names is None:
            self.sample_names = [os.path.basename(st) for st in self.scotch_target]
        else:
            sample_names = self._ensure_list(sample_names)
            if len(sample_names) == 1 and self.n_samples == 1:
                self.sample_names = sample_names
            elif len(sample_names) == self.n_samples:
                self.sample_names = sample_names
            else:
                raise ValueError('sample_names must contain one entry per sample.')

        self.bam_path = self._parse_comma_list(bam_path) if bam_path is not None else None

        self.snv_classifier_model = None


        self._clf_source_path = None
        self._clf_digest = None
        if snv_classifier is not None and str(snv_classifier).strip():
            self.snv_classifier_model = joblib_load(snv_classifier)
            self._clf_source_path = snv_classifier
            _, self._clf_digest = _load_asset_bytes(snv_classifier, kind='classifier')
            mes = f'Loaded SNV classifier from {snv_classifier}'
            print(mes) if self.logger is None else self.logger.info(mes)


            if snv_confidence is None:


                self._check_classifier_het_scale(snv_classifier, het_beta,
                                                 max_baseq)
            else:
                mes = ('genotype mode: the classifier will not be used, so its '
                       'het_prob scale is not checked against --het_beta')
                print(mes) if self.logger is None else self.logger.info(mes)
        if ref_pickle_path is not None:
            gsi_path = ref_pickle_path
        else:
            gsi_path = _resolve_reference_pickle_path(self.scotch_target, self.logger)
        gsi_blob, gsi_digest = _load_gsi_with_digest(gsi_path, self.logger)
        self.geneStructureInformation = gsi_blob


        self._gsi_source_path = gsi_path
        self._gsi_digest = gsi_digest
        self._gsi_len = len(gsi_blob) if isinstance(gsi_blob, dict) else None
        self._gsi_from_disk = gsi_digest is not None
        self.prefix = prefix or None
        if self.sample_name_parse is not None:
            self.read_isoform_mapping_path_list = [resolve_scotch_auxiliary_tsv(self.scotch_target[0], self.sample_name_parse)]
        else:
            self.read_isoform_mapping_path_list = [resolve_scotch_auxiliary_tsv(self.scotch_target[i]) for i in range(self.n_samples)]

        self.em_input = os.path.join(self.target, 'em_input')

        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.seed = seed
        self.mtx = mtx
        self.csv = csv
        self.n_alt = n_alt
        self.depth = depth
        self.heterozygous_filter = heterozygous_filter


        self.max_baseq = int(max_baseq) if max_baseq is not None else None
        self.snv_confidence = snv_confidence


        self.phase_block_min_shared = phase_block_min_shared
        self.phase_block_min_agreement = phase_block_min_agreement


        self.het_beta = het_beta
        self.shrink_denominator = shrink_denominator
        self.heterozygous_coverage_factor = heterozygous_coverage_factor
        self.rna_editing_db = self._load_rna_editing_db(rna_editing_db)
        if self.rna_editing_db is not None:
            n_rna_editing_sites = sum(
                arr.size
                for by_chrom in self.rna_editing_db.values()
                for arr in by_chrom.values()
            )
            mes = (
                f'Loaded RNA editing DB from {rna_editing_db} with '
                f'{n_rna_editing_sites} canonical A-to-I sites (0-based coordinates).'
            )
            print(mes) if self.logger is None else self.logger.info(mes)
        self.chi_min_frac = chi_min_frac
        self.chi_group_novel = chi_group_novel
        self.gene_subset = set(gene_subset) if gene_subset is not None else None
        if cell_type_df_path is None:
            self.cell_type_df_list = None
        else:
            cell_type_df_path = self._ensure_list(cell_type_df_path)
            cell_type_df_list = []
            for cell_type_df_path_ in cell_type_df_path:
                df_ = pd.read_csv(cell_type_df_path_)
                cell_type_df_list.append(df_)
            self.cell_type_df_list = cell_type_df_list


        self.high_artifact_mode = bool(high_artifact_mode)
        self.novel_exon_pct_max = float(novel_exon_pct_max)
        self.read_intronic_pct_max = float(read_intronic_pct_max)


        self.read_sj_min = int(read_sj_min)
        self.nascent_leak_intervals = None
        self.canonical_exons = None
        self._bam_path_cache = {}
        if self.high_artifact_mode:
            base_path = gsi_base_pkl_path or _resolve_base_reference_pickle_path(
                self.scotch_target, self.logger)
            if base_path is None:
                raise FileNotFoundError(
                    '--high_artifact_mode requires the SCOTCH base pickle '
                    '(geneStructureInformation.pkl or metageneStructureInformation.pkl) under '
                    f'{os.path.join(self.scotch_target[0], "reference")}; '
                    'rerun SCOTCH annotation step or supply --gsi_base_pkl_path. '
                    'Refusing to silently fall back to the SCOTCH-augmented annotation as the base.'
                )
            gsi_base = _load_gene_structure_information(base_path, self.logger)
            if gsi_base is None:
                raise RuntimeError(
                    f'--high_artifact_mode: failed to load base pickle from {base_path}'
                )
            self.canonical_exons = compute_canonical_exons(gsi_base, self.logger)


            if self.bam_path is not None and len(self.bam_path) != self.n_samples:
                raise KnobCInputError(f'--high_artifact_mode: {len(self.bam_path)} --bam_path '
                                      f'entries for {self.n_samples} sample(s); the read filter '
                                      f'indexes BAMs by sample')
            check_knob_c_bams(self.bam_path, logger=self.logger,
                              chroms={v[0] for v in self.canonical_exons.values()})
            self.nascent_leak_intervals = compute_nascent_leak_intervals(
                self.geneStructureInformation, self.logger)
            mes = (f'[high_artifact_mode] enabled. '
                   f'Knob B cutoff (intron_filled_pct) = {self.novel_exon_pct_max}; '
                   f'Knob C cutoff (read intronic_pct) = {self.read_intronic_pct_max}. '
                   f'Scope: long-read snRNA-seq nascent-leak filters at step3.')


            print(mes)
            if self.logger is not None:
                self.logger.info(mes)
    def _get_paths(self, sample_index = None):
        if self.n_samples==1:
            self.geneIDs = _list_gene_ids_from_em_input(self.em_input)
            if self.gene_subset is not None:
                self.geneIDs = [g for g in self.geneIDs if g in self.gene_subset]
            self.snv_hap_path = os.path.join(self.target, 'snv_hap_' + self.prefix) if self.prefix is not None else os.path.join(self.target, 'snv_hap')
            self.summary_statistics_path = os.path.join(self.target, 'summary_statistics_' + self.prefix) if self.prefix is not None else os.path.join(self.target, 'summary_statistics')
            self.count_hap_folder_path = os.path.join(self.target, 'count_matrix_hap_' + self.prefix) if self.prefix is not None else os.path.join(self.target, 'count_matrix_hap')
        else:
            self.geneIDs = _list_gene_ids_from_em_input(os.path.join(self.em_input, self.sample_names[sample_index]))
            if self.gene_subset is not None:
                self.geneIDs = [g for g in self.geneIDs if g in self.gene_subset]
            self.snv_hap_path = os.path.join(self.target,self.sample_names[sample_index],
                                             'snv_hap_' + self.prefix) if self.prefix is not None else os.path.join(
                self.target, self.sample_names[sample_index],'snv_hap')
            self.summary_statistics_path = os.path.join(self.target,self.sample_names[sample_index],
                                                        'summary_statistics_' + self.prefix) if self.prefix is not None else os.path.join(
                self.target, self.sample_names[sample_index],'summary_statistics')
            self.count_hap_folder_path = os.path.join(self.target,self.sample_names[sample_index],
                                                      'count_matrix_hap_' + self.prefix) if self.prefix is not None else os.path.join(
                self.target, self.sample_names[sample_index], 'count_matrix_hap')
    @staticmethod
    def _ensure_list(x):
        if isinstance(x, str):
            return [x]
        if isinstance(x, Iterable):
            return list(x)
        raise TypeError("Expected str or iterable of str")
    @staticmethod
    def _parse_comma_list(x):
        if x is None:
            return None
        if isinstance(x, str):
            return [p.strip() for p in x.split(',') if p.strip()]
        if isinstance(x, Iterable):
            result = []
            for item in x:
                if isinstance(item, str):
                    result.extend(p.strip() for p in item.split(',') if p.strip())
                else:
                    result.append(item)
            return result
        return [x]

    def _extract_snv_classifier_features(self, site_reads_dict, df_pileup_filtered):
        rows = []
        for row in df_pileup_filtered.itertuples(index=False):
            chrom = str(row.chrom)
            pos0 = int(row.pos)
            ref_base = str(row.ref).upper()
            alt_base = str(row.alt).upper()

            tuples = site_reads_dict.get((chrom, pos0), set())
            del_counts_dict = site_reads_dict.get('__del_counts__', {})
            del_count = del_counts_dict.get((chrom, pos0), 0)
            mapqs = []
            alt_bqs = []
            ref_bqs = []
            alt_positions = []
            alt_base_counts = defaultdict(int)
            alt_fwd, alt_rev, ref_fwd, ref_rev = 0, 0, 0, 0

            for _read_name, qpos, base, baseq, mapq, read_len, is_reverse in tuples:
                base = str(base).upper()


                baseq = int(baseq) if baseq is not None else None
                mapq = int(mapq)
                read_len = int(read_len)
                is_reverse = int(is_reverse)
                qpos = int(qpos)

                if base not in {'A', 'C', 'G', 'T'}:
                    continue
                mapqs.append(mapq)
                if base != ref_base:
                    alt_base_counts[base] += 1
                if base == alt_base:
                    if baseq is not None:
                        alt_bqs.append(baseq)
                    if read_len > 1:
                        alt_positions.append(float(qpos) / max(read_len - 1, 1))
                    if is_reverse:
                        alt_rev += 1
                    else:
                        alt_fwd += 1
                elif base == ref_base:
                    if baseq is not None:
                        ref_bqs.append(baseq)
                    if is_reverse:
                        ref_rev += 1
                    else:
                        ref_fwd += 1

            depth = len(mapqs)
            alt_ct = int(alt_base_counts.get(alt_base, 0))

            x00, x01, x10, x11 = ref_fwd+1, ref_rev+1, alt_fwd+1, alt_rev+1
            sym = (x00*x11)/(x01*x10) + (x01*x10)/(x00*x11)
            ref_ratio = min(x00,x01) / max(x00,x01)
            alt_ratio = min(x10,x11) / max(x10,x11)
            strand_sor = float(np.log(sym) + np.log(ref_ratio) - np.log(alt_ratio))

            total_with_del = depth + del_count
            del_frac = del_count / total_with_del if total_with_del > 0 else 0.0


            gc_content_11bp = 0.5
            homopolymer_len = 1
            is_homopolymer_ge5 = 0
            creates_homopolymer = 0
            flanking_is_AT = 0
            is_transition = 0
            if self.fasta_handle is not None:
                try:
                    with self._fasta_lock:
                        chrom_len = self.fasta_handle.get_reference_length(chrom)
                        fetch_start = max(0, pos0 - 5)
                        fetch_end = min(chrom_len, pos0 + 6)
                        seq = self.fasta_handle.fetch(chrom, fetch_start, fetch_end).upper()
                    if len(seq) < 11:
                        seq = seq + 'N' * (11 - len(seq))
                    snv_idx = pos0 - fetch_start

                    gc_content_11bp = sum(1 for b in seq if b in 'GC') / len(seq)

                    homopolymer_len = max((len(m.group()) for m in re.finditer(r'(.)\1*', seq)), default=1)
                    is_homopolymer_ge5 = 1 if homopolymer_len >= 5 else 0

                    if 0 < snv_idx < len(seq) - 1:
                        trinuc = seq[snv_idx - 1:snv_idx + 2]
                        creates_homopolymer = 1 if (alt_base == trinuc[0] or alt_base == trinuc[2]) else 0
                        flanking_is_AT = 1 if (trinuc[0] in 'AT' and trinuc[2] in 'AT') else 0
                except Exception:
                    pass

            is_transition = 1 if (ref_base, alt_base) in {('A','G'),('G','A'),('C','T'),('T','C')} else 0

            rows.append({
                'depth': depth,
                'alt_count': alt_ct,
                'het_prob': float(row.het_prob),
                'mean_mapq': float(np.mean(mapqs)) if mapqs else 0.0,
                'mean_bq_alt': float(np.mean(alt_bqs)) if alt_bqs else 0.0,
                'mean_bq_ref': float(np.mean(ref_bqs)) if ref_bqs else 0.0,
                'n_distinct_alt': sum(1 for c in alt_base_counts.values() if c >= 2),
                'alt_pos_on_read_mean': float(np.mean(alt_positions)) if alt_positions else 0.5,
                'alt_pos_on_read_std': float(np.std(alt_positions)) if len(alt_positions) > 1 else 0.0,
                'strand_sor': strand_sor,
                'del_frac': del_frac,
                'gc_content_11bp': gc_content_11bp,
                'homopolymer_len': homopolymer_len,
                'is_homopolymer_ge5': is_homopolymer_ge5,
                'creates_homopolymer': creates_homopolymer,
                'flanking_is_AT': flanking_is_AT,
                'is_transition': is_transition,
            })
        return pd.DataFrame(rows, index=df_pileup_filtered.index, columns=SNV_CLF_FEATURE_COLUMNS)

    @staticmethod
    def heterozygous_prob(depth, n_alt, e=0.01):
        priors = (1 / 3, 1 / 3, 1 / 3)
        log_binom = math.lgamma(depth + 1) - math.lgamma(n_alt + 1) - math.lgamma(depth - n_alt + 1)
        log_p_het = log_binom + depth * math.log(0.5)
        log_p_hom_ref = log_binom + n_alt * math.log(e) + (depth - n_alt) * math.log(1 - e)
        log_p_hom_alt = log_binom + n_alt * math.log(1 - e) + (depth - n_alt) * math.log(e)
        log_post = [math.log(priors[0]) + log_p_het, math.log(priors[1]) + log_p_hom_ref, math.log(priors[2]) + log_p_hom_alt]
        m = max(log_post)
        den = sum(math.exp(x - m) for x in log_post)
        return math.exp(log_post[0] - m) / den
    @staticmethod
    def heterozygous_prob_vec(depth, n_alt, e=0.01, priors=None, het_beta=None):
        if priors is None:
            priors = (1 / 3, 1 / 3, 1 / 3)
        depth = np.asarray(depth, dtype=np.int64)
        n_alt = np.asarray(n_alt, dtype=np.int64)
        log_binom = gammaln(depth + 1) - gammaln(n_alt + 1) - gammaln(depth - n_alt + 1)
        if het_beta is None:
            log_p_het = log_binom + depth * np.log(0.5)
        else:
            _a, _b = float(het_beta[0]), float(het_beta[1])
            log_p_het = (log_binom
                         + betaln(n_alt + _a, depth - n_alt + _b)
                         - betaln(_a, _b))
        log_p_hom_ref = log_binom + n_alt * np.log(e) + (depth - n_alt) * np.log(1 - e)
        log_p_hom_alt = log_binom + n_alt * np.log(1 - e) + (depth - n_alt) * np.log(e)
        log_post = np.stack([
            np.log(priors[0]) + log_p_het,
            np.log(priors[1]) + log_p_hom_ref,
            np.log(priors[2]) + log_p_hom_alt,
        ])
        m = np.max(log_post, axis=0)
        den = np.exp(log_post - m).sum(axis=0)
        return np.exp(log_post[0] - m) / den
    @staticmethod
    def heterozygous_prob_per_read_bq(site_reads_dict, df_pileup_filtered,
                                      max_baseq=None, het_beta=None,
                                      priors=(0.6, 0.1, 0.3)):
        e_floor = 1e-3 if max_baseq is None else 10.0 ** (-float(max_baseq) / 10.0)


        log_priors = np.log(np.asarray(priors, dtype=float))
        out = np.empty(len(df_pileup_filtered), dtype=float)


        _use_beta = het_beta is not None
        if _use_beta:
            _ba, _bb = float(het_beta[0]), float(het_beta[1])
            _x, _w = np.polynomial.legendre.leggauss(48)
            _pgrid, _wgrid = 0.5 * (_x + 1), 0.5 * _w
            _lprior = ((_ba - 1) * np.log(_pgrid) + (_bb - 1) * np.log(1 - _pgrid)
                       - betaln(_ba, _bb) + np.log(_wgrid))
        for i, row in enumerate(df_pileup_filtered.itertuples(index=False)):
            chrom, pos0 = str(row.chrom), int(row.pos)
            ref_base = str(row.ref).upper()
            alt_base = str(row.alt).upper()
            tuples = site_reads_dict.get((chrom, pos0), set())
            log_post = log_priors.copy()


            lp_het = np.zeros_like(_pgrid) if _use_beta else None
            n_valid = 0
            for _read_name, _qpos, base, baseq, _mapq, _read_len, _is_reverse in tuples:
                base = str(base).upper()
                if base not in {'A', 'C', 'G', 'T'}:
                    continue


                if baseq is None:
                    e = 0.25
                else:


                    e = float(np.clip(10.0 ** (-int(baseq) / 10.0), e_floor, 0.5))
                p_match_het = 0.5 * (1.0 - e) + 0.5 * (e / 3.0)
                p_err = e / 3.0
                if base == ref_base:
                    log_post[0] += np.log(1.0 - e)
                    log_post[1] += np.log(p_match_het)
                    log_post[2] += np.log(p_err)
                    if _use_beta:
                        lp_het += np.log((1.0 - _pgrid) * (1.0 - e) + _pgrid * p_err)
                elif base == alt_base:
                    log_post[0] += np.log(p_err)
                    log_post[1] += np.log(p_match_het)
                    log_post[2] += np.log(1.0 - e)
                    if _use_beta:
                        lp_het += np.log(_pgrid * (1.0 - e) + (1.0 - _pgrid) * p_err)
                else:
                    log_post += np.log(p_err)
                    if _use_beta:
                        lp_het += np.log(p_err)
                n_valid += 1
            if n_valid == 0:
                out[i] = Haplotyping.heterozygous_prob(int(row.depth), int(row.alt_count), e=0.01)
                continue
            if _use_beta:
                _z = lp_het + _lprior
                _mz = _z.max()
                log_post[1] = log_priors[1] + _mz + np.log(np.exp(_z - _mz).sum())
            m = np.max(log_post)
            den = np.exp(log_post - m).sum()
            out[i] = np.exp(log_post[1] - m) / den
        return out
    @staticmethod
    def _extract_gene_id_from_isoform_filename(filename):
        match = re.search(r'_(ENSG[^_]+)_(?:isoform_agg(?:_balance|_unbalance|_extrap|_pmax)?)\.csv$', os.path.basename(filename))
        if match is None:
            raise ValueError(f'Could not extract geneID from filename: {filename}')
        return match.group(1)
    def _iter_sample_indices(self):
        if self.job_array_by_sample:
            if self.job_index < 0 or self.job_index >= self.n_samples:
                raise ValueError(f'job_index {self.job_index} out of range for {self.n_samples} samples.')
            return [self.job_index]
        return range(self.n_samples)
    def _should_parallelize(self, items):
        return self.n_workers != 1 and len(items) > 1
    def _load_isoform_agg_frame(self, file_path):
        df = pd.read_csv(file_path, index_col=0)
        df['geneID'] = self._extract_gene_id_from_isoform_filename(file_path)
        return df


    @staticmethod
    def _stream_concat_gene_csvs(entries, out_path, drop_first_col):
        if not entries:
            return False
        for _, gene_name, gene_id in entries:
            if any(ch in gene_name or ch in gene_id for ch in (',', '"', '\n', '\r')):
                return False
        first_header = None
        for path, _, _ in entries:
            with open(path) as fh:
                header = fh.readline().rstrip('\r\n')
            if first_header is None:
                first_header = header
            elif header != first_header:
                return False
        if not first_header or '"' in first_header or first_header.startswith('\ufeff'):
            return False
        fields = first_header.split(',')
        n_fields = len(fields)
        if drop_first_col:
            out_fields = fields[1:]
        else:
            out_fields = list(fields)
            if out_fields and out_fields[0] == '':

                out_fields[0] = 'Unnamed: 0'
        if '' in out_fields or len(set(out_fields)) != len(out_fields):
            return False
        if any(c.endswith('_x') or c.endswith('_y') for c in out_fields):
            return False
        if 'geneName' in out_fields or 'geneID' in out_fields:
            return False
        tmp_path = out_path + '.stream_tmp'
        counter = 0
        try:
            with open(tmp_path, 'w') as out:
                out.write(',' + ','.join(out_fields) + ',geneName,geneID\n')
                for path, gene_name, gene_id in entries:
                    with open(path) as fh:
                        text = fh.read()
                    if '"' in text:
                        return False
                    for line in text.splitlines()[1:]:
                        if not line.strip():
                            continue
                        if line.count(',') != n_fields - 1:
                            return False
                        if drop_first_col:
                            line = line[line.find(',') + 1:]
                        out.write(f'{counter},{line},{gene_name},{gene_id}\n')
                        counter += 1
            os.replace(tmp_path, out_path)
            return True
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    def _stream_isoform_agg_csv(self, file_paths, out_csv):
        gene_ids = [self._extract_gene_id_from_isoform_filename(p) for p in file_paths]
        if any(ch in g for g in gene_ids for ch in (',', '"', '\n', '\r')):
            return False
        tmp_path = out_csv + '.stream_tmp'
        wrote_header = False
        try:
            with open(tmp_path, 'w') as out:
                for path, gene_id in zip(file_paths, gene_ids):
                    with open(path) as fh:
                        text = fh.read()
                    if '"' in text:
                        return False
                    lines = text.splitlines()
                    if not lines:
                        return False
                    header = lines[0]
                    if not header.strip() or header.startswith('\ufeff'):
                        return False
                    hfields = header.split(',')


                    data_fields = hfields[1:]
                    if ('geneID' in hfields or '' in data_fields
                            or len(set(data_fields)) != len(data_fields)):
                        return False
                    n_commas = header.count(',')
                    if not wrote_header:
                        out.write(header + ',geneID\n')
                        wrote_header = True
                    for line in lines[1:]:
                        if not line.strip():
                            continue
                        if line.count(',') != n_commas:
                            return False
                        out.write(f'{line},{gene_id}\n')
            os.replace(tmp_path, out_csv)
            return True
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    def _append_isoform_agg_pandas(self, file_paths, out_csv):


        if self._should_parallelize(file_paths):
            frames = Parallel(n_jobs=self.n_workers, prefer='threads')(
                delayed(self._load_isoform_agg_frame)(file_path)
                for file_path in file_paths
            )
        else:
            frames = [self._load_isoform_agg_frame(file_path) for file_path in file_paths]
        for idx, df in enumerate(frames):
            df.to_csv(out_csv, mode='w' if idx == 0 else 'a', header=(idx == 0))
    def _append_isoform_agg_csv(self, file_paths, out_csv):
        if not file_paths:
            return
        if self._stream_isoform_agg_csv(file_paths, out_csv):
            return
        self._append_isoform_agg_pandas(file_paths, out_csv)
    def _collect_count_triples(self, hap_files):


        if not hap_files:
            return [], []
        if self._should_parallelize(hap_files):
            results = Parallel(n_jobs=self.n_workers, prefer='threads')(
                delayed(self._process_count_gene)(hap_file)
                for hap_file in hap_files
            )
        else:
            results = [self._process_count_gene(hap_file) for hap_file in hap_files]
        triple_transcript_list, triple_gene_list = [], []
        for triple_isoform, triple_gene in results:
            triple_transcript_list.extend(triple_isoform)
            triple_gene_list.extend(triple_gene)
        return triple_transcript_list, triple_gene_list
    def bulk_lrt_allelic_balance_gene(self, df_r, df_pi, em_result,
                                      positions=None):
        reads_keep_mask = em_result["reads_keep_mask"]


        alpha_hat = float(np.mean(em_result["hat_I"][reads_keep_mask]))
        h_A_hat, h_m_hat = np.asarray(em_result["h_A"]).reshape(-1), np.asarray(em_result["h_m"]).reshape(-1)
        ll_alt = observed_loglikelihood(df_r=df_r.loc[reads_keep_mask], df_pi=df_pi.loc[reads_keep_mask],
                                        alpha=alpha_hat, h_A=h_A_hat, h_m=h_m_hat,
                                        kept_mask=em_result['kept_mask'])


        fit_null = run_em_fixed_alpha(df_r=df_r, df_pi=df_pi,
                                      alpha_fixed=0.5, max_iter=self.max_iter, tol=self.tol,
                                      verbose=self.verbose, seed=self.seed,
                                      heterozygous_priors=(0.4, 0.2, 0.4),
                                      heterozygous_coverage_factor=self.heterozygous_coverage_factor,
                                      het_beta=self.het_beta,
                                      shrink_denominator=self.shrink_denominator,


                                      h_m_init=em_result['h_m_init_used'],
                                      h_A_init=em_result['h_A_init_used'],
                                      kept_mask=em_result['kept_mask'])


        if self.phase_flip and positions is not None and len(positions) > 1:


            fit_null['reads_keep_mask'] = em_result['reads_keep_mask']
            fit_null, _ = guarded_switch_flip(df_r, df_pi, fit_null,
                                              np.asarray(positions))
        ll_null = observed_loglikelihood(df_r=df_r.loc[reads_keep_mask], df_pi=df_pi.loc[reads_keep_mask],
                                         alpha=0.5, h_A=fit_null["h_A"], h_m=fit_null["h_m"],
                                         kept_mask=em_result['kept_mask'])

        lrt_stat = max(0.0, 2.0 * (ll_alt - ll_null))
        p_value = chi2.sf(lrt_stat, df=1)
        out = {"alpha_hat": min(alpha_hat, 1 - alpha_hat),
               "ll_alt": float(ll_alt),
               "ll_null": float(ll_null),
               "lrt_stat": float(lrt_stat),
               "p_value": float(p_value)}
        return out
    def bulk_lrt_orientation_robust(self, df_r, df_pi, em_result,
                                    marker_blocks, read_blocks, positions=None,
                                    max_orientations=64):
        B = int(np.max(marker_blocks)) if len(marker_blocks) else 0
        out = self.bulk_lrt_allelic_balance_gene(df_r, df_pi, em_result,
                                                positions=positions)
        if B <= 1:
            out['p_value_orient_min'] = out['p_value']
            out['n_orientations'] = 1
            return out
        n_flip = B - 1
        if 2 ** n_flip > max_orientations:


            mes = (f'gene has {B} phase blocks (> {max_orientations} '
                   'orientations): orientation-robust ASE test not '
                   'evaluated (p_value = NA)')
            print(mes) if self.logger is None else self.logger.warning(mes)
            out.update({'ll_alt': None, 'lrt_stat': None, 'p_value': None,
                        'p_value_orient_min': None, 'n_orientations': 0})
            return out
        reads_keep = em_result['reads_keep_mask']
        kept = em_result['kept_mask']
        h_m_hat = np.asarray(em_result['h_m']).reshape(-1)
        ll_null = out['ll_null']
        df_r_keep = df_r.loc[reads_keep]
        df_pi_keep = df_pi.loc[reads_keep]

        def profile_ll(h_A_o):


            res = minimize_scalar(
                lambda a: -observed_loglikelihood(
                    df_r_keep, df_pi_keep, alpha=a, h_A=h_A_o, h_m=h_m_hat,
                    kept_mask=kept),
                bounds=(0.0, 1.0), method='bounded')
            return -float(res.fun)

        p_list, ll_list = [], []
        for code in range(2 ** n_flip):
            flip_blocks = {b + 2 for b in range(n_flip) if (code >> b) & 1}
            h_A_o = np.asarray(em_result['h_A'], dtype=float).copy()
            if flip_blocks:
                m_flip = np.isin(marker_blocks, list(flip_blocks))
                h_A_o[m_flip] = 1.0 - h_A_o[m_flip]
            ll_alt_o = profile_ll(h_A_o)
            lrt_o = max(0.0, 2.0 * (ll_alt_o - ll_null))
            p_list.append(float(chi2.sf(lrt_o, df=1)))
            ll_list.append(ll_alt_o)
        p_max = max(p_list)
        i_max = p_list.index(p_max)
        out['p_value'] = p_max
        out['ll_alt'] = ll_list[i_max]
        out['lrt_stat'] = max(0.0, 2.0 * (ll_list[i_max] - ll_null))
        out['p_value_orient_min'] = min(p_list)
        out['n_orientations'] = len(p_list)
        return out

    def ct_lrt_allelic_balance_gene(self, df_r, df_pi, em_result):
        reads_keep_mask = em_result["reads_keep_mask"]


        alpha_hat = float(np.mean(em_result["hat_I"][reads_keep_mask]))
        h_A_hat, h_m_hat = np.asarray(em_result["h_A"]).reshape(-1), np.asarray(em_result["h_m"]).reshape(-1)
        ll_alt = observed_loglikelihood(df_r=df_r.loc[reads_keep_mask], df_pi=df_pi.loc[reads_keep_mask],
                                        alpha=alpha_hat, h_A=h_A_hat, h_m=h_m_hat,
                                        kept_mask=em_result['kept_mask'])
        ll_null = observed_loglikelihood(df_r=df_r.loc[reads_keep_mask], df_pi=df_pi.loc[reads_keep_mask],
                                         alpha=0.5, h_A=h_A_hat, h_m=h_m_hat,
                                         kept_mask=em_result['kept_mask'])

        lrt_stat = max(0.0, 2.0 * (ll_alt - ll_null))
        p_value = chi2.sf(lrt_stat, df=1)
        out = {"alpha_hat": min(alpha_hat, 1 - alpha_hat),
               "ll_alt": float(ll_alt),
               "ll_null": float(ll_null),
               "lrt_stat": float(lrt_stat),
               "p_value": float(p_value)}
        return out


    @property
    def geneStructureInformation(self):
        return self._gene_structure

    @geneStructureInformation.setter
    def geneStructureInformation(self, value):


        self._gene_structure = value
        self._gsi_from_disk = False

    def _rebuildable(self, kind):
        if kind == 'gsi':
            if not getattr(self, '_gsi_from_disk', False):
                return None


            if len(self._gene_structure) != getattr(self, '_gsi_len', None):
                return None
        path = getattr(self, f'_{kind}_source_path', None)


        if not path or not os.path.exists(path):
            return None
        return getattr(self, f'_{kind}_digest', None)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_fasta_lock'] = None
        state['fasta_handle'] = None


        state['_gsi_stripped'] = self._rebuildable('gsi')
        if state['_gsi_stripped']:
            state['_gene_structure'] = None
        state['_clf_stripped'] = (self._rebuildable('clf')
                                  if self.snv_classifier_model is not None else None)
        if state['_clf_stripped']:
            state['snv_classifier_model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fasta_lock = threading.Lock()
        self.fasta_handle = (pysam.FastaFile(self.ref_fasta_path)
                             if getattr(self, 'ref_fasta_path', None)
                             else None)
        def _load_gsi(path, digest=None):
            raw, _ = _load_asset_bytes(path, digest, 'gene structure')
            return _gsi_from_bytes(raw, path, None)

        def _load_clf(path, digest=None):
            raw, _ = _load_asset_bytes(path, digest, 'classifier')
            return joblib_load(io.BytesIO(raw))

        for kind, attr, loader in (('gsi', '_gene_structure', _load_gsi),
                                   ('clf', 'snv_classifier_model', _load_clf)):
            digest = state.get(f'_{kind}_stripped')
            if not digest:
                continue
            path = getattr(self, f'_{kind}_source_path')
            blob = _worker_cached_asset(kind, path, digest, loader)
            if blob is None:
                raise RuntimeError(f'failed to rebuild {kind} from {path} in worker')
            object.__setattr__(self, attr, blob)

    def _load_gene_index_hap(self):
        idx_list, cols_list = [], []
        needed = ('Read', 'geneName', 'geneID', 'Isoform', 'Cell', 'Umi',
                  'Keep')
        for tsv_path in self.read_isoform_mapping_path_list:
            idx_path = tsv_path + '.geneidx.tsv'
            if not os.path.exists(idx_path):
                return None, None
            idx = {}
            with open(idx_path) as f:
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 4:
                        continue
                    idx[parts[0]] = (int(parts[1]), int(parts[2]),
                                     int(parts[3]))
            with open(tsv_path) as f:
                header = f.readline().rstrip('\r\n').split('\t')
            cols = {name: i for i, name in enumerate(header)}
            if any(c not in cols for c in needed):
                return None, None
            idx_list.append(idx)
            cols_list.append(cols)
        return idx_list, cols_list

    def _gene_mapping_df(self, sample_index, geneID):
        if self.mapping_df_dict_list is not None:
            return self.mapping_df_dict_list[sample_index][geneID]
        cache_key = (sample_index, geneID)
        cached = self._block_df_cache.get(cache_key)
        if cached is not None:
            return cached
        idx = self._hap_gene_index[sample_index].get(geneID)
        if idx is None:
            raise KeyError(geneID)
        offset, length, _ = idx
        cols = self._hap_gene_cols[sample_index]
        with open(self.read_isoform_mapping_path_list[sample_index],
                  'rb') as f:
            f.seek(offset)
            block = f.read(length)
        rows = []
        order = [cols[c] for c in ('Read', 'geneName', 'geneID', 'Isoform',
                                   'Cell', 'Umi')]
        keep_i = cols['Keep']
        for line in block.split(b'\n'):
            if not line:
                continue
            fields = line.split(b'\t')
            if _keep_is_one_bytes(fields[keep_i]):
                vals = [fields[i].decode() for i in order]
                vals[0] = canonicalize_read_name(vals[0])
                rows.append(vals)
        df = pd.DataFrame(rows, columns=['Read', 'geneName', 'geneID',
                                         'Isoform', 'Cell', 'Umi'])
        if len(self._block_df_cache) > 8:
            self._block_df_cache.clear()
        self._block_df_cache[cache_key] = df
        return df

    def _read_mapping(self):
        mapping_df_dict_list = []
        for read_isoform_mapping_path in self.read_isoform_mapping_path_list:
            pieces = defaultdict(list)
            for chunk in pd.read_csv(read_isoform_mapping_path, sep='\t', chunksize=100000):
                chunk = chunk[chunk['Keep'] == 1][['Read', 'geneName', 'geneID', 'Isoform', 'Cell', 'Umi']]
                chunk['Read'] = chunk['Read'].map(canonicalize_read_name)
                for gene_id, sub_df in chunk.groupby('geneID', sort=False):
                    pieces[gene_id].append(sub_df)
                del chunk
            mapping_df_dict = {
                gene_id: pd.concat(parts, ignore_index=True)
                for gene_id, parts in pieces.items()
            }
            mapping_df_dict_list.append(mapping_df_dict)
            del pieces
            del mapping_df_dict
        return mapping_df_dict_list
    def _get_gene_name(self, geneID):
        if self.mapping_df_dict_list is None:
            for i in range(len(self.read_isoform_mapping_path_list)):
                try:
                    df = self._gene_mapping_df(i, geneID)
                except KeyError:
                    continue
                if not df.empty:
                    return df['geneName'].iloc[0]
            return None
        for mapping_df_dict in self.mapping_df_dict_list:
            if geneID in mapping_df_dict:
                mapping_df_gene = mapping_df_dict[geneID]
                if not mapping_df_gene.empty:
                    return mapping_df_gene['geneName'].iloc[0]
        return None
    @staticmethod
    @staticmethod
    @staticmethod
    def _hapA_share_unfolded(read_hap_df):
        if read_hap_df is None or 'hat_I' not in read_hap_df.columns:
            return None
        ph = read_hap_df[read_hap_df['reads_phasable'] == 1] \
            if 'reads_phasable' in read_hap_df.columns else read_hap_df
        vals = pd.to_numeric(ph['hat_I'], errors='coerce').dropna()
        return float(vals.mean()) if len(vals) else None

    @staticmethod
    def _extrapolate_isoform_counts(matA, matB, matA_ph, matB_ph, alpha_hat):
        a = matA_ph.sum(axis=0)
        b = matB_ph.sum(axis=0)
        a.index = [c.replace('_hapA', '') for c in matA_ph.columns]
        b.index = [c.replace('_hapB', '') for c in matB_ph.columns]
        allA = matA.sum(axis=0)
        allB = matB.sum(axis=0)
        allA.index = [c.replace('_hapA', '') for c in matA.columns]
        allB.index = [c.replace('_hapB', '') for c in matB.columns]


        isos = allA.index
        R = allA.reindex(isos, fill_value=0.0) + allB.reindex(isos, fill_value=0.0)
        a = a.reindex(isos, fill_value=0.0)
        b = b.reindex(isos, fill_value=0.0)
        ph = a + b
        fracA = pd.Series(np.where(ph > 0, a / ph.replace(0, np.nan), alpha_hat),
                          index=isos)
        A = R * fracA
        B = R * (1.0 - fracA)
        outA = pd.DataFrame([A.values], columns=[i + '_hapA' for i in isos])
        outB = pd.DataFrame([B.values], columns=[i + '_hapB' for i in isos])
        return outA, outB

    @staticmethod
    def _aggregate_isoform_hap_table(matA, matB, min_frac=0.10, drop_uncategorized=True, group_novel = False):
        sumA = matA.sum(axis=0)
        sumB = matB.sum(axis=0)
        cols = pd.Index([c.replace('_hapA', '') for c in matA.columns])
        if drop_uncategorized:
            keep_mask = ~cols.str.contains("uncategorized", case=False, regex=False)
            cols = cols[keep_mask]
            sumA = sumA[keep_mask]
            sumB = sumB[keep_mask]
        if len(cols) == 0:
            return pd.DataFrame(columns=["hapA", "hapB"], dtype=float)
        sumA.index = cols
        sumB.index = cols
        if group_novel:
            novel_cols = [c for c in cols if "novel" in c.lower()]
            if len(novel_cols) > 0:
                base = cols[0].split("_")[0]
                novel_label = f"{base}_Novel"
                sumA.loc[novel_label] = float(sumA.loc[novel_cols].sum())
                sumB.loc[novel_label] = float(sumB.loc[novel_cols].sum())
                cols = [c for c in cols if c not in novel_cols] + [novel_label]
        sumA = sumA.loc[cols]
        sumB = sumB.loc[cols]
        totA = float(sumA.sum())
        totB = float(sumB.sum())
        fracA = sumA / totA
        fracB = sumB / totB
        keep_mask = (fracA >= float(min_frac)) | (fracB >= float(min_frac))
        majors = [c for c, k in zip(cols, keep_mask) if k]
        minors = [c for c, k in zip(cols, keep_mask) if not k]
        O = pd.DataFrame({
            "hapA": sumA.loc[majors],
            "hapB": sumB.loc[majors],
        })
        if len(minors) > 0:
            otherA = float(sumA.loc[minors].sum())
            otherB = float(sumB.loc[minors].sum())
            O.loc[cols[0].split('_')[0] + '_Other'] = [otherA, otherB]
        return O
    @staticmethod
    def _phasability_decompose(tab, tab_balance):


        _extra = set(tab.index) - set(tab_balance.index)
        if _extra:
            raise _PhasabilityPartitionMismatch(
                f'{sorted(_extra)} appear among the phasable reads but not '
                f'among all reads, which is impossible unless the two tables '
                f'were partitioned differently')
        _o1 = {i for i in tab.index if str(i).endswith('_Other')}
        _o2 = {i for i in tab_balance.index if str(i).endswith('_Other')}
        if bool(_o1) != bool(_o2):
            raise _PhasabilityPartitionMismatch(
                f'one table merged rare isoforms into {sorted(_o1 or _o2)} and '
                f'the other did not, so that row holds different isoforms on '
                f'the two sides and their totals cannot be subtracted')
        isos = list(dict.fromkeys(list(tab.index) + list(tab_balance.index)))
        a = np.array([float(tab.at[i, 'hapA']) if i in tab.index else 0.0 for i in isos])
        b = np.array([float(tab.at[i, 'hapB']) if i in tab.index else 0.0 for i in isos])
        R = np.array([float(tab_balance.loc[i, ['hapA', 'hapB']].sum())
                      if i in tab_balance.index else 0.0 for i in isos])
        if np.any(a + b > R + 1e-6):
            bad = [isos[i] for i in np.where(a + b > R + 1e-6)[0]]
            raise _PhasabilityPartitionMismatch(
                f'{bad} carry more phasable reads than they have reads in '
                f'total, so the two tables are not counting the same isoform')
        n = np.maximum(R - (a + b), 0.0)
        return isos, a, b, n, (a + b + n)

    @staticmethod
    def _chi2_of(a_col, b_col):
        O = np.stack([np.asarray(a_col, float), np.asarray(b_col, float)], axis=1)
        R, C, N = O.sum(1), O.sum(0), O.sum()
        if N <= 0 or C.min() <= 0 or (R > 0).sum() < 2:
            return 0.0
        E = np.outer(R, C) / N
        with np.errstate(divide='ignore', invalid='ignore'):
            return float(np.nansum(np.where(E > 0, (O - E) ** 2 / E, 0.0)))

    @classmethod
    def _tab_max_p(cls, tab, tab_balance):
        try:
            isos, a, b, n, R = cls._phasability_decompose(tab, tab_balance)
        except _PhasabilityPartitionMismatch as e:
            print(f'[ASTU] max-p bound falls back to the proportional table: {e}')
            return tab_balance.copy().astype(float)
        out = tab_balance.copy().astype(float)
        pos = R > 0
        if pos.sum() < 2:
            return out
        lo = float(np.max((a[pos]) / R[pos]))
        hi = float(np.min((a[pos] + n[pos]) / R[pos]))
        if lo <= hi:
            t = 0.5 * (lo + hi)
        else:

            grid = np.unique(np.concatenate([
                (a[pos]) / R[pos], (a[pos] + n[pos]) / R[pos],
                np.linspace(0.0, 1.0, 1001)]))
            best, t = None, grid[0]
            for g in grid:
                xa = np.clip(g * R - a, 0.0, n)
                v = cls._chi2_of(a + xa, b + (n - xa))
                if best is None or v < best:
                    best, t = v, g
        xa = np.clip(t * R - a, 0.0, n)
        for i, iso in enumerate(isos):
            if iso in out.index:
                out.loc[iso, ['hapA', 'hapB']] = [a[i] + xa[i], b[i] + (n[i] - xa[i])]
        return out


    _MINP_MAX_ISOFORMS = 16

    @classmethod
    def _tab_min_p(cls, tab, tab_balance, logger=None):
        import itertools

        try:
            isos, a, b, n, R = cls._phasability_decompose(tab, tab_balance)
        except _PhasabilityPartitionMismatch as e:
            mes = f'[ASTU] min-p bound falls back to the legacy greedy: {e}'
            print(mes) if logger is None else logger.warning(mes)
            return cls._most_imbalanced_tab(tab, tab_balance)
        movable = [i for i in range(len(isos)) if n[i] > 0]
        if len(movable) > cls._MINP_MAX_ISOFORMS:


            mes = (f'[ASTU] {len(movable)} isoforms carry unphasable reads, over '
                   f'the {cls._MINP_MAX_ISOFORMS} the exact search enumerates; '
                   f'using coordinate ascent, which is feasible but not '
                   f'guaranteed optimal')
            print(mes) if logger is None else logger.warning(mes)
            xa = np.zeros(len(isos))
            for _sweep in range(8):
                moved = False
                for i in movable:
                    cur = xa[i]
                    lo = cls._chi2_of(a + np.where(np.arange(len(isos)) == i, 0.0, xa),
                                      b + n - np.where(np.arange(len(isos)) == i, 0.0, xa))
                    xa[i] = n[i]
                    hi = cls._chi2_of(a + xa, b + (n - xa))
                    xa[i] = 0.0 if lo >= hi else n[i]
                    moved |= (xa[i] != cur)
                if not moved:
                    break
            out = tab_balance.copy().astype(float)
            for i, iso in enumerate(isos):
                if iso in out.index:
                    out.loc[iso, ['hapA', 'hapB']] = [a[i] + xa[i],
                                                      b[i] + (n[i] - xa[i])]
            return out
        best_v, best_x = None, np.zeros(len(isos))
        for choice in itertools.product((0.0, 1.0), repeat=len(movable)):
            xa = np.zeros(len(isos))
            for j, i in enumerate(movable):
                xa[i] = n[i] * choice[j]
            v = cls._chi2_of(a + xa, b + (n - xa))
            if best_v is None or v > best_v:
                best_v, best_x = v, xa
        out = tab_balance.copy().astype(float)
        for i, iso in enumerate(isos):
            if iso in out.index:
                out.loc[iso, ['hapA', 'hapB']] = [a[i] + best_x[i],
                                                  b[i] + (n[i] - best_x[i])]
        return out

    @staticmethod
    def _most_imbalanced_tab(tab, tab_balance):
        tab_imbalanced = tab.copy().astype(float)
        def col_with_larger_total(df):
            return 'hapA' if df['hapA'].sum() >= df['hapB'].sum() else 'hapB'
        unphasable_isoforms = [iso for iso in tab_balance.index if iso not in tab.index]
        for iso in unphasable_isoforms:
            total_reads = tab_balance.loc[iso, ['hapA', 'hapB']].sum()
            if total_reads == 0:
                continue
            tab_imbalanced.loc[iso] = [0.0, 0.0]
            a, b = tab_imbalanced.at[iso, 'hapA'], tab_imbalanced.at[iso, 'hapB']
            total_A, total_B = tab_imbalanced['hapA'].sum(), tab_imbalanced['hapB'].sum()
            to_A = abs((a + total_reads) / (total_A + total_reads) - b / total_B) if total_B > 0 else float('inf')
            to_B = abs((b + total_reads) / (total_B + total_reads) - a / total_A) if total_A > 0 else float('inf')
            if to_A > to_B:
                tab_imbalanced.at[iso, 'hapA'] = a + total_reads
            elif to_B > to_A:
                tab_imbalanced.at[iso, 'hapB'] = b + total_reads
            else:
                target = col_with_larger_total(tab_imbalanced)
                tab_imbalanced.at[iso, target] += total_reads
        return tab_imbalanced
    def _chisq_or_skip(self, tab):
        if self.skip_astu_test:
            return {"chi2_isoform": None, "df_isoform": None,
                    "p_value_isoform": None}
        return self._chisq_test(tab)


    MAX_ASTU_ORIENTATIONS = 64

    @staticmethod
    def _fold_label(iso, gene_name, folded_index):
        lab = f'{gene_name}_{iso}'
        return lab if lab in folded_index else f'{gene_name}_Other'

    def _astu_block_sums(self, read_iso_df, gene_name, tab_index, bal_index):
        df = read_iso_df.copy()
        df['hat_I'] = df['hat_I'].astype(float)
        df['_flip'] = (df['reads_phasable'].astype(int) == 1) & (df['read_block'] > 0)
        df['_blk'] = np.where(df['_flip'], df['read_block'], 0)
        df['_ph'] = df['reads_phasable'].astype(int) == 1
        rows = []
        for part, idx in (('ph', tab_index), ('bal', bal_index)):
            sub = df if part == 'bal' else df[df['_ph']]
            lab = sub['Isoform'].map(lambda i: self._fold_label(i, gene_name, idx))
            g = sub.assign(_lab=lab).groupby(['_blk', '_lab'])['hat_I'].agg(['sum', 'count'])
            for (blk, label), r in g.iterrows():
                rows.append({'partition': part, 'read_block': int(blk),
                             'label': label, 'sum_hatI': float(r['sum']),
                             'n_reads': int(r['count'])})
        return pd.DataFrame(rows)

    @staticmethod
    def _tables_for_orientation(sums_df, flip_blocks):
        out = {}
        for part in ('ph', 'bal'):
            sub = sums_df[sums_df['partition'] == part]
            a = {}
            n = {}
            for r in sub.itertuples():
                s = (r.n_reads - r.sum_hatI) if r.read_block in flip_blocks else r.sum_hatI
                a[r.label] = a.get(r.label, 0.0) + s
                n[r.label] = n.get(r.label, 0.0) + r.n_reads
            labels = sorted(n)
            out[part] = pd.DataFrame(
                {'hapA': [a[l] for l in labels],
                 'hapB': [n[l] - a[l] for l in labels]}, index=labels)
        return out['ph'], out['bal']

    def _astu_orientation_pvals(self, sums_df, n_blocks):
        import itertools
        if 2 ** (n_blocks - 1) > self.MAX_ASTU_ORIENTATIONS:
            return None
        flippable = sorted(set(sums_df.loc[sums_df['read_block'] > 0, 'read_block']))[1:]
        points, highs, lows = [], [], []
        for mask in itertools.product((False, True), repeat=len(flippable)):
            flip = {b for b, f in zip(flippable, mask) if f}
            tab_o, bal_o = self._tables_for_orientation(sums_df, flip)
            out_o = self._chisq_or_skip(tab_o)
            p_pt = out_o.get('p_value_isoform')
            cands = [p_pt,
                     self._chisq_or_skip(bal_o).get('p_value_isoform'),
                     self._chisq_or_skip(self._tab_min_p(tab_o, bal_o)).get('p_value_isoform'),
                     self._chisq_or_skip(self._tab_max_p(tab_o, bal_o)).get('p_value_isoform')]
            cands = [c for c in cands if c is not None and not pd.isna(c)]
            if p_pt is not None and not pd.isna(p_pt):
                points.append(float(p_pt))
            if cands:
                highs.append(max(cands))
                lows.append(min(cands))
        if not points:
            return None
        return {'p_value_isoform': max(points),
                'p_value_isoform_orient_min': min(points),
                'p_value_isoform_high': max(highs) if highs else None,
                'p_value_isoform_low': min(lows) if lows else None}

    @staticmethod
    def _chisq_test(tab):
        out = {"chi2_isoform": None, "df_isoform": None, "p_value_isoform": None}
        O = np.asarray(tab, dtype=float)
        K = tab.shape[0]
        if K >=  2:
            row_tot = O.sum(axis=1, keepdims=True)
            col_tot = O.sum(axis=0, keepdims=True)
            grand = col_tot.sum()
            E = row_tot @ (col_tot / grand)

            chi2_stat = float(np.sum((O - E) ** 2 / (E + 1e-12)))
            df = K - 1


            p_asym = float(chi2.sf(chi2_stat, df))
            out = {"chi2_isoform": chi2_stat, "df_isoform": df, "p_value_isoform": p_asym}
        return out
    @staticmethod
    def _check_stretch(seq, min_length, snv_pos_in_seq=None, max_kmer=1):
        seq = seq.upper()
        n = len(seq)
        if n == 0 or max_kmer <= 0:
            return 0

        def has_repeat_covering_snv(unit_size, repeat_threshold):
            max_start = n - unit_size * repeat_threshold
            for start in range(max_start + 1):
                unit = seq[start:start + unit_size]
                if len(unit) < unit_size:
                    continue
                repeat_count = 1
                while start + (repeat_count + 1) * unit_size <= n:
                    next_start = start + repeat_count * unit_size
                    if seq[next_start:next_start + unit_size] != unit:
                        break
                    repeat_count += 1
                if repeat_count >= repeat_threshold:
                    repeat_end = start + repeat_count * unit_size
                    if snv_pos_in_seq is None or start <= snv_pos_in_seq < repeat_end:
                        return 1
            return 0

        if max_kmer >= 1 and has_repeat_covering_snv(1, min_length):
            return 1
        if max_kmer >= 2 and has_repeat_covering_snv(2, 3):
            return 1
        if max_kmer >= 3 and has_repeat_covering_snv(3, 3):
            return 1
        return 0

    @staticmethod
    def _load_rna_editing_db(path):

        if path is None or str(path).strip().lower() in ('', 'none'):
            return None
        with np.load(path, allow_pickle=True) as data:
            db = {
                'AG': {
                    key.split('AG__', 1)[1]: np.asarray(data[key], dtype=np.uint32)
                    for key in data.files
                    if key.startswith('AG__')
                },
                'TC': {
                    key.split('TC__', 1)[1]: np.asarray(data[key], dtype=np.uint32)
                    for key in data.files
                    if key.startswith('TC__')
                },
            }
            if '__metadata__' in data.files:
                meta = list(data['__metadata__'])
                if 'coords=0_based' not in meta:
                    raise ValueError(
                        f'RNA editing DB {path} does not use 0-based coordinates '
                        f'(metadata: {meta}). LongAllele requires 0-based positions.'
                    )
        if not db['AG'] and not db['TC']:
            raise ValueError(
                f'RNA editing DB {path} does not contain any AG__/TC__ arrays.'
            )
        return db

    @staticmethod
    def _editing_exemption_mask(df_read_snv, snv_ids, is_editing,
                                min_shared_reads=10, min_affinity=0.95):
        snv_ids = list(snv_ids)
        n = len(snv_ids)
        exempt = np.zeros(n, dtype=bool)
        is_editing = np.asarray(is_editing, dtype=bool)
        if is_editing.shape != (n,):
            raise ValueError(f'is_editing must have shape ({n},), '
                             f'got {is_editing.shape}')
        if n < 2 or not is_editing.any() or min_affinity > 1.0:
            return exempt
        if len(set(snv_ids)) != n:


            print('[DIAG] editing exemption skipped: duplicate SNV IDs')
            return exempt
        missing = [i for i in snv_ids if i not in df_read_snv.columns]
        if missing:


            print(f'[DIAG] editing exemption skipped: {len(missing)} SNV ID(s) '
                  f'absent from the read matrix (e.g. {missing[:3]})')
            return exempt
        r = np.asarray(df_read_snv.loc[:, snv_ids])


        alt = (r == EM_ALT_CODE).astype(np.float64)
        ref = (r == EM_REF_CODE).astype(np.float64)
        s_ = alt - ref
        W = s_.T @ s_
        v_ = alt + ref
        co = v_.T @ v_
        np.fill_diagonal(W, 0)
        np.fill_diagonal(co, 0)
        outside = ~is_editing
        for j in np.flatnonzero(is_editing):
            usable = outside & (co[j] >= min_shared_reads)
            if not usable.any():
                continue
            if float((np.abs(W[j][usable]) / co[j][usable]).max()) > min_affinity:
                exempt[j] = True
        return exempt

    @staticmethod
    def _is_known_rna_editing(df_pileup_filtered, rna_editing_db):
        if rna_editing_db is None or df_pileup_filtered.empty:
            return np.zeros(len(df_pileup_filtered), dtype=bool)

        chroms = df_pileup_filtered['chrom'].astype(str).to_numpy()
        positions = df_pileup_filtered['pos'].astype(np.uint32).to_numpy()
        refs = df_pileup_filtered['ref'].astype(str).str.upper().to_numpy()
        alts = df_pileup_filtered['alt'].astype(str).str.upper().to_numpy()

        is_editing = np.zeros(len(df_pileup_filtered), dtype=bool)

        for chrom in np.unique(chroms):
            chrom_mask = chroms == chrom

            ag_positions = rna_editing_db.get('AG', {}).get(chrom)
            if ag_positions is not None and len(ag_positions) > 0:
                mask = chrom_mask & (refs == 'A') & (alts == 'G')
                if np.any(mask):
                    query_positions = positions[mask]
                    idx = np.searchsorted(ag_positions, query_positions)


                    probe = np.minimum(idx, len(ag_positions) - 1)
                    hits = (idx < len(ag_positions)) & (ag_positions[probe] == query_positions)
                    is_editing[np.flatnonzero(mask)] = hits

            tc_positions = rna_editing_db.get('TC', {}).get(chrom)
            if tc_positions is not None and len(tc_positions) > 0:
                mask = chrom_mask & (refs == 'T') & (alts == 'C')
                if np.any(mask):
                    query_positions = positions[mask]
                    idx = np.searchsorted(tc_positions, query_positions)


                    probe = np.minimum(idx, len(tc_positions) - 1)
                    hits = (idx < len(tc_positions)) & (tc_positions[probe] == query_positions)
                    is_editing[np.flatnonzero(mask)] = hits

        return is_editing

    def _check_classifier_het_scale(self, model_path, het_beta, max_baseq=None):
        import json
        want = 'fixed_p0.5' if het_beta is None else \
            f'beta_{float(het_beta[0]):g}_{float(het_beta[1]):g}'
        meta_path = str(model_path) + '.meta.json'
        got = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as fh:
                    _meta = json.load(fh)
                got = _meta.get('het_prob_scale')


                _mb = _meta.get('max_baseq')
                if _mb is not None and max_baseq is not None and \
                        float(_mb) != float(max_baseq):
                    raise ValueError(
                        f'classifier/max_baseq mismatch: the model was trained '
                        f'with max_baseq={_mb} and this run uses '
                        f'{max_baseq}. That is the error floor, so it '
                        f'shifts het_prob -- the feature this check protects.')
            except Exception as e:
                raise ValueError(f'classifier metadata {meta_path} exists but '
                                 f'could not be read ({e}); refusing rather '
                                 f'than assuming the scale')
        elif 'clfbeta' in os.path.basename(str(model_path)):
            got = 'beta_2_2'
        if got is None:
            mes = (f'[WARN] {os.path.basename(str(model_path))} has no '
                   f'.meta.json and no clfbeta in its name, so its het_prob '
                   f'scale cannot be checked against --het_beta ({want}). '
                   f'Pairing is on you.')
            print(mes) if self.logger is None else self.logger.warning(mes)
            return
        if got != want:
            raise ValueError(
                f'classifier/het_beta mismatch: the model was trained on '
                f'het_prob scale {got!r} but this run computes {want!r}. '
                f'het_prob is one of its 17 features and the two scales differ '
                f'by tens of orders of magnitude, so its scores would be '
                f'meaningless. Pass the matching model, or the matching '
                f'--het_beta.')
        mes = f'classifier het_prob scale {got} matches --het_beta'
        print(mes) if self.logger is None else self.logger.info(mes)

    def _snv_confidence_index(self):
        if getattr(self, '_snv_conf_idx', None) is not None:
            return self._snv_conf_idx
        df = self.snv_confidence
        key = next((c for c in ('gene_id', 'geneID') if c in df.columns), None)
        self._snv_conf_key = key or 'chrom'
        self._snv_conf_idx = {k: v for k, v in df.groupby(self._snv_conf_key)}
        return self._snv_conf_idx

    def _snv_confidence_shortfall(self, geneID, df_pileup_filtered, merge_keys):
        idx = self._snv_confidence_index()
        if self._snv_conf_key in ('gene_id', 'geneID'):
            sub, scope = idx.get(geneID), 'this gene'
        else:
            chroms = set(df_pileup_filtered['chrom'].astype(str))
            parts = [v for k, v in idx.items() if str(k) in chroms]
            sub = pd.concat(parts) if parts else None
            scope = ('this chromosome -- the table has no gene column, so this '
                     'is NOT a per-gene count')
        if sub is None or not len(sub):
            return 0, 0, 0, scope
        want = set(map(tuple, sub.loc[:, merge_keys].astype(str).to_numpy()))
        have = set(map(tuple, df_pileup_filtered.loc[:, merge_keys]
                       .astype(str).to_numpy()))
        return len(want), len(want & have), len(want - have), scope

    def run_em_gene(self, geneID, sample_index = None):
        em_input = self.em_input if self.n_samples==1 else os.path.join(self.em_input, self.sample_names[sample_index])
        pileup_path = os.path.join(em_input, f'{geneID}_pileup.csv')
        read_npz_path = os.path.join(em_input, f'{geneID}_read_matrices.npz')
        read_snv_path = os.path.join(em_input, f'{geneID}_read_snv.csv')
        read_pi_path = os.path.join(em_input, f'{geneID}_read_pi.csv')
        df_pileup = pd.read_csv(pileup_path)
        df_read_snv, df_read_pi = _load_em_input(read_npz_path, read_snv_path, read_pi_path)


        knob_c_blacklist = set()
        if self.high_artifact_mode and self.canonical_exons is not None:
            sidx = sample_index if sample_index is not None else 0
            n_reads_pre_c = int(df_read_snv.shape[0])
            knob_c_blacklist, _kc = compute_knob_c_blacklist(
                geneID, sidx, self.bam_path, self.canonical_exons,
                self.read_intronic_pct_max,
                candidate_reads=set(df_read_snv.index.astype(str).tolist()),
                logger=self.logger,
                bam_cache=self._bam_path_cache,
            )
            if knob_c_blacklist:
                keep_mask = ~df_read_snv.index.astype(str).isin(knob_c_blacklist)
                df_read_snv = df_read_snv.loc[keep_mask]
                df_read_pi = df_read_pi.loc[keep_mask]
            n_reads_post_c = int(df_read_snv.shape[0])
            n_dropped_c = n_reads_pre_c - n_reads_post_c


            _knob_c_msg = (
                f'[KnobC] {geneID}: {_kc["status"]}; candidates {_kc["n_candidates"]}, '
                f'met as primary alignments in the gene window {_kc["n_found"]}, '
                f'dropped {n_dropped_c} ({n_reads_pre_c} -> {n_reads_post_c}, '
                f'cutoff intronic_pct > {self.read_intronic_pct_max})'
            )
            if _kc['status'] == 'ok' and _kc['n_candidates'] > 0 and _kc['n_found'] == 0:
                _knob_c_msg += (' ⚠️ NONE of the candidates were met: read names differ between '
                                'BAM and SCOTCH, or they align only outside the gene window / '
                                'only as secondary or supplementary records')


            print(_knob_c_msg)
            if self.logger is not None:
                self.logger.info(_knob_c_msg)
            if n_reads_post_c == 0:
                _knob_c_skip = f'[KnobC] {geneID}: all reads dropped as nascent; skipping gene'
                print(_knob_c_skip)
                if self.logger is not None:
                    self.logger.info(_knob_c_skip)
                return None, None, None, None, None, None, None, None


        if self.read_sj_min > 0:
            read_blocks_path = os.path.join(self.target, 'variant_align1',
                                            'variants_by_gene',
                                            f'{geneID}_read_blocks.pkl')
            read_blocks_pkl = load_pickle(read_blocks_path)
            if isinstance(read_blocks_pkl, dict):
                n_reads_pre_d = int(df_read_snv.shape[0])
                read_index = df_read_snv.index.astype(str)

                def _meets_sj(rn):
                    entry = read_blocks_pkl.get(rn)
                    if entry is None:
                        return False
                    try:
                        _, intron_spans = entry
                    except (TypeError, ValueError):
                        return False
                    return len(intron_spans) >= self.read_sj_min
                keep_mask = read_index.map(_meets_sj).to_numpy(dtype=bool)
                df_read_snv = df_read_snv.loc[keep_mask]
                df_read_pi = df_read_pi.loc[keep_mask]
                n_reads_post_d = int(df_read_snv.shape[0])
                n_dropped_d = n_reads_pre_d - n_reads_post_d
                if n_dropped_d > 0:
                    _knob_d_msg = (
                        f'[KnobD] {geneID}: dropped {n_dropped_d} truncated reads '
                        f'({n_reads_pre_d} -> {n_reads_post_d}, '
                        f'read_sj_min={self.read_sj_min})'
                    )
                    print(_knob_d_msg)
                    if self.logger is not None:
                        self.logger.info(_knob_d_msg)
                if n_reads_post_d == 0:
                    _knob_d_skip = (
                        f'[KnobD] {geneID}: all reads dropped as truncated; skipping gene'
                    )
                    print(_knob_d_skip)
                    if self.logger is not None:
                        self.logger.info(_knob_d_skip)
                    return None, None, None, None, None, None, None, None
            else:
                _knob_d_no_pkl = (
                    f'[KnobD] {geneID}: read_blocks.pkl missing at {read_blocks_path}; '
                    f'cannot apply read_sj_min={self.read_sj_min} filter, keeping all reads '
                    f'(run --task step1_5 + --task step1_5_merge to populate)'
                )
                print(_knob_d_no_pkl)
                if self.logger is not None:
                    self.logger.warning(_knob_d_no_pkl)


        _af_ok3 = (pd.Series(True, index=df_pileup.index)
                   if self.min_alt_frac <= 0 else
                   df_pileup.alt_count >= np.ceil(self.min_alt_frac * df_pileup.depth))
        if self.snv_confidence is not None:


            df_pileup_filtered = df_pileup.reset_index(drop=True)
            _msg = (f'[genotype] {geneID}: step3 candidate gate SKIPPED for the '
                    f'{len(df_pileup)} pileup site(s) -- supplied sites are not '
                    f'subject to n_alt/depth/AF')
            _log_with_fallback(self.logger, _msg)
        else:
            df_pileup_filtered = df_pileup[
                (df_pileup.alt_count > self.n_alt) & (df_pileup.depth >= self.depth)
                & _af_ok3].reset_index(drop=True)
        _n_before = len(df_pileup); _n_after_depth = len(df_pileup_filtered)
        if len(df_pileup_filtered)==0:
            _depth_max = df_pileup['depth'].max() if len(df_pileup) > 0 else None
            _alt_max = df_pileup['alt_count'].max() if len(df_pileup) > 0 else None
            _msg = (f'[DIAG] {geneID}: {_n_before} raw → 0 after depth/alt filter '
                    f'(n_alt={self.n_alt}, depth={self.depth}, '
                    f'max_depth={_depth_max}, max_alt={_alt_max}, '
                    f'dtypes={df_pileup[["depth","alt_count"]].dtypes.to_dict()}, '
                    f'pileup={pileup_path})')
            _log_with_fallback(self.logger, _msg)


            if self.snv_confidence is not None:
                _mk = ['chrom', 'pos'] + (['ref'] if 'ref' in
                                          self.snv_confidence.columns else [])
                _n_sup, _n_hit, _n_miss, _scope = self._snv_confidence_shortfall(
                    geneID, df_pileup_filtered, _mk)
                _m2 = (f'[DIAG] {geneID}: snv_confidence — 0 pileup site(s), '
                       f'{_n_sup} supplied over {_scope}, 0 matched, '
                       f'{_n_miss} supplied key(s) with no pileup row (the gate '
                       f'left nothing for them to match)')
                _log_with_fallback(self.logger, _m2)
            return None, None, None, None, None, None, None, None
        df_pileup_filtered["ID"] = df_pileup_filtered["chrom"].astype(str) + "_" + df_pileup_filtered["pos"].astype(
            int).astype(str) + "_" + df_pileup_filtered["ref"].astype(str)


        _site_reads_dict = None
        _variants_dir = os.path.join(self.target, 'variant_align1', 'variants_by_gene')
        _site_reads_path = os.path.join(_variants_dir, f'{geneID}_site_reads.pkl')
        if os.path.exists(_site_reads_path):
            _site_reads_dict = load_pickle(_site_reads_path)


        if _site_reads_dict is not None:
            df_pileup_filtered['het_prob'] = self.heterozygous_prob_per_read_bq(
                _site_reads_dict, df_pileup_filtered, max_baseq=self.max_baseq,
                het_beta=self.het_beta)
        else:
            df_pileup_filtered['het_prob'] = self.heterozygous_prob_vec(
                df_pileup_filtered['depth'].values, df_pileup_filtered['alt_count'].values, e=0.01,
                het_beta=self.het_beta)
        if self.snv_confidence is not None:
            merge_keys = ['chrom', 'pos']
            if 'ref' in self.snv_confidence.columns:
                merge_keys.append('ref')
            snv_confidence_cols = merge_keys + [
                col for col in self.snv_confidence.columns
                if col not in merge_keys and col not in df_pileup_filtered.columns
            ]
            _n_pileup = len(df_pileup_filtered)
            _n_sup, _n_hit, _n_miss, _scope = self._snv_confidence_shortfall(
                geneID, df_pileup_filtered, merge_keys)
            df_pileup_filtered = df_pileup_filtered.merge(
                self.snv_confidence.loc[:, snv_confidence_cols], how='inner', on=merge_keys
            )


            _msg = (f'[DIAG] {geneID}: snv_confidence — {_n_pileup} pileup '
                    f'site(s), {_n_sup} supplied over {_scope}, {_n_hit} '
                    f'matched, {_n_miss} supplied key(s) with no pileup row '
                    f'(below n_alt={self.n_alt} / depth={self.depth} / '
                    f'min_alt_frac={self.min_alt_frac}, or absent from the BAM). '
                    f'Keys: {merge_keys}')
            _log_with_fallback(self.logger, _msg)
        if self.heterozygous_filter >= 0 and self.snv_confidence is None:
            _n_pre_het = len(df_pileup_filtered)

            geneInfo, exonInfo, _ = self.geneStructureInformation[geneID]
            keep_n = math.ceil(6.6 * sum([b-a for a, b in exonInfo])/1000)
            n = min(keep_n, len(df_pileup_filtered))
            if n == 0:
                df_pileup_filtered = df_pileup_filtered.iloc[0:0]
            else:


                df_pileup_filtered = df_pileup_filtered[
                    df_pileup_filtered["het_prob"] >= self.heterozygous_filter
                ].reset_index(drop=True)
            _n_post_het = len(df_pileup_filtered)
        else:
            _n_pre_het = _n_post_het = len(df_pileup_filtered)

        if (len(df_pileup_filtered) > 0 and self.snv_confidence is None
                and self.fasta_handle is not None and self.alt_stretch_len > 0):
            df_pileup_filtered = df_pileup_filtered.sort_values(by=['pos']).reset_index(drop=True)
            chrom = df_pileup_filtered.chrom[0]
            with self._fasta_lock:
                chrom_len = self.fasta_handle.get_reference_length(chrom)
                is_stretch = [0] * len(df_pileup_filtered)
                positions = df_pileup_filtered['pos'].astype(int).tolist()
                fetch_start = max(0, min(positions) - 20)
                fetch_end = min(chrom_len, max(positions) + 21)
                stretch_seq = self.fasta_handle.fetch(chrom, fetch_start, fetch_end).upper()
            for i in range(len(df_pileup_filtered)):
                pos_ = int(df_pileup_filtered.iloc[i]['pos'])
                start, end = max(0, pos_ - 20), min(chrom_len, pos_ + 21)
                rel_start = start - fetch_start
                rel_end = end - fetch_start
                if rel_start < 0 or rel_end > len(stretch_seq) or rel_start >= rel_end:
                    continue
                seq = stretch_seq[rel_start:rel_end]
                snv_pos_in_seq = pos_ - start
                is_stretch[i] = self._check_stretch(
                    seq, self.alt_stretch_len, snv_pos_in_seq,
                    max_kmer=self.repeat_filter_kmer
                )
            df_pileup_filtered['is_stretch'] = is_stretch
            df_pileup_filtered = df_pileup_filtered[
                (df_pileup_filtered['is_stretch'] == 0) | (df_pileup_filtered['alt_count'] >= self.alt_stretch_filter)]
            df_pileup_filtered = df_pileup_filtered.drop(columns=['is_stretch'])
        _n_post_stretch = len(df_pileup_filtered)

        if len(df_pileup_filtered) > 0 and self.snv_confidence is None and self.rna_editing_db is not None:
            df_pileup_filtered['is_rna_editing'] = self._is_known_rna_editing(
                df_pileup_filtered, self.rna_editing_db
            )


            if self.editing_exempt_affinity <= 1.0:
                _ex = self._editing_exemption_mask(
                    df_read_snv, df_pileup_filtered['ID'].tolist(),
                    df_pileup_filtered['is_rna_editing'].to_numpy(),
                    min_shared_reads=self.editing_exempt_min_reads,
                    min_affinity=self.editing_exempt_affinity)
                if _ex.any():
                    df_pileup_filtered.loc[_ex, 'is_rna_editing'] = False
                    _m = (f'[DIAG] {geneID}: {int(_ex.sum())} REDIportal site(s) '
                          f'exempted by co-read linkage '
                          f'(>{self.editing_exempt_affinity})')
                    _log_with_fallback(self.logger, _m)
            df_pileup_filtered = df_pileup_filtered[
                ~df_pileup_filtered['is_rna_editing']
            ].reset_index(drop=True)
            df_pileup_filtered = df_pileup_filtered.drop(columns=['is_rna_editing'])
        _n_post_editing = len(df_pileup_filtered)


        if len(df_pileup_filtered) >= self.var_cluster_n and self.snv_confidence is None:
            pos_arr = df_pileup_filtered['pos'].values
            n = len(pos_arr)
            k = self.var_cluster_n
            is_clustered = np.zeros(n, dtype=np.int8)
            dists = pos_arr[k - 1:] - pos_arr[:n - k + 1]
            cluster_starts = np.where(dists <= self.var_cluster_window)[0]
            for start in cluster_starts:
                is_clustered[start:start + k] = 1
            df_pileup_filtered['is_clustered'] = is_clustered
            df_pileup_filtered = df_pileup_filtered[(df_pileup_filtered['is_clustered'] == 0) | (
                        df_pileup_filtered['alt_count'] >= self.alt_cluster_filter)]
            df_pileup_filtered = df_pileup_filtered.drop(columns=['is_clustered'])
        _n_post_cluster = len(df_pileup_filtered)
        _msg = (f'[DIAG] {geneID}: {_n_before} raw → {_n_after_depth} depth/alt(n_alt={self.n_alt},depth={self.depth}) → '
                f'{_n_post_het} het → {_n_post_stretch} stretch → '
                f'{_n_post_editing} editing → {_n_post_cluster} cluster')
        _log_with_fallback(self.logger, _msg)


        if (self.high_artifact_mode
                and self.nascent_leak_intervals is not None
                and self.snv_confidence is None
                and len(df_pileup_filtered) > 0):
            leak = self.nascent_leak_intervals.get(geneID)
            if leak is not None:
                intron_filled_pct, novel_intervals = leak
                if intron_filled_pct > self.novel_exon_pct_max and novel_intervals:
                    n_snvs_pre_b = len(df_pileup_filtered)
                    positions = df_pileup_filtered['pos'].astype(int).to_numpy()
                    in_novel = np.zeros(positions.shape[0], dtype=bool)
                    for s, e in novel_intervals:
                        in_novel |= (positions >= s) & (positions < e)
                    df_pileup_filtered = df_pileup_filtered[~in_novel].reset_index(drop=True)
                    n_snvs_post_b = len(df_pileup_filtered)
                    _knob_b_msg = (
                        f'[KnobB] {geneID}: intron_filled_pct={intron_filled_pct:.3f} > '
                        f'{self.novel_exon_pct_max}, dropped {n_snvs_pre_b - n_snvs_post_b} SNVs '
                        f'in {len(novel_intervals)} novel sub-exon intervals '
                        f'({n_snvs_pre_b} -> {n_snvs_post_b})'
                    )


                    print(_knob_b_msg)
                    if self.logger is not None:
                        self.logger.info(_knob_b_msg)
        if len(df_pileup_filtered) == 0:
            return None, None, None, None, None, None, None, None
        snv_list = df_pileup_filtered["ID"].tolist()
        df_read_snv_filtered = df_read_snv.loc[:, snv_list]
        df_read_pi_filtered = df_read_pi.loc[:, snv_list]
        df_r, df_pi = df_read_snv_filtered, df_read_pi_filtered
        n_reads, n_snvs = df_r.shape
        gamma = float((df_r.to_numpy() == EM_MISSING_CODE).sum()) / (n_reads * n_snvs)

        _clf_prob_surviving = None
        if self.snv_classifier_model is not None and self.snv_confidence is None and len(df_pileup_filtered) > 0:
            if _site_reads_dict is not None:
                clf_features = self._extract_snv_classifier_features(_site_reads_dict, df_pileup_filtered)
                if (clf_features['depth'] == 0).any():
                    mes = f'[WARN] {geneID}: SNV classifier skipped — some positions had zero cached reads'
                    print(mes) if self.logger is None else self.logger.warning(mes)
                else:
                    clf_prob = self.snv_classifier_model.predict_proba(
                        clf_features.loc[:, SNV_CLF_FEATURE_COLUMNS]
                    )[:, 1]

                    hard_mask = clf_prob >= self.clf_hard_threshold
                    n_hard_removed = (~hard_mask).sum()

                    if self.clf_hard_threshold > 0 and n_hard_removed > 0:
                        df_pileup_filtered = df_pileup_filtered[hard_mask].reset_index(drop=True)
                        clf_prob = clf_prob[hard_mask]
                        mes = f'[DIAG] {geneID}: classifier hard filter removed {n_hard_removed} SNVs (clf_prob < {self.clf_hard_threshold}), {len(df_pileup_filtered)} remain'
                        _log_with_fallback(self.logger, mes)
                        if len(df_pileup_filtered) == 0:
                            return None, None, None, None, None, None, None, None
                        snv_list = df_pileup_filtered["ID"].tolist()
                        df_r = df_read_snv.loc[:, snv_list]
                        df_pi = df_read_pi.loc[:, snv_list]
                        n_reads, n_snvs = df_r.shape
                        gamma = float((df_r.to_numpy() == EM_MISSING_CODE).sum()) / (n_reads * n_snvs)
                    _clf_prob_surviving = np.asarray(clf_prob, dtype=float)
            else:
                mes = f'[WARN] {geneID}: site_reads.pkl not available, classifier skipped'
                print(mes) if self.logger is None else self.logger.warning(mes)

        h_m_init = None
        if self.snv_confidence is not None:


            h_m_init = np.ones(n_snvs, dtype=float)


            if not getattr(self, '_said_genotype_h_m_init', False):
                self._said_genotype_h_m_init = True
                _m = (f'h_m start set as: 1 for every supplied site '
                      f'(genotype mode overrides h_m_init_from='
                      f'{self.h_m_init_from!r}); first gene {n_snvs} site(s)')


                _log_with_fallback(getattr(self, 'logger', None), _m)
        elif self.h_m_init_from == 'linkage_lr':


            _, _, _loglr, _n_partner = linkage_loglr(
                df_r, df_pi, min_shared=self.init_link_min_shared)
            h_m_init = 1.0 / (1.0 + np.exp(-np.clip(_loglr, -30.0, 30.0)))
            h_m_init = np.clip(h_m_init, 1e-6, 1 - 1e-6)


            df_pileup_filtered['n_partner'] = _n_partner
        elif self.h_m_init_from == 'linkage':


            _, _, _s_marker, _n_partner = linkage_agreement(
                df_r, min_shared=self.init_link_min_shared)
            h_m_init = np.clip(_s_marker, 1e-6, 1 - 1e-6)
            df_pileup_filtered['n_partner'] = _n_partner
        elif self.h_m_init_from == 'none':


            h_m_init = None
        elif self.h_m_init_from == 'clf' and _clf_prob_surviving is not None:


            h_m_init = _clf_prob_surviving


        if (h_m_init is not None and self.snv_confidence is None
                and self.clf_pruning_frac < 1.0 and len(h_m_init) > 0):
            h_m_arr = np.array(h_m_init)
            keep_indices = np.arange(len(h_m_arr))
            while len(h_m_arr) > 0:
                frac_low = np.mean(h_m_arr < self.clf_pruning_threshold)
                if frac_low <= self.clf_pruning_frac:
                    break
                worst = np.argmin(h_m_arr)
                keep_indices = np.delete(keep_indices, worst)
                h_m_arr = np.delete(h_m_arr, worst)
            if len(h_m_arr) == 0:
                return None, None, None, None, None, None, None, None
            if len(h_m_arr) < len(h_m_init):
                _n_pruned = len(h_m_init) - len(h_m_arr)
                df_pileup_filtered = df_pileup_filtered.iloc[keep_indices].reset_index(drop=True)
                snv_list = df_pileup_filtered["ID"].tolist()
                df_r = df_read_snv.loc[:, snv_list]
                df_pi = df_read_pi.loc[:, snv_list]
                n_reads, n_snvs = df_r.shape
                gamma = float((df_r.to_numpy() == EM_MISSING_CODE).sum()) / (n_reads * n_snvs)
                h_m_init = h_m_arr
                mes = f'[DIAG] {geneID}: iterative pruning removed {_n_pruned} low-scoring SNVs, {len(h_m_arr)} remain'
                _log_with_fallback(self.logger, mes)
        em_snv_filter = bool(self.em_snv_filter and self.snv_confidence is None)
        em_kwargs = dict(max_iter=self.max_iter, tol=self.tol, verbose=self.verbose,
                         seed=self.seed,
                         heterozygous_priors=(0.4, 0.2, 0.4),
                         heterozygous_coverage_factor=self.heterozygous_coverage_factor,
                         het_beta=self.het_beta,
                         shrink_denominator=self.shrink_denominator,
                         h_m_filter=em_snv_filter, filter_reads=True, h_m_init=h_m_init,
                         gap_tau=self.gap_tau, init_method=self.em_init_method,
                         link_min_agreement=self.init_link_min_agreement,
                         link_min_shared=self.init_link_min_shared,
                         h_m_init_cleaned=not self.h_m_init_from.startswith('linkage'))


        results = run_em_capped(df_r, df_pi, cap=int(self.em_max_reads or 0),
                                gene_id=geneID, logger=self.logger, **em_kwargs)
        if results is None:
            return None, None, None, None, None, None, None, None
        if np.sum(results['h_m']>0.5)==0:
            return None, None, None, None, None, None, None, None


        _marker_positions = df_pileup_filtered['pos'].to_numpy()
        if self.phase_flip:
            results, _flip_info = guarded_switch_flip(
                df_r, df_pi, results, _marker_positions)


            if _flip_info.get('flip_applied'):
                _fm = (f'[FLIP] {geneID}: suffix flip applied at gap '
                       f'{_flip_info.get("flip_gap_pos")}, dll '
                       f'{_flip_info.get("flip_dll")}')
                _log_with_fallback(self.logger, _fm)


        _marker_blocks = assign_phase_blocks(
            df_r, results, _marker_positions,
            min_shared=self.phase_block_min_shared,
            min_agreement=self.phase_block_min_agreement)
        df_pileup_filtered['phase_block'] = _marker_blocks
        _read_blocks = assign_read_blocks(df_r, _marker_blocks)
        _n_phase_blocks = int(_marker_blocks.max()) if len(_marker_blocks) else 0
        alpha_hat = results['alpha']
        df_pileup_filtered['h_A'] = results['h_A']
        df_pileup_filtered['h_m'] = results['h_m']
        df_pileup_filtered['hat_Z_binary'] = results['hat_Z_binary']
        read_hap_df = pd.DataFrame({'Read': df_r.index.tolist(),'reads_phasable': results['reads_keep_mask']+0, 'hat_I': results['hat_I']})
        read_hap_df["hat_I"] = read_hap_df["hat_I"]
        read_hap_df["hat_I_B"] = 1 - read_hap_df["hat_I"]
        rp, s = read_hap_df['reads_phasable'].eq(1).astype(int), read_hap_df['hat_I']


        alpha_low, alpha_high = block_aware_alpha_bounds(
            read_hap_df['hat_I'].to_numpy(), rp.to_numpy(), _read_blocks)


        blk_low, blk_high = block_orientation_alpha_bounds(
            read_hap_df['hat_I'].to_numpy(), rp.to_numpy(), _read_blocks)
        lrt_test = ({'ll_alt': None, 'll_null': None, 'lrt_stat': None,
                     'p_value': None} if self.skip_ase_test
                    else self.bulk_lrt_orientation_robust(
                        df_r, df_pi, results, _marker_blocks, _read_blocks,
                        positions=_marker_positions))
        mapping_df_gene = self._gene_mapping_df(sample_index, geneID).reset_index(drop=True)


        if knob_c_blacklist:
            mapping_df_gene = mapping_df_gene[
                ~mapping_df_gene['Read'].astype(str).isin(knob_c_blacklist)
            ].reset_index(drop=True)
        mapping_df_gene = pd.merge(mapping_df_gene, read_hap_df, how = 'left', on = 'Read')
        count_matrix_hap0 = mapping_df_gene.pivot_table(index="Cell", columns="Isoform", values="hat_I", aggfunc="sum", fill_value=0)
        count_matrix_hap1 = mapping_df_gene.pivot_table(index="Cell", columns="Isoform", values="hat_I_B", aggfunc="sum", fill_value=0)
        mapping_df_gene_phasable = mapping_df_gene[mapping_df_gene.reads_phasable==1].reset_index(drop=True)
        count_matrix_hap0_phasable = mapping_df_gene_phasable.pivot_table(index="Cell", columns="Isoform", values="hat_I", aggfunc="sum",
                                                        fill_value=0)
        count_matrix_hap1_phasable = mapping_df_gene_phasable.pivot_table(index="Cell", columns="Isoform", values="hat_I_B",
                                                        aggfunc="sum", fill_value=0)
        count_matrix_hapA, count_matrix_hapB = count_matrix_hap0, count_matrix_hap1
        count_matrix_hapA_phasable, count_matrix_hapB_phasable = count_matrix_hap0_phasable, count_matrix_hap1_phasable
        geneName = mapping_df_gene.geneName[0]
        major_hap_bulk = 'B' if read_hap_df['hat_I_B'].mean() >= read_hap_df['hat_I'].mean() else 'A'
        if major_hap_bulk == 'A':
            alpha_hat_low, alpha_hat_high = 1 - alpha_high, 1 - alpha_low
            blk_hat_low, blk_hat_high = 1 - blk_high, 1 - blk_low
        else:
            alpha_hat_low, alpha_hat_high = alpha_low, alpha_high
            blk_hat_low, blk_hat_high = blk_low, blk_high


        read_hap_df["read_block"] = (
            _read_blocks.astype(int) if len(_read_blocks) else 0)
        result_dict = {'geneID': geneID, 'geneName': geneName, 'gamma': gamma,
                       'n_reads': n_reads, 'n_reads_phasable': read_hap_df.reads_phasable.sum(),
                       'n_snvs': n_snvs, 'alpha_hat': alpha_hat,
                       'alpha_hat_low': alpha_hat_low, 'alpha_hat_high': alpha_hat_high,
                       'alpha_hat_block_low': blk_hat_low,
                       'alpha_hat_block_high': blk_hat_high,
                       'major_hap': major_hap_bulk,
                       'll_alt': lrt_test['ll_alt'], 'll_null': lrt_test['ll_null'],
                       'lrt_stat': lrt_test['lrt_stat'], 'p_value': lrt_test['p_value'],
                       'n_phase_blocks': _n_phase_blocks,
                       'p_value_orient_min': lrt_test.get('p_value_orient_min'),
                       'n_orientations': lrt_test.get('n_orientations'),
                       'CellType': 'Bulk'}
        count_matrix_hapA.columns = [geneName + '_' + col + '_hapA' for col in count_matrix_hapA.columns.tolist()]
        count_matrix_hapB.columns = [geneName + '_' + col + '_hapB' for col in count_matrix_hapB.columns.tolist()]
        count_matrix_hapA_phasable.columns = [geneName + '_' + col + '_hapA' for col in count_matrix_hapA_phasable.columns.tolist()]
        count_matrix_hapB_phasable.columns = [geneName + '_' + col + '_hapB' for col in count_matrix_hapB_phasable.columns.tolist()]

        ct_results_df = None
        if self.cell_type_df_list is not None:
            ct_results_list = [result_dict]

            mapping_df_gene = mapping_df_gene.merge(self.cell_type_df_list[sample_index][['Cell','CellType']], how='left', on='Cell')
            mdf = mapping_df_gene.dropna(subset=['CellType', 'Read'])
            celltype_reads = mdf.groupby('CellType')['Read'].apply(list).to_dict()
            celltype_masks = {ct: df_r.index.isin(reads) for ct, reads in celltype_reads.items()}
            mdf_phasable = mdf[mdf['reads_phasable'] == 1]
            celltype_phasable_reads = mdf_phasable.groupby('CellType')['Read'].apply(list).to_dict()
            for celltype, celltype_mask in celltype_masks.items():
                row = {
                    'geneID': geneID,
                    'geneName': geneName,
                    'gamma': gamma,
                    'n_reads': len(celltype_reads[celltype]),
                    'n_reads_phasable': len(celltype_phasable_reads[celltype]) if celltype in celltype_phasable_reads.keys() else 0,
                    'n_snvs': n_snvs,
                    'alpha_hat': None,
                    'alpha_hat_low': None,
                    'alpha_hat_high': None,
                    'alpha_hat_block_low': None,
                    'alpha_hat_block_high': None,
                    'major_hap': None,
                    'll_alt': None,
                    'll_null': None,
                    'lrt_stat': None,
                    'p_value': None,
                    'n_phase_blocks': _n_phase_blocks,
                    'CellType': celltype}
                try:
                    df_r_ct = df_r.loc[celltype_mask]
                    df_pi_ct = df_pi.loc[celltype_mask]
                    result_ct = run_em(df_r_ct, df_pi_ct, max_iter=self.max_iter,
                                       tol=self.tol, verbose=self.verbose,
                                       seed=self.seed, heterozygous_priors=(0.4, 0.2, 0.4),
                                       heterozygous_coverage_factor=self.heterozygous_coverage_factor,
                                       het_beta=self.het_beta,
                                       shrink_denominator=self.shrink_denominator,
                                       h_m_filter=self.em_snv_filter, filter_reads=True, results = results)
                    lrt_test_ct = ({} if self.skip_ase_test else
                                   self.ct_lrt_allelic_balance_gene(df_r_ct, df_pi_ct, result_ct))
                    read_hap_df_ct = pd.DataFrame(
                        {'Read': df_r_ct.index.tolist(), 'reads_phasable': result_ct['reads_keep_mask'] + 0,
                         'hat_I': result_ct['hat_I']})
                    read_hap_df_ct["hat_I"] = read_hap_df_ct["hat_I"]
                    read_hap_df_ct["hat_I_B"] = 1 - read_hap_df_ct["hat_I"]
                    rp, s = read_hap_df_ct['reads_phasable'].eq(1).astype(int), read_hap_df_ct['hat_I']


                    _ct_pos = df_r.index.get_indexer(df_r_ct.index)
                    _ct_blocks = (_read_blocks[_ct_pos]
                                  if len(_read_blocks) else _ct_pos * 0)
                    alpha_ct_low, alpha_ct_high = block_aware_alpha_bounds(
                        read_hap_df_ct['hat_I'].to_numpy(), rp.to_numpy(),
                        _ct_blocks)
                    blk_ct_low, blk_ct_high = block_orientation_alpha_bounds(
                        read_hap_df_ct['hat_I'].to_numpy(), rp.to_numpy(),
                        _ct_blocks)
                    alpha_ct = result_ct['hat_I'].mean()
                    major_hap_ct = 'B' if alpha_ct <= 0.5 else 'A'
                    if major_hap_ct == 'A':
                        alpha_hat_low, alpha_hat_high = 1 - alpha_ct_high, 1 - alpha_ct_low
                        blk_hat_low, blk_hat_high = 1 - blk_ct_high, 1 - blk_ct_low
                    else:
                        alpha_hat_low = alpha_ct_low
                        alpha_hat_high = alpha_ct_high
                        blk_hat_low, blk_hat_high = blk_ct_low, blk_ct_high
                    read_hap_df_ct["hat_I"] = read_hap_df_ct["hat_I"].round(3)
                    read_hap_df_ct["hat_I_B"] = read_hap_df_ct["hat_I_B"].round(3)
                    row.update({
                        'n_reads_phasable': int(read_hap_df_ct.reads_phasable.sum()),
                        'alpha_hat': min(alpha_ct, 1 - alpha_ct),
                        'alpha_hat_low': alpha_hat_low,
                        'alpha_hat_high': alpha_hat_high,
                        'alpha_hat_block_low': blk_hat_low,
                        'alpha_hat_block_high': blk_hat_high,
                        'major_hap': major_hap_ct,
                        'll_alt': lrt_test_ct.get('ll_alt'),
                        'll_null': lrt_test_ct.get('ll_null'),
                        'lrt_stat': lrt_test_ct.get('lrt_stat'),
                        'p_value': lrt_test_ct.get('p_value'),


                        'n_phase_blocks': _n_phase_blocks})
                except Exception as e:
                    if getattr(self, 'verbose', False):
                        print(f"LRT failed for {geneID}/{celltype}: {e}")
                ct_results_list.append(row)
            ct_results_df = pd.DataFrame(ct_results_list)


        df_pileup_filtered = self._tag_in_gene_span(
            df_pileup_filtered, self.geneStructureInformation[geneID][0])
        return count_matrix_hapA, count_matrix_hapB, count_matrix_hapA_phasable, count_matrix_hapB_phasable, result_dict, df_pileup_filtered, read_hap_df, ct_results_df
    @staticmethod
    def _tag_in_gene_span(df, geneInfo):
        if df is None or 'pos' not in df.columns:
            return df
        df = df.copy()
        pos = pd.to_numeric(df['pos'], errors='coerce')
        df['in_gene_span'] = ((pos >= int(geneInfo['geneStart']))
                              & (pos <= int(geneInfo['geneEnd']))).astype(int)
        return df

    def generate_count_hap_gene(self, geneID, sample_index):
        geneName = self._get_gene_name(geneID)


        geneName = path_safe_gene_name(geneName)
        hapA_file_path = os.path.join(self.count_hap_folder_path, 'all_genes_separate', f'{geneName}_{geneID}_hapA.csv')
        hapB_file_path = os.path.join(self.count_hap_folder_path, 'all_genes_separate',f'{geneName}_{geneID}_hapB.csv')
        hapA_phasable_file_path = os.path.join(self.count_hap_folder_path, 'all_genes_separate', f'{geneName}_{geneID}_hapA_phasable.csv')
        hapB_phasable_file_path = os.path.join(self.count_hap_folder_path, 'all_genes_separate', f'{geneName}_{geneID}_hapB_phasable.csv')
        isoform_agg_file_path = os.path.join(self.count_hap_folder_path, 'all_genes_isoform_separate', f'{geneName}_{geneID}_isoform_agg.csv')
        isoform_agg_balance_file_path = os.path.join(self.count_hap_folder_path, 'all_genes_isoform_separate',
                                             f'{geneName}_{geneID}_isoform_agg_balance.csv')
        isoform_agg_imbalance_file_path = os.path.join(self.count_hap_folder_path, 'all_genes_isoform_separate',
                                             f'{geneName}_{geneID}_isoform_agg_unbalance.csv')
        snv_file_path = os.path.join(self.snv_hap_path, 'all_genes_separate_snv', f'{geneName}_{geneID}.csv')
        em_result_file_path = os.path.join(self.summary_statistics_path, 'all_genes_separate', f'{geneName}_{geneID}_summary.csv')
        read_hap_path = os.path.join(self.snv_hap_path, 'all_genes_separate', f'{geneName}_{geneID}_read_hap.csv')
        if os.path.exists(em_result_file_path) == False or self.cover_existing:
            print(f"----- Start Processing gene: {geneID} = {geneName} -----")
            try:
                print(f"Running EM algorithm")
                count_matrix_hapA, count_matrix_hapB, count_matrix_hapA_phasable, count_matrix_hapB_phasable, result_dict, snv_info_df, read_hap_df, ct_results_df = self.run_em_gene(geneID, sample_index)
                if snv_info_df is not None:
                    count_matrix_hapA.to_csv(hapA_file_path)
                    count_matrix_hapB.to_csv(hapB_file_path)
                    count_matrix_hapA_phasable.to_csv(hapA_phasable_file_path)
                    count_matrix_hapB_phasable.to_csv(hapB_phasable_file_path)
                    snv_info_df.to_csv(snv_file_path, index=False)
                    read_hap_df.round({'hat_I': 3, 'hat_I_B': 3}).to_csv(read_hap_path)
                    tab = self._aggregate_isoform_hap_table(count_matrix_hapA_phasable, count_matrix_hapB_phasable,
                                                            min_frac=self.chi_min_frac, group_novel = self.chi_group_novel)
                    tab_balance = self._aggregate_isoform_hap_table(count_matrix_hapA, count_matrix_hapB,
                                                                    min_frac=self.chi_min_frac,
                                                                    group_novel=self.chi_group_novel)


                    tab_imbalance = self._tab_min_p(tab, tab_balance, self.logger)
                    tab_pmax = self._tab_max_p(tab, tab_balance)
                    tab.to_csv(isoform_agg_file_path)
                    tab_balance.to_csv(isoform_agg_balance_file_path)
                    tab_imbalance.to_csv(isoform_agg_imbalance_file_path)


                    tab_pmax.to_csv(os.path.join(
                        self.count_hap_folder_path, 'all_genes_isoform_separate',
                        f'{geneName}_{geneID}_isoform_agg_pmax.csv'))


                    _m_hapA = self._hapA_share_unfolded(read_hap_df)
                    _exA, _exB = self._extrapolate_isoform_counts(
                        count_matrix_hapA, count_matrix_hapB,
                        count_matrix_hapA_phasable, count_matrix_hapB_phasable,
                        (_m_hapA if _m_hapA is not None else 0.5))
                    tab_extrap = self._aggregate_isoform_hap_table(
                        _exA, _exB, min_frac=self.chi_min_frac,
                        group_novel=self.chi_group_novel)
                    tab_extrap.to_csv(os.path.join(
                        self.count_hap_folder_path, 'all_genes_isoform_separate',
                        f'{geneName}_{geneID}_isoform_agg_extrap.csv'))
                    out, out_imbalance, out_balance = self._chisq_or_skip(tab), self._chisq_or_skip(tab_imbalance), self._chisq_or_skip(tab_pmax)


                    out_prop = self._chisq_or_skip(tab_balance)
                    pvals = [out_balance.get('p_value_isoform'), out_imbalance.get('p_value_isoform'), out.get('p_value_isoform'), out_prop.get('p_value_isoform')]
                    pvals = [v for v in pvals if v is not None and not pd.isna(v)]
                    out['p_value_isoform_high'] = max(pvals) if pvals else None
                    out['p_value_isoform_low'] = min(pvals) if pvals else None


                    out['p_value_isoform_orient_min'] = out.get('p_value_isoform')
                    _nb = result_dict.get('n_phase_blocks') or 0
                    if _nb > 1 and not self.skip_astu_test:
                        _ri = self._gene_mapping_df(sample_index, geneID)[
                            ['Read', 'Isoform']].merge(
                            read_hap_df[['Read', 'hat_I', 'reads_phasable',
                                         'read_block']],
                            on='Read', how='inner')
                        _sums = self._astu_block_sums(
                            _ri, geneName, set(tab.index), set(tab_balance.index))
                        _orient = self._astu_orientation_pvals(_sums, _nb)
                        if _orient is None:
                            mes = (f'{geneID}: {_nb} phase blocks exceed '
                                   f'{self.MAX_ASTU_ORIENTATIONS} orientations '
                                   f'— ASTU p and interval set to NA (mirrors '
                                   f'the ASE LRT refusal)')
                            print(mes) if self.logger is None else self.logger.warning(mes)
                            for _k in ('p_value_isoform', 'p_value_isoform_high',
                                       'p_value_isoform_low',
                                       'p_value_isoform_orient_min',
                                       'chi2_isoform', 'df_isoform'):
                                out[_k] = None
                        else:


                            out.update(_orient)
                            out['chi2_isoform'] = None
                            out['df_isoform'] = None
                            _sums.to_csv(os.path.join(
                                self.count_hap_folder_path,
                                'all_genes_isoform_separate',
                                f'{geneName}_{geneID}_astu_block_sums.csv'),
                                index=False)
                    df1 = pd.DataFrame([result_dict])
                    df2 = pd.DataFrame([out])
                    df = pd.concat([df1, df2], axis=1)
                    df["CellType"] = 'Bulk'

                    dfs_ct = []
                    if self.cell_type_df_list is not None:
                        celltype_cells = self.cell_type_df_list[sample_index].groupby("CellType")["Cell"].apply(list).to_dict()
                        for ct, cells in celltype_cells.items():

                            mask = count_matrix_hapA.index.isin(cells)
                            hapA_ct = count_matrix_hapA.loc[mask]
                            hapB_ct = count_matrix_hapB.loc[mask]
                            mask = count_matrix_hapA_phasable.index.isin(cells)
                            hapA_phasable_ct = count_matrix_hapA_phasable.loc[mask]
                            hapB_phasable_ct = count_matrix_hapB_phasable.loc[mask]
                            if hapA_ct.empty or hapB_ct.empty:
                                continue
                            tab_ct = self._aggregate_isoform_hap_table(hapA_phasable_ct, hapB_phasable_ct,
                                                                       min_frac=self.chi_min_frac,
                                                                       group_novel=self.chi_group_novel)
                            tab_balance_ct = self._aggregate_isoform_hap_table(hapA_ct, hapB_ct,
                                                                               min_frac=self.chi_min_frac,
                                                                               group_novel=self.chi_group_novel)

                            tab_imbalance_ct = self._tab_min_p(tab_ct, tab_balance_ct, self.logger)
                            tab_pmax_ct = self._tab_max_p(tab_ct, tab_balance_ct)

                            safe_ct = ct.replace('/', '_').replace(' ', '_')
                            ct_iso_dir = os.path.join(self.count_hap_folder_path, 'ct_isoform_separate', safe_ct)
                            tab_ct.to_csv(os.path.join(ct_iso_dir, f'{geneName}_{geneID}_isoform_agg.csv'))
                            tab_balance_ct.to_csv(os.path.join(ct_iso_dir, f'{geneName}_{geneID}_isoform_agg_balance.csv'))
                            tab_imbalance_ct.to_csv(os.path.join(ct_iso_dir, f'{geneName}_{geneID}_isoform_agg_unbalance.csv'))
                            tab_pmax_ct.to_csv(os.path.join(ct_iso_dir, f'{geneName}_{geneID}_isoform_agg_pmax.csv'))


                            _exA_ct, _exB_ct = self._extrapolate_isoform_counts(
                                hapA_ct, hapB_ct, hapA_phasable_ct,
                                hapB_phasable_ct,
                                (_m_hapA if _m_hapA is not None else 0.5))
                            tab_extrap_ct = self._aggregate_isoform_hap_table(
                                _exA_ct, _exB_ct, min_frac=self.chi_min_frac,
                                group_novel=self.chi_group_novel)
                            tab_extrap_ct.to_csv(os.path.join(
                                ct_iso_dir,
                                f'{geneName}_{geneID}_isoform_agg_extrap.csv'))
                            out_ct = self._chisq_or_skip(tab_ct)
                            out_ct_imbalance = self._chisq_or_skip(tab_imbalance_ct)
                            out_ct_balance = self._chisq_or_skip(tab_pmax_ct)
                            out_ct_prop = self._chisq_or_skip(tab_balance_ct)
                            pvals_ct = [out_ct_prop.get("p_value_isoform"),
                                        out_ct_balance.get("p_value_isoform"),
                                        out_ct_imbalance.get("p_value_isoform"),
                                        out_ct.get("p_value_isoform")]
                            pvals_ct = [v for v in pvals_ct if v is not None and not pd.isna(v)]
                            out_ct["p_value_isoform_high"] = max(pvals_ct) if pvals_ct else None
                            out_ct["p_value_isoform_low"] = min(pvals_ct) if pvals_ct else None
                            df_ct = pd.concat([pd.DataFrame([result_dict]), pd.DataFrame([out_ct])], axis=1)
                            df_ct["CellType"] = ct
                            dfs_ct.append(df_ct)
                        df = pd.concat([df] + dfs_ct, ignore_index=True)


                        df_ = df[['chi2_isoform','df_isoform','p_value_isoform','p_value_isoform_low','p_value_isoform_high','p_value_isoform_orient_min','CellType']]
                        df = ct_results_df.merge(df_, on = 'CellType')
                        df['CellType'] = df.pop('CellType')
                    df.to_csv(em_result_file_path, index=False)
                    _msg = f"Results Saved for {geneID}"
                    print(_msg) if self.logger is None else self.logger.info(_msg)
                else:
                    _msg = f'Empty results for {geneID}'
                    print(_msg) if self.logger is None else self.logger.info(_msg)
            except KnobCInputError:


                raise
            except Exception as e:
                import traceback
                _emsg = f"Error processing {geneID}: {e}\n{traceback.format_exc()}"
                print(_emsg) if self.logger is None else self.logger.error(_emsg)
                return 'failed'
        return 'ok'
    def _generate_count_hap_gene_safe(self, geneID, sample_index):
        import time


        attach_worker_log_handler(self.logger, getattr(self, '_log_file', None))
        t0 = time.perf_counter()
        status = 'processed'
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    ret = self.generate_count_hap_gene(geneID, sample_index)


            if ret == 'failed':
                status = 'failed'
        except KnobCInputError as exc:


            _m = f'⛔ gene {geneID}: Knob C read-filter input unusable, stopping the run: {exc}'
            print(_m, flush=True) if self.logger is None else self.logger.error(_m)
            raise
        except Exception:
            status = 'failed'
            _m = f'gene {geneID}: unhandled exception in step3 worker'
            print(_m) if self.logger is None else self.logger.error(_m)
        wall = round(time.perf_counter() - t0, 3)


        if status == 'processed' and wall < 0.05:
            status = 'skipped_or_trivial'
        return (geneID, wall, status)
    def generate_count_hap_genes(self):
        mes = f'Perform haplotype phasing for {self.n_samples} sample in total'
        print(mes) if self.logger is None else self.logger.info(mes)
        warn_if_output_is_symlinked(
            [getattr(self, a, None) for a in ('count_hap_folder_path', 'snv_hap_path',
                                               'summary_statistics_path')], self.logger)


        self._hap_gene_index, self._hap_gene_cols = self._load_gene_index_hap()
        if self._hap_gene_index is not None:
            self.mapping_df_dict_list = None
            mes = 'step3: using per-gene index (seek mode); skipping full mapping load'
        else:
            mes = 'Load SCOTCH read mapping information'
            self.mapping_df_dict_list = self._read_mapping()
        print(mes) if self.logger is None else self.logger.info(mes)
        for i in self._iter_sample_indices():
            self._get_paths(sample_index = i)
            os.makedirs(os.path.join(self.snv_hap_path, 'all_genes_separate'), exist_ok=True)
            os.makedirs(os.path.join(self.snv_hap_path, 'all_genes_separate_snv'), exist_ok=True)
            os.makedirs(os.path.join(self.summary_statistics_path, 'all_genes_separate'), exist_ok=True)
            os.makedirs(os.path.join(self.count_hap_folder_path, 'all_genes_separate'), exist_ok=True)
            os.makedirs(os.path.join(self.count_hap_folder_path, 'all_genes_isoform_separate'), exist_ok=True)
            if self.cell_type_df_list is not None:
                for ct in self.cell_type_df_list[i]['CellType'].unique():
                    safe_ct = ct.replace('/', '_').replace(' ', '_')
                    os.makedirs(os.path.join(self.count_hap_folder_path, 'ct_isoform_separate', safe_ct), exist_ok=True)


            geneIDs_job = np.array_split(self.geneIDs, self.n_jobs)[self.job_index]
            em_dir = (self.em_input if self.n_samples == 1
                      else os.path.join(self.em_input, self.sample_names[i]))
            geneIDs_job = _order_genes_heavy_first(list(geneIDs_job), em_dir)
            mes = (f'{len(geneIDs_job)} genes in this job for sample index {i} '
                   f'(backend={self.step3_backend}, heaviest EM input first)')
            print(mes) if self.logger is None else self.logger.info(mes)


            blas_vars = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                         'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
            saved_env = {}
            if self.step3_backend == 'loky':


                for var in blas_vars:
                    saved_env[var] = os.environ.get(var)
                    os.environ[var] = '1'
                par = Parallel(n_jobs=self.n_workers, backend='loky')
            else:
                par = Parallel(n_jobs=self.n_workers, prefer='threads')
            try:
                timings = par(
                    delayed(self._generate_count_hap_gene_safe)(geneID, i)
                    for geneID in geneIDs_job)
            finally:
                for var, val in saved_env.items():
                    if val is None:
                        os.environ.pop(var, None)
                    else:
                        os.environ[var] = val


            timing_rows = [t for t in timings if t is not None]
            if timing_rows:
                tdir = os.path.join(self.target, 'step3_timing')
                os.makedirs(tdir, exist_ok=True)
                tpath = os.path.join(
                    tdir, f'timing_s{i}_job{self.job_index}.csv')
                tdf = pd.DataFrame(timing_rows,
                                   columns=['geneID', 'wall_s', 'status'])
                if os.path.exists(tpath):


                    old = pd.read_csv(tpath)
                    if 'status' not in old.columns:
                        old['status'] = 'processed'
                    keep = old[(old['status'] == 'processed')
                               & ~old['geneID'].isin(
                                   set(tdf.loc[tdf['status'] == 'processed',
                                               'geneID']))]
                    tdf = pd.concat([keep, tdf[~tdf['geneID'].isin(
                        set(keep['geneID']))]], ignore_index=True)
                tdf.to_csv(tpath, index=False)
            mes = f'job {self.job_index} finished for sample index {i}'
            print(mes) if self.logger is None else self.logger.info(mes)
    def _process_count_gene(self, hapA_file):
        hapB_file = hapA_file.replace('_hapA', '_hapB')
        geneName = os.path.basename(hapA_file).split('_')[0]
        df_A = pd.read_csv(hapA_file, index_col=0)
        df_B = pd.read_csv(hapB_file, index_col=0)
        isoform_df = pd.concat([df_A, df_B], axis=1)
        gene_df = pd.DataFrame({f'{geneName}_hapA': df_A.sum(axis=1),f'{geneName}_hapB': df_B.sum(axis=1)})
        triple_isoform = self._df_to_triple(isoform_df)
        triple_gene = self._df_to_triple(gene_df)
        return triple_isoform, triple_gene
    @staticmethod
    def _df_to_triple(df):
        rowNames = df.index.tolist()
        colNmes = df.columns.tolist()
        mat = np.array(df)
        x = mat[np.nonzero(mat)]
        rowIndex, colIndex = np.nonzero(mat)
        ij_list = list(zip(x, rowIndex, colIndex))
        triple = [(x, rowNames[i], colNmes[j]) for x, i, j in ij_list]
        return triple
    @staticmethod
    def _generate_adata(triple_list):


        cells_dict = {}
        features_dict = {}
        data = []
        cells = []
        features = []
        for x, cell, feature in triple_list:
            if cell not in cells_dict:
                cells_dict[cell] = len(cells_dict)
                cells.append(cell)
            if feature not in features_dict:
                features_dict[feature] = len(features_dict)
                features.append(feature)
            data.append((x, cells_dict[cell], features_dict[feature]))
        x, cells_ind, features_ind = zip(*data)
        sparse_matrix = csr_matrix((x, (cells_ind, features_ind)))
        adata = ad.AnnData(sparse_matrix)
        adata.obs_names = cells
        adata.var_names = features
        return adata


    def _process_count_gene_arrays(self, hapA_file):
        hapB_file = hapA_file.replace('_hapA', '_hapB')
        geneName = os.path.basename(hapA_file).split('_')[0]
        df_A = pd.read_csv(hapA_file, index_col=0)
        df_B = pd.read_csv(hapB_file, index_col=0)
        isoform_df = pd.concat([df_A, df_B], axis=1)
        gene_df = pd.DataFrame({f'{geneName}_hapA': df_A.sum(axis=1),
                                f'{geneName}_hapB': df_B.sum(axis=1)})
        return self._df_to_coo(isoform_df), self._df_to_coo(gene_df)
    @staticmethod
    def _df_to_coo(df):
        mat = np.asarray(df)
        r, c = np.nonzero(mat)
        vals = mat[r, c]
        cells = df.index.to_numpy(dtype=object)[r]
        feats = df.columns.to_numpy(dtype=object)[c]
        return vals, cells, feats
    def _collect_count_arrays(self, hap_files):
        if not hap_files:
            return [], []
        if self._should_parallelize(hap_files):
            results = Parallel(n_jobs=self.n_workers, prefer='threads')(
                delayed(self._process_count_gene_arrays)(hap_file)
                for hap_file in hap_files
            )
        else:
            results = [self._process_count_gene_arrays(hap_file) for hap_file in hap_files]
        transcript_parts = [r[0] for r in results]
        gene_parts = [r[1] for r in results]
        return transcript_parts, gene_parts
    @staticmethod
    def _generate_adata_from_parts(parts):
        parts = [p for p in parts if len(p[0])]
        if not parts:
            return ad.AnnData(csr_matrix((0, 0)))
        vals = np.concatenate([p[0] for p in parts])
        cells = np.concatenate([p[1] for p in parts])
        feats = np.concatenate([p[2] for p in parts])
        uniq_cells = pd.unique(cells)
        uniq_feats = pd.unique(feats)
        cell_idx = pd.Index(uniq_cells).get_indexer(cells)
        feat_idx = pd.Index(uniq_feats).get_indexer(feats)
        adata = ad.AnnData(csr_matrix((vals, (cell_idx, feat_idx))))
        adata.obs_names = list(uniq_cells)
        adata.var_names = list(uniq_feats)
        return adata
    def generate_count_matrix(self):
        warn_if_output_is_symlinked(
            [getattr(self, a, None) for a in ('count_hap_folder_path', 'snv_hap_path',
                                               'summary_statistics_path')], self.logger)
        for i in self._iter_sample_indices():
            self._get_paths(sample_index = i)
            all_genes_dir = os.path.join(self.count_hap_folder_path, 'all_genes')
            all_genes_sep_isoform_dir = os.path.join(self.count_hap_folder_path, 'all_genes_isoform_separate')
            os.makedirs(all_genes_dir, exist_ok=True)

            isoform_agg_files = sorted(
                os.path.join(all_genes_sep_isoform_dir, f)
                for f in os.listdir(all_genes_sep_isoform_dir)
                if f.endswith('_isoform_agg.csv')
            )
            isoform_agg_balance_files = sorted(
                os.path.join(all_genes_sep_isoform_dir, f)
                for f in os.listdir(all_genes_sep_isoform_dir)
                if f.endswith('_isoform_agg_balance.csv')
            )
            isoform_agg_unbalance_files = sorted(
                os.path.join(all_genes_sep_isoform_dir, f)
                for f in os.listdir(all_genes_sep_isoform_dir)
                if f.endswith('_isoform_agg_unbalance.csv')
            )
            isoform_agg_extrap_files = sorted(
                os.path.join(all_genes_sep_isoform_dir, f)
                for f in os.listdir(all_genes_sep_isoform_dir)
                if f.endswith('_isoform_agg_extrap.csv')
            )
            isoform_agg_pmax_files = sorted(
                os.path.join(all_genes_sep_isoform_dir, f)
                for f in os.listdir(all_genes_sep_isoform_dir)
                if f.endswith('_isoform_agg_pmax.csv')
            )
            iso_agg_csv = os.path.join(all_genes_dir, 'isoform_agg.csv')
            iso_agg_balance_csv = os.path.join(all_genes_dir, 'isoform_agg_balance.csv')
            iso_agg_unbalance_csv = os.path.join(all_genes_dir, 'isoform_agg_unbalance.csv')
            iso_agg_extrap_csv = os.path.join(all_genes_dir, 'isoform_agg_extrap.csv')
            self._append_isoform_agg_csv(isoform_agg_files, iso_agg_csv)
            self._append_isoform_agg_csv(isoform_agg_balance_files, iso_agg_balance_csv)
            self._append_isoform_agg_csv(isoform_agg_unbalance_files, iso_agg_unbalance_csv)
            self._append_isoform_agg_csv(isoform_agg_extrap_files, iso_agg_extrap_csv)
            self._append_isoform_agg_csv(isoform_agg_pmax_files,
                                         os.path.join(all_genes_dir, 'isoform_agg_pmax.csv'))

            ct_iso_sep_dir = os.path.join(self.count_hap_folder_path, 'ct_isoform_separate')
            if os.path.isdir(ct_iso_sep_dir):
                for safe_ct in os.listdir(ct_iso_sep_dir):
                    ct_dir = os.path.join(ct_iso_sep_dir, safe_ct)
                    if not os.path.isdir(ct_dir):
                        continue
                    for tag, suffix in [('isoform_agg', '_isoform_agg.csv'),
                                        ('isoform_agg_balance', '_isoform_agg_balance.csv'),
                                        ('isoform_agg_unbalance', '_isoform_agg_unbalance.csv'),
                                        ('isoform_agg_extrap', '_isoform_agg_extrap.csv'),
                                        ('isoform_agg_pmax', '_isoform_agg_pmax.csv')]:
                        tag_files = sorted(f for f in os.listdir(ct_dir) if f.endswith(suffix))
                        out_csv = os.path.join(all_genes_dir, f'ct_{safe_ct}_{tag}.csv')
                        self._append_isoform_agg_csv(
                            [os.path.join(ct_dir, fname) for fname in tag_files],
                            out_csv,
                        )
        for i in self._iter_sample_indices():
            self._get_paths(sample_index=i)
            all_genes_dir = os.path.join(self.count_hap_folder_path, 'all_genes')
            all_genes_sep_dir = os.path.join(self.count_hap_folder_path, 'all_genes_separate')
            hapA_files = sorted(
                os.path.join(all_genes_sep_dir, f)
                for f in os.listdir(all_genes_sep_dir)
                if f.endswith('hapA.csv')
            )
            hapA_phasable_files = sorted(
                os.path.join(all_genes_sep_dir, f)
                for f in os.listdir(all_genes_sep_dir)
                if f.endswith('hapA_phasable.csv')
            )
            transcript_parts, gene_parts = self._collect_count_arrays(hapA_files)
            adata_gene = self._generate_adata_from_parts(gene_parts)
            adata_transcript = self._generate_adata_from_parts(transcript_parts)
            transcript_phasable_parts, gene_phasable_parts = self._collect_count_arrays(hapA_phasable_files)
            adata_gene_phasable = self._generate_adata_from_parts(gene_phasable_parts)
            adata_transcript_phasable = self._generate_adata_from_parts(transcript_phasable_parts)
            if self.mtx:

                gene_mtx_path = os.path.join(all_genes_dir, 'count_mat_gene.mtx')
                gene_meta_path = os.path.join(all_genes_dir, 'count_mat_gene_meta.pkl')
                gene_phasable_mtx_path = os.path.join(all_genes_dir, 'count_mat_gene_phasable.mtx')
                gene_phasable_meta_path = os.path.join(all_genes_dir, 'count_mat_gene_phasable_meta.pkl')
                gene_meta = {'obs': adata_gene.obs.index.tolist(), "var": adata_gene.var.index.tolist()}
                gene_phasable_meta = {'obs': adata_gene_phasable.obs.index.tolist(), "var": adata_gene_phasable.var.index.tolist()}
                with open(gene_meta_path, 'wb') as f:
                    pickle.dump(gene_meta, f)
                mmwrite(gene_mtx_path, adata_gene.X)
                with open(gene_phasable_meta_path, 'wb') as f:
                    pickle.dump(gene_phasable_meta, f)
                mmwrite(gene_phasable_mtx_path, adata_gene_phasable.X)

                transcript_mtx_path = os.path.join(all_genes_dir, 'count_mat_transcript.mtx')
                transcript_meta_path = os.path.join(all_genes_dir, 'count_mat_transcript_meta.pkl')
                transcript_phasable_mtx_path = os.path.join(all_genes_dir, 'count_mat_transcript_phasable.mtx')
                transcript_phasable_meta_path = os.path.join(all_genes_dir, 'count_mat_transcript_phasable_meta.pkl')
                transcript_meta = {'obs': adata_transcript.obs.index.tolist(), "var": adata_transcript.var.index.tolist()}
                transcript_phasable_meta = {'obs': adata_transcript_phasable.obs.index.tolist(), "var": adata_transcript_phasable.var.index.tolist()}
                with open(transcript_meta_path, 'wb') as f:
                    pickle.dump(transcript_meta, f)
                mmwrite(transcript_mtx_path, adata_transcript.X)
                with open(transcript_phasable_meta_path, 'wb') as f:
                    pickle.dump(transcript_phasable_meta, f)
                mmwrite(transcript_phasable_mtx_path, adata_transcript_phasable.X)
            if self.csv:
                transcript_csv = os.path.join(all_genes_dir, 'count_mat_transcript.csv')
                gene_csv = os.path.join(all_genes_dir, 'count_mat_gene.csv')
                adata_gene_df = adata_gene.to_df()
                adata_gene_df.to_csv(gene_csv)
                adata_transcript_df = adata_transcript.to_df()
                adata_transcript_df.to_csv(transcript_csv)
                transcript_phasable_csv = os.path.join(all_genes_dir, 'count_mat_transcript_phasable.csv')
                gene_phasable_csv = os.path.join(all_genes_dir, 'count_mat_gene_phasable.csv')
                adata_gene_phasable_df = adata_gene_phasable.to_df()
                adata_gene_phasable_df.to_csv(gene_phasable_csv)
                adata_transcript_phasable_df = adata_transcript_phasable.to_df()
                adata_transcript_phasable_df.to_csv(transcript_phasable_csv)

            if self.cell_type_df_list is not None:
                ct_df = self.cell_type_df_list[i]
                bulk_adatas = [
                    (adata_gene,                  'count_mat_gene'),
                    (adata_transcript,             'count_mat_transcript'),
                    (adata_gene_phasable,          'count_mat_gene_phasable'),
                    (adata_transcript_phasable,    'count_mat_transcript_phasable'),
                ]
                for ct, ct_cells_df in ct_df.groupby('CellType'):
                    safe_ct = ct.replace('/', '_').replace(' ', '_')
                    ct_cells = set(ct_cells_df['Cell'].tolist())
                    for adata_obj, tag in bulk_adatas:
                        ct_obs = [c for c in adata_obj.obs_names if c in ct_cells]
                        if not ct_obs:
                            continue
                        adata_ct = adata_obj[ct_obs, :]
                        ct_tag = f'{tag}_ct_{safe_ct}'
                        if self.mtx:
                            mtx_path = os.path.join(all_genes_dir, f'{ct_tag}.mtx')
                            meta_path = os.path.join(all_genes_dir, f'{ct_tag}_meta.pkl')
                            meta = {'obs': adata_ct.obs_names.tolist(),
                                    'var': adata_ct.var_names.tolist()}
                            with open(meta_path, 'wb') as f:
                                pickle.dump(meta, f)
                            mmwrite(mtx_path, adata_ct.X)
                        if self.csv:
                            csv_path = os.path.join(all_genes_dir, f'{ct_tag}.csv')
                            adata_ct.to_df().to_csv(csv_path)

    def get_summary_statistics(self):
        for i in self._iter_sample_indices():
            self._get_paths(sample_index = i)

            mes = f'summarizing summary files for sample index {i}'
            print(mes) if self.logger is None else self.logger.info(mes)
            summary_statistics_path = os.path.join(self.summary_statistics_path, 'all_genes_separate')
            files = [os.path.join(summary_statistics_path, f) for f in os.listdir(summary_statistics_path) if f.endswith('summary.csv')]
            if self._should_parallelize(files):
                df_list = Parallel(n_jobs=self.n_workers, prefer='threads')(
                    delayed(pd.read_csv)(file) for file in files)
            else:
                df_list = [pd.read_csv(file) for file in files]
            df = pd.concat(df_list).reset_index(drop = True)
            groups = []
            for ct, g in df.groupby("CellType"):
                if "p_value" in g:
                    mask = g["p_value"].notna()
                    g["p_value_gene_adj"] = pd.NA
                    if mask.any():
                        g.loc[mask, "p_value_gene_adj"] = multipletests(g.loc[mask, "p_value"],method="fdr_bh")[1]
                if "p_value_isoform" in g:
                    mask = g["p_value_isoform"].notna()
                    g["p_value_isoform_adj"] = pd.NA
                    g["p_value_isoform_adj_high"] = pd.NA
                    g["p_value_isoform_adj_low"] = pd.NA
                    if mask.any():
                        g.loc[mask, "p_value_isoform_adj"] = multipletests(g.loc[mask, "p_value_isoform"], method="fdr_bh")[1]
                        g.loc[mask, "p_value_isoform_adj_high"] = multipletests(g.loc[mask, "p_value_isoform_high"], method="fdr_bh")[1]
                        g.loc[mask, "p_value_isoform_adj_low"] = multipletests(g.loc[mask, "p_value_isoform_low"], method="fdr_bh")[1]
                groups.append(g)
            df = pd.concat(groups).reset_index(drop=True)
            df = df.sort_values(["geneName", "CellType"]).reset_index(drop=True)
            filepath = os.path.join(self.summary_statistics_path, 'summary_statistics.csv')
            df.to_csv(filepath)
            mes = f'summary file saved for sample index {i} at {filepath}'
            print(mes) if self.logger is None else self.logger.info(mes)

            mes = f'summarizing read-haplotype mapping files for sample index {i}'
            print(mes) if self.logger is None else self.logger.info(mes)
            read_hap_path = os.path.join(self.snv_hap_path, 'all_genes_separate')
            files = [os.path.join(read_hap_path, f) for f in os.listdir(read_hap_path) if f.endswith('read_hap.csv')]
            filepath = os.path.join(self.snv_hap_path, 'read_hap_map.csv')
            entries = []
            for file in files:
                geneName, geneID = os.path.basename(file).split('_')[:2]
                entries.append((file, geneName, geneID))


            if not self._stream_concat_gene_csvs(entries, filepath, drop_first_col=True):
                df_list = []
                for file, geneName, geneID in entries:
                    df_ = pd.read_csv(file, index_col=0)
                    df_['geneName'] = geneName
                    df_['geneID'] = geneID
                    df_list.append(df_)
                df = pd.concat(df_list).reset_index(drop=True)
                df.to_csv(filepath)
            mes = f'read-haplotype mapping file saved for sample index {i} at {filepath}'
            print(mes) if self.logger is None else self.logger.info(mes)

            mes = f'summarizing snv-haplotype mapping files for sample index {i}'
            print(mes) if self.logger is None else self.logger.info(mes)
            snv_hap_path = os.path.join(self.snv_hap_path, 'all_genes_separate_snv')
            files = [os.path.join(snv_hap_path, f) for f in os.listdir(snv_hap_path) if f.endswith('.csv')]
            filepath = os.path.join(self.snv_hap_path, 'snv_hap_map.csv')
            entries = []
            for file in files:
                geneName, geneID = os.path.basename(file).split('_')[:2]
                geneID = geneID.replace('.csv', '')
                entries.append((file, geneName, geneID))


            if not self._stream_concat_gene_csvs(entries, filepath, drop_first_col=False):
                df_list = []
                for file, geneName, geneID in entries:
                    df_ = pd.read_csv(file)
                    df_['geneName'] = geneName
                    df_['geneID'] = geneID
                    df_list.append(df_)
                df = pd.concat(df_list).reset_index(drop=True)
                df = _collapse_legacy_merge_suffixes(
                    df, lambda mes: print(mes) if self.logger is None else self.logger.warning(mes)
                )
                df.to_csv(filepath)
            mes = f'read-haplotype mapping file saved for sample index {i} at {filepath}'
            print(mes) if self.logger is None else self.logger.info(mes)
