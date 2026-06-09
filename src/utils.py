import pysam
from collections import defaultdict
import pandas as pd
import os
import pickle
import numpy as np
import anndata as ad
from src.inference import run_em, compute_P_het
from scipy import sparse
from scipy.io import mmwrite
from typing import Union, Sequence
import warnings
from scipy.stats import chi2
from scipy.special import gammaln
import subprocess
import tempfile
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


# SCOTCH 10x-pacbio mode appends `_<alignment_length_bp>` to PacBio CCS read names
# in the aux TSV (compatible.py:876); 10x-ont / parse-ont modes leave them alone.
# Anchor the strip to PacBio CCS naming (`<movie>/<zmw>/ccs`) so ONT UUIDs and any
# other naming scheme are never touched even if they happen to end in `_<digits>`.
_READNAME_SUFFIX_RE = re.compile(r'(?<=/ccs)_\d+$')


def canonicalize_read_name(name):
    if not isinstance(name, str):
        return name
    return _READNAME_SUFFIX_RE.sub('', name)


def compute_intron_spans(read):
    """Reference-coordinate (start, end) tuples of CIGAR N ops in this read.

    Step1 calls this once per BAM read while collecting site_reads and dumps the
    result to `{gene}_read_blocks.pkl`, so step5 obs_* validation can compute
    CIGAR-observed event membership without re-opening BAM (which has
    petagene-concurrency issues under joblib parallel workers).
    """
    spans = []
    cigar = read.cigartuples
    if cigar is None:
        return spans
    ref_pos = read.reference_start
    for op, length in cigar:
        if op == 3:  # N — splice gap
            spans.append((ref_pos, ref_pos + length))
            ref_pos += length
        elif op in (0, 2, 7, 8):  # M, D, =, X — consume reference
            ref_pos += length
        # I (1), S (4), H (5), P (6) do not consume reference
    return spans


def _log_with_fallback(logger, message):
    print(message) if logger is None else logger.info(message)


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


# ---------------------------------------------------------------------------
# --high_artifact_mode helpers (Knob B + Knob C)
#
# Scope: opt-in mode for high-artifact data such as long-read snRNA-seq with
# nascent / pre-mRNA contamination. NOT a standard pipeline addition.
# Both filters live at step3 (Haplotyping) and are no-op when high_artifact_mode is OFF.
# ---------------------------------------------------------------------------

def _resolve_base_reference_pickle_path(scotch_target, logger=None):
    """Return path to SCOTCH base (non-augmented) gene structure pickle for cohort sample[0]."""
    ref_dir = os.path.join(scotch_target[0], 'reference')
    candidates = [
        'geneStructureInformation.pkl',       # base, non-meta (preferred)
        'metageneStructureInformation.pkl',   # base, meta (fallback, will be flattened)
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


def compute_nascent_leak_intervals(gsi_updated, logger=None):
    """For each known-ENST gene in the SCOTCH-augmented pkl, identify SCOTCH-novel-only sub-exons
    (those not referenced by any ENST isoform after SCOTCH update_isoform_info reindex) and compute
    intron_filled_pct = novel_exon_len / base_intron_len.

    Returns: {geneID: (intron_filled_pct, sorted_novel_intervals)}.
    Skips SCOTCH-only novel genes (geneID startswith 'gene_') and monoexonic / pathological cases
    where base_intron_len <= 0.
    """
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
    """For each known-ENST gene in the SCOTCH base pkl, return (chrom, geneStart, geneEnd, sorted_exon_intervals)
    as the canonical GENCODE exon set. Used by Knob C to score per-read intronic_pct.
    Skips SCOTCH-only novel genes (geneID startswith 'gene_').
    """
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
    """Sum bp of overlap between half-open block [s, e) and sorted non-overlapping
    [start, end) intervals. Linear scan (n_exons per gene is small).
    """
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


def compute_knob_c_blacklist(geneID, sample_index, bam_path_list, canonical_exons,
                             read_intronic_pct_max, candidate_reads, logger=None):
    """Open BAM, fetch reads in gene region, compute intronic_pct = intronic_aligned_bp/total_aligned_bp
    against the GENCODE canonical exon set. Return set of read names whose intronic_pct exceeds
    read_intronic_pct_max. Restricted to candidate_reads (the read pool actually entering EM)
    to avoid scanning reads we wouldn't use anyway.

    Returns empty set if canonical_exons is missing for this gene (e.g., SCOTCH-only gene_*) or
    BAM is unavailable.
    """
    info = canonical_exons.get(geneID) if canonical_exons else None
    if info is None:
        return set()
    chrom, gstart, gend, exon_intervals = info
    if not bam_path_list:
        return set()
    bam_path = bam_path_list[sample_index] if isinstance(bam_path_list, list) else bam_path_list
    if bam_path is None or not os.path.isfile(bam_path):
        return set()
    if not candidate_reads:
        return set()
    needed = set(candidate_reads)
    blacklist = set()
    seen_qnames = set()
    bam = pysam.AlignmentFile(bam_path, 'rb')
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
    finally:
        bam.close()
    return blacklist


EM_MISSING_CODE = -1
EM_REF_CODE = 0
EM_ALT_CODE = 1
EM_OTHER_CODE = 2

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
                 logger = None):
        self.logger = logger
        #only one target but can be multiple scotch and bam input
        self.sample_name_parse = sample_name_parse
        self.bam_path = bam_path
        self.target = target
        self.scotch_target = self._ensure_list(scotch_target if scotch_target is not None else target)
        self.sample_names = [os.path.basename(st) if sample_names is None else sample_names for st in self.scotch_target]
        self.n_samples = len(self.scotch_target)
        self.bam_path = self._ensure_list(bam_path)
        if self.sample_name_parse is not None:
            self.read_isoform_mapping_path = [os.path.join(st, f'samples/{str(sample_name_parse)}/auxillary/all_read_isoform_exon_mapping.tsv') for st in scotch_target]
        else:
            self.read_isoform_mapping_path = [os.path.join(st, 'auxillary/all_read_isoform_exon_mapping.tsv') for st in scotch_target]
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
        self.n_alt = n_alt
        self.min_mapq = min_mapq
        self.min_baseq = min_baseq
        self.min_dist_to_end = min_dist_to_end
        self.gene_subset = set(gene_subset) if gene_subset is not None else None
        self.het_prefilter_threshold = het_prefilter_threshold
        self._bam_path_cache = {}
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
    def heterozygous_prob_vec(depth, n_alt, e=0.01, priors=None):
        if priors is None:
            priors = (1 / 3, 1 / 3, 1 / 3)
        depth = np.asarray(depth, dtype=np.int64)
        n_alt = np.asarray(n_alt, dtype=np.int64)
        log_binom = gammaln(depth + 1) - gammaln(n_alt + 1) - gammaln(depth - n_alt + 1)
        log_p_het = log_binom + depth * np.log(0.5)
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
        match = re.search(r'_(ENSG[^_]+)_(?:isoform_agg(?:_balance|_unbalance)?)\.csv$', os.path.basename(filename))
        if match is None:
            raise ValueError(f'Could not extract geneID from filename: {filename}')
        return match.group(1)
    def _read_mapping(self):
        mapping_df_list = []
        for read_isoform_mapping_path in self.read_isoform_mapping_path:
            pieces = defaultdict(list)
            for chunk in pd.read_csv(read_isoform_mapping_path, sep='\t', chunksize=100000):
                chunk = chunk[chunk['Keep'] == 1][['Read', 'geneName', 'geneID', 'geneChr', 'Cell', 'Umi']]
                chunk['Read'] = chunk['Read'].str.replace(_READNAME_SUFFIX_RE, '', regex=True)
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
    def _get_bam_file_path(self, bam_path, chrom=None):
        if os.path.isfile(bam_path):
            return bam_path
        if chrom is None:
            raise ValueError(f'chrom is required when bam_path is a directory: {bam_path}')
        chrom_cache = self._bam_path_cache.get(bam_path)
        if chrom_cache is None:
            chrom_cache = {}
            chrom_pattern = re.compile(r'^chr(\d+|[XYM]|MT)$')
            for fname in os.listdir(bam_path):
                if not fname.endswith('.bam'):
                    continue
                parts = fname.replace('.bam', '').split('.')
                for p in parts:
                    if chrom_pattern.match(p):
                        chrom_cache.setdefault(p, os.path.join(bam_path, fname))
                        break
            self._bam_path_cache[bam_path] = chrom_cache
        bam_file = chrom_cache.get(chrom)
        if bam_file is None:
            raise FileNotFoundError(f'No BAM found for chromosome {chrom} in {bam_path}')
        return bam_file
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
    @staticmethod
    def _parse_pileup_bases(bases: str):
        """Return (ref_count, alt_count) from mpileup bases column."""
        i, n = 0, len(bases)
        ref_count = 0
        alt_count = 0
        while i < n:
            c = bases[i]
            if c == '^':  # start of read, next char is MAPQ
                i += 2
                continue
            if c == '$':  # end of read
                i += 1
                continue
            if c in '+-':  # indel: +/-<len><seq>
                i += 1
                j = i
                while j < n and bases[j].isdigit():
                    j += 1
                if j == i:  # malformed; bail on this char
                    i += 1
                    continue
                length = int(bases[i:j])
                i = j + length  # skip the indel sequence
                continue
            if c == '*':  # deletion placeholder (ignore)
                i += 1
                continue
            if c in '.,':  # match to reference
                ref_count += 1
                i += 1
                continue
            if c in 'ACGTNacgtn':  # explicit mismatch
                alt_count += 1
                i += 1
                continue
            # any other symbol: ignore
            i += 1
        return ref_count, alt_count
    def _samtools_snv_counts(self, bam_path_gene, ref_fasta, chrom, start, end):
        region_str = f"{chrom}:{start + 1}-{end}"
        cmd = ["samtools", "mpileup", "-f", ref_fasta, "-q", "0", "-Q", "0", "-r", region_str, bam_path_gene]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        rows = []
        for line in proc.stdout:# mpileup: chrom  pos(1-based)  ref  depth  bases  quals
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 5:
                continue
            chrom, pos1, ref, depth_raw, bases = parts[0], int(parts[1]), parts[2].upper(), int(parts[3]), parts[4]
            ref_count, alt_count = self._parse_pileup_bases(bases)
            eff_depth = ref_count + alt_count
            if eff_depth == 0 or alt_count == 0:
                continue
            alt_frac = alt_count / eff_depth
            rows.append((chrom, pos1 - 1, ref, eff_depth, alt_count, alt_frac))  # 0-based pos
        proc.wait()
        df = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "depth", "alt_count", "alt_frac"])
        return df
    def _reads_set_for_gene(self, geneID):
        #Get per-gene read set
        geneInfo, _, _ = self.geneStructureInformation[geneID]
        reads_set_list = []
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
    # Legacy helper retained for compatibility; the main pipeline now reads source BAMs directly.
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
    # Legacy helper retained for compatibility; the main pipeline now reads source BAMs directly.
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
    def _call_snvs_for_gene_interval(self, geneID, bam_path_gene, allowed_reads=None): #round1 or 2
        geneInfo, exonInfo, _ = self.geneStructureInformation[geneID]
        geneChr, geneStart, geneEnd = geneInfo['geneChr'], geneInfo['geneStart'], geneInfo['geneEnd']
        ref_len = self.ref_fasta.get_reference_length(geneChr)
        fetch_start = max(0, geneStart)
        fetch_end = min(ref_len, geneEnd + 1)
        ref_seq = self.ref_fasta.fetch(geneChr, fetch_start, fetch_end).upper()
        _het_prefilter_priors = (0.6, 0.1, 0.3)
        _het_prefilter_e = 0.01
        _het_prefilter_threshold = self.het_prefilter_threshold

        # --- Pass 1: count only, no site_reads collection ---
        rows = []
        bam = self._read_bam(bam_path=[bam_path_gene])
        try:
            for col in bam[0].pileup(
                geneChr, geneStart, geneEnd + 1,
                stepper="samtools", min_base_quality=self.min_baseq,
                min_mapping_quality=self.min_mapq, truncate=True,
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
                    if allowed_reads is not None and canonicalize_read_name(aln.query_name) not in allowed_reads:
                        continue
                    if pr.is_del or pr.is_refskip or pr.query_position is None:
                        continue
                    seq = aln.query_sequence
                    qpos = pr.query_position
                    if seq is None or qpos >= len(seq):
                        continue
                    base = seq[qpos].upper()
                    if self.min_dist_to_end > 0:
                        read_len = aln.query_length or aln.infer_read_length() or 0
                        if read_len > 0:
                            dist = min(qpos, read_len - 1 - qpos)
                            if base != ref_base and dist < self.min_dist_to_end:
                                continue
                    if base in {"A", "C", "G", "T"}:
                        if base == ref_base:
                            ref_count += 1
                        else:
                            alt_count += 1
                eff_depth = ref_count + alt_count
                if eff_depth == 0 or alt_count == 0:
                    continue
                rows.append((geneChr, pos0, ref_base, eff_depth, alt_count, alt_count / eff_depth))
        finally:
            bam[0].close()

        # Filter candidates by depth, alt_count, and het_prob prefilter
        snv_df = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "depth", "alt_count", "alt_frac"])
        snv_df = snv_df[(snv_df["depth"] >= self.depth) & (snv_df["alt_count"] > self.n_alt)].copy()
        snv_df = snv_df[(snv_df["pos"] >= geneStart) & (snv_df["pos"] <= geneEnd)].reset_index(drop=True)
        if snv_df.empty:
            return None, None
        snv_df['het_prob'] = self.heterozygous_prob_vec(
            snv_df['depth'].values, snv_df['alt_count'].values,
            e=_het_prefilter_e, priors=_het_prefilter_priors)
        if _het_prefilter_threshold >= 0:
            snv_df = snv_df[snv_df['het_prob'] >= _het_prefilter_threshold].reset_index(drop=True)
            if snv_df.empty:
                return None, None
        candidate_positions = set(snv_df['pos'].astype(int).tolist())

        # --- Pass 2: collect site_reads and del_counts only for candidate positions ---
        # NOTE: per-read CIGAR blocks (read_blocks) used by step5 obs_* live in a
        # separate step1.5 task (process_read_blocks_round1_5) — collecting them
        # inline here imposed ~4x wall regression on AD joint 12-donor (job
        # 9785242, 2026-05-23, 4h25min for 6%) from per-pileup-row overhead
        # plus 30-task petagene cache contention. step1.5 amortizes BAM fetch
        # one-task-per-BAM.
        site_reads_all = {}
        site_del_counts = {}
        bam = self._read_bam(bam_path=[bam_path_gene])
        try:
            for col in bam[0].pileup(
                geneChr, geneStart, geneEnd + 1,
                stepper="samtools", min_base_quality=0,
                min_mapping_quality=self.min_mapq, truncate=True,
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
                    canonical_qname = canonicalize_read_name(aln.query_name)
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
                    quals = aln.query_qualities or []
                    base = seq[qpos].upper()
                    baseq = int(quals[qpos]) if qpos < len(quals) else 0
                    if baseq < self.min_baseq:
                        continue
                    if self.min_dist_to_end > 0 and read_len > 0:
                        dist = min(qpos, read_len - 1 - qpos)
                        if base != ref_base and dist < self.min_dist_to_end:
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
        site_reads['__del_counts__'] = {
            (row.chrom, int(row.pos)): site_del_counts.get((row.chrom, int(row.pos)), 0)
            for row in snv_df.itertuples(index=False)
        }
        return snv_df, site_reads
    def process_genes_round1_1(self): #initial variant calling
        self.mapping_df_list = self._read_mapping()  # correspond to scotch output order
        os.makedirs(self.variants_by_gene_folder_1, exist_ok=True)
        geneIDs = list(self.geneStructureInformation.keys())
        if self.gene_subset is not None:
            geneIDs = [g for g in geneIDs if g in self.gene_subset]
        # Cost-aware chunk assignment: interleave by cost, then sort within chunk
        def _gene_cost(g):
            gInfo = self.geneStructureInformation[g][0]
            gene_len = max(1, gInfo['geneEnd'] - gInfo['geneStart'])
            n_reads = sum(len(d.get(g, [])) for d in self.mapping_df_list)
            return gene_len * max(1, n_reads)
        gene_costs = [(g, _gene_cost(g)) for g in geneIDs]
        gene_costs.sort(key=lambda x: -x[1])  # descending by cost
        # LPT scheduling: assign each gene to the chunk with lowest total cost
        chunk_costs = [0] * self.n_jobs
        chunk_genes = [[] for _ in range(self.n_jobs)]
        for g, cost in gene_costs:
            min_idx = chunk_costs.index(min(chunk_costs))
            chunk_genes[min_idx].append(g)
            chunk_costs[min_idx] += cost
        # Sort within each chunk by (chrom, start) for BAM locality
        for i in range(self.n_jobs):
            chunk_genes[i].sort(key=lambda g: (
                self.geneStructureInformation[g][0]['geneChr'],
                self.geneStructureInformation[g][0]['geneStart']))
        geneIDs_job = chunk_genes[self.job_index]
        # Completion gate uses site_reads.pkl only — read_blocks.pkl is built by
        # the separate step1.5 task (process_read_blocks_round1_5).
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
                except FileNotFoundError:
                    continue
                snv_df_sample, site_reads_sample = self._call_snvs_for_gene_interval(
                    geneID,
                    bam_path_gene,
                    allowed_reads=allowed_reads,
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
                        e=0.01, priors=(0.6, 0.1, 0.3)
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
        """step1.5 — per-BAM single-pass fetch + per-gene read_blocks.pkl.

        Decoupled from step1 to avoid the 4x wall regression hit when read-blocks
        collection ran inside the joint-variant-calling pileup loop on petagene
        BAMs (AD job 9785242, 2026-05-23). One SLURM task per BAM, 12 tasks
        parallel for a 12-donor cohort; each task scans its assigned BAM once
        and writes sample-tagged intermediate files
        `{geneID}_read_blocks_{job_index}.pkl`. After all per-BAM tasks finish,
        a single merge job (--task step1_5_merge) unions per-sample dicts into
        the canonical `{geneID}_read_blocks.pkl` consumed by step5.

        job_index addresses a BAM in self.bam_path. n_jobs must equal n_samples.
        """
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
        # Group genes by chromosome so we reuse one chrom-resolved BAM path per chunk.
        by_chrom = defaultdict(list)
        for g in geneIDs:
            info = self.geneStructureInformation[g][0]
            by_chrom[info['geneChr']].append(g)

        n_genes_written = 0
        n_genes_skipped = 0
        for chrom, gene_list in by_chrom.items():
            try:
                bam_path_chrom = self._get_bam_file_path(bam_path_root, chrom=chrom)
            except FileNotFoundError:
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
                        # Still write an empty file so the merge step has a
                        # predictable input set per sample.
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
        """step1.5 merge — combine per-sample {geneID}_read_blocks_{N}.pkl files
        into the canonical {geneID}_read_blocks.pkl consumed by step5.

        Single-task; runs after all process_read_blocks_round1_5 tasks done.
        Canonical read names are unique per physical read; cross-sample
        collisions only happen with duplicate-BAM / replicate-input cohorts
        (warning logged).
        """
        if not os.path.isdir(self.variants_by_gene_folder_1):
            return
        intermediate_re = re.compile(r'^(.+)_read_blocks_(\d+)\.pkl$')
        groups = defaultdict(list)  # {geneID: [(sample_index, path), ...]}
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
        # matrices (codes: -1=na, 0=ref, 1=alt, 2=other)
        r_code = np.full((nR, nS), -1, dtype=np.int8)
        pi_arr = np.full((nR, nS), np.nan, dtype=np.float32)
        alt_for_site = {}
        depth_alt = np.zeros((nS, 2), dtype=np.int32)  # [:,0]=depth, [:,1]=alt_count
        ACGT = {"A", "C", "G", "T"}
        def q_to_pi(q):
            try:
                qv = float(q)
            except (TypeError, ValueError):
                return 0.25
            if not np.isfinite(qv):
                return 0.25
            return float(np.clip(10.0 ** (-(qv / 10.0)), 0.0, 1.0))
        for j, (chrom, pos0, refb) in enumerate(sites):
            tuples = site_reads.get((chrom, pos0), set())
            if not tuples:
                alt_for_site[(chrom, pos0)] = None
                continue
            # ---- decide ALT (majority; tie -> lowest avg π) ----
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
                    else:# still count as support, but no quality for tie-break
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
                    # tie-break: lowest mean error prob (ignoring NaNs)
                    def mean_pi(b):
                        vals = [v for v in pis_by_base[b] if np.isfinite(v)]
                        return (np.mean(vals) if vals else np.inf)
                    alt = min(cands, key=lambda b: (mean_pi(b), b))
            alt_for_site[(chrom, pos0)] = alt
            # ---- second pass: fill matrices + depth/alt ----
            for tup in tuples:
                rn, qpos, base, q = tup[0], tup[1], tup[2], tup[3]
                i = read2row.get(rn)
                if i is None:
                    continue
                if r_code[i, j] != -1:
                    continue
                b = base.upper() if isinstance(base, str) else None
                pi = q_to_pi(q)
                if b is None or b not in ACGT:
                    code = -1  # na
                    pi = np.nan
                elif b == refb:
                    code = 0  # ref
                    depth_alt[j, 0] += 1
                elif alt is not None and b == alt:
                    code = 1  # alt
                    depth_alt[j, 0] += 1
                    depth_alt[j, 1] += 1
                else:
                    code = 2  # other
                    depth_alt[j, 0] += 1
                r_code[i, j] = code
                pi_arr[i, j] = pi
            # --- build snv_df_final (summary per site) ---
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
    def process_genes_final(self): #generate em input
        self.mapping_df_list = self._read_mapping()  # correspond to scotch output order
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
        for geneID in geneIDs_job:
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
                    print('input file already exist')
                    continue
            try:
                snv1, site_reads1 = self._load_round_outputs(geneID)
                if snv1 is None:
                    continue
                for ind, f_pile, f_npz in need:
                    snv_df_final, r_code, pi_arr, reads, cols = self._build_em_input_gene(geneID, snv1, site_reads1, ind)
                    _atomic_to_csv(snv_df_final, f_pile, index=False)
                    _save_em_input_npz(f_npz, r_code, pi_arr, reads, cols)
            except Exception as e:
                print(f"Error processing {geneID}: {e}")


class Haplotyping:
    def __init__(self, scotch_target:Union[str, Sequence[str]],
                 bam_path = None,
                 target = None, sample_names = None,
                 max_iter=50, tol=1e-3, verbose=False, seed = 42,
                 mtx = True, csv = False, n_jobs = 1, job_index = 0, n_alt = 10, depth = 20,
                 heterozygous_filter = -1, alt_stretch_filter = 20, alt_cluster_filter = 20,
                 het_fallback = False,
                 repeat_filter_kmer=1,
                 var_cluster_window = 20, var_cluster_n = 3,
                 sample_name_parse = None, prefix = 'LongAllele',
                 em_snv_filter = True, snv_confidence = None, snv_classifier = None,
                 clf_hard_threshold = 0.005, clf_init = False,
                 gap_tau = 0.10, clf_pruning_threshold = 0.1, clf_pruning_frac = 1.0,
                 rna_editing_db = None,
                 chi_min_frac = 0.10, chi_group_novel = False,
                 cell_type_df_path = None, ref_pickle_path = None, cover_existing = False, n_workers = -1,
                 ref_fasta_path=None, gene_subset = None, logger = None,
                 job_array_by_sample = False,
                 high_artifact_mode = False, novel_exon_pct_max = 0.25,
                 read_intronic_pct_max = 0.60, read_sj_min = 0,
                 gsi_base_pkl_path = None):
        #cell_type_df_path: need columns of Cell and CellType
        self.logger = logger
        self.n_workers = n_workers
        self.job_array_by_sample = job_array_by_sample
        self.cover_existing = cover_existing
        self.sample_name_parse = sample_name_parse
        self.em_snv_filter = em_snv_filter
        self.clf_hard_threshold = clf_hard_threshold
        self.clf_init = clf_init
        self.gap_tau = gap_tau
        self.clf_pruning_threshold = clf_pruning_threshold
        self.clf_pruning_frac = clf_pruning_frac
        self.var_cluster_window = var_cluster_window
        self.var_cluster_n = var_cluster_n
        self.alt_stretch_filter = alt_stretch_filter
        self.alt_cluster_filter = alt_cluster_filter
        self.repeat_filter_kmer = repeat_filter_kmer
        self.fasta_handle = pysam.FastaFile(ref_fasta_path) if ref_fasta_path is not None else None
        self.n_jobs = n_jobs
        self.job_index = job_index
        self.target = target #root folder for long allele results
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
        # BAM path for SNV classifier feature extraction
        self.bam_path = self._parse_comma_list(bam_path) if bam_path is not None else None
        # SNV classifier for h_m initialization (uses site_reads.pkl from step 1, no BAM needed)
        self.snv_classifier_model = None
        if snv_classifier is not None:
            self.snv_classifier_model = joblib_load(snv_classifier)
            mes = f'Loaded SNV classifier from {snv_classifier}'
            print(mes) if self.logger is None else self.logger.info(mes)
        if ref_pickle_path is not None:
            self.geneStructureInformation = _load_gene_structure_information(ref_pickle_path, self.logger)
        else:
            gsi_path = _resolve_reference_pickle_path(self.scotch_target, self.logger)
            self.geneStructureInformation = _load_gene_structure_information(gsi_path, self.logger)
        self.prefix = prefix or None
        if self.sample_name_parse is not None:
            self.read_isoform_mapping_path_list = [os.path.join(self.scotch_target[0],
                                                          f'samples/{str(self.sample_name_parse)}/auxillary/all_read_isoform_exon_mapping.tsv')]
        else:
            self.read_isoform_mapping_path_list = [os.path.join(self.scotch_target[i], 'auxillary/all_read_isoform_exon_mapping.tsv') for i in range(self.n_samples)]
        #self.mapping_df_dict_list = self._read_mapping()
        self.em_input = os.path.join(self.target, 'em_input')
        # em settings
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.seed = seed
        self.mtx = mtx
        self.csv = csv
        self.n_alt = n_alt
        self.depth = depth
        self.heterozygous_filter = heterozygous_filter
        self.het_fallback = het_fallback
        self.snv_confidence = snv_confidence #users can use variant caller to predefine confident snv sites: df, chrom, pos
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
        # --- high_artifact_mode (Knob B + Knob C) ---
        # Opt-in mode for high-artifact data (e.g. lr-snRNA-seq with nascent contamination).
        # Default OFF preserves the standard pipeline byte-for-byte.
        self.high_artifact_mode = bool(high_artifact_mode)
        self.novel_exon_pct_max = float(novel_exon_pct_max)
        self.read_intronic_pct_max = float(read_intronic_pct_max)
        # --- Knob D: read truncation filter (standalone, not haf-gated) ---
        # Drop reads with fewer than read_sj_min internal splice junctions before EM.
        # Mitigates EM-phasing bias from truncated single-block long-read fragments
        # (Karenxzr 2026-05-23 directive, ORMDL1 + HLA-DQB1 catch via littlefox).
        # Default 0 preserves the standard pipeline byte-for-byte.
        self.read_sj_min = int(read_sj_min)
        self.nascent_leak_intervals = None
        self.canonical_exons = None
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
            self.nascent_leak_intervals = compute_nascent_leak_intervals(
                self.geneStructureInformation, self.logger)
            mes = (f'[high_artifact_mode] enabled. '
                   f'Knob B cutoff (intron_filled_pct) = {self.novel_exon_pct_max}; '
                   f'Knob C cutoff (read intronic_pct) = {self.read_intronic_pct_max}. '
                   f'Scope: long-read snRNA-seq nascent-leak filters at step3.')
            # Always print to stdout AND log to file when logger is set, so the setting reaches
            # both SLURM .out and the pipeline log file regardless of joblib worker logger state.
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
        """Parse a string or list that may contain comma-separated values."""
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
        """Extract per-SNV features from cached site_reads for the classifier.
        site_reads_dict: {(chrom, pos): set of tuples} from step 1 site_reads.pkl.
        Tuples are (read_name, qpos, base, baseq, mapq, read_len, is_reverse).
        Returns DataFrame with SNV_CLF_FEATURE_COLUMNS, aligned to df_pileup_filtered index."""
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
                baseq = int(baseq)
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
                    alt_bqs.append(baseq)
                    if read_len > 1:
                        alt_positions.append(float(qpos) / max(read_len - 1, 1))
                    if is_reverse:
                        alt_rev += 1
                    else:
                        alt_fwd += 1
                elif base == ref_base:
                    ref_bqs.append(baseq)
                    if is_reverse:
                        ref_rev += 1
                    else:
                        ref_fwd += 1

            depth = len(mapqs)
            alt_ct = int(alt_base_counts.get(alt_base, 0))
            # Strand SOR (GATK-style, same as longcallR)
            x00, x01, x10, x11 = ref_fwd+1, ref_rev+1, alt_fwd+1, alt_rev+1
            sym = (x00*x11)/(x01*x10) + (x01*x10)/(x00*x11)
            ref_ratio = min(x00,x01) / max(x00,x01)
            alt_ratio = min(x10,x11) / max(x10,x11)
            strand_sor = float(np.log(sym) + np.log(ref_ratio) - np.log(alt_ratio))
            # Deletion fraction
            total_with_del = depth + del_count
            del_frac = del_count / total_with_del if total_with_del > 0 else 0.0

            # Sequence context features (from reference FASTA)
            gc_content_11bp = 0.5
            homopolymer_len = 1
            is_homopolymer_ge5 = 0
            creates_homopolymer = 0
            flanking_is_AT = 0
            is_transition = 0
            if self.fasta_handle is not None:
                try:
                    chrom_len = self.fasta_handle.get_reference_length(chrom)
                    fetch_start = max(0, pos0 - 5)
                    fetch_end = min(chrom_len, pos0 + 6)
                    seq = self.fasta_handle.fetch(chrom, fetch_start, fetch_end).upper()
                    if len(seq) < 11:
                        seq = seq + 'N' * (11 - len(seq))  # pad to match training
                    snv_idx = pos0 - fetch_start
                    # GC content
                    gc_content_11bp = sum(1 for b in seq if b in 'GC') / len(seq)
                    # Homopolymer: longest run in ±5bp window (matches training)
                    homopolymer_len = max((len(m.group()) for m in re.finditer(r'(.)\1*', seq)), default=1)
                    is_homopolymer_ge5 = 1 if homopolymer_len >= 5 else 0
                    # Trinucleotide context (needs flanking bases)
                    if 0 < snv_idx < len(seq) - 1:
                        trinuc = seq[snv_idx - 1:snv_idx + 2]
                        creates_homopolymer = 1 if (alt_base == trinuc[0] or alt_base == trinuc[2]) else 0
                        flanking_is_AT = 1 if (trinuc[0] in 'AT' and trinuc[2] in 'AT') else 0
                except Exception:
                    pass
            # Is transition? (no FASTA needed)
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
    def heterozygous_prob_vec(depth, n_alt, e=0.01, priors=None):
        if priors is None:
            priors = (1 / 3, 1 / 3, 1 / 3)
        depth = np.asarray(depth, dtype=np.int64)
        n_alt = np.asarray(n_alt, dtype=np.int64)
        log_binom = gammaln(depth + 1) - gammaln(n_alt + 1) - gammaln(depth - n_alt + 1)
        log_p_het = log_binom + depth * np.log(0.5)
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
    def heterozygous_prob_per_read_bq(site_reads_dict, df_pileup_filtered):
        """Compute P(het) using per-read base qualities and conservative prior.
        Uses actual BQ per read instead of uniform e=0.01.
        Prior (0.6, 0.1, 0.3) favors hom-ref over het."""
        log_priors = np.log(np.array([0.6, 0.1, 0.3]))  # hom_ref, het, hom_alt
        out = np.empty(len(df_pileup_filtered), dtype=float)
        for i, row in enumerate(df_pileup_filtered.itertuples(index=False)):
            chrom, pos0 = str(row.chrom), int(row.pos)
            ref_base = str(row.ref).upper()
            alt_base = str(row.alt).upper()
            tuples = site_reads_dict.get((chrom, pos0), set())
            log_post = log_priors.copy()
            n_valid = 0
            for _read_name, _qpos, base, baseq, _mapq, _read_len, _is_reverse in tuples:
                base = str(base).upper()
                if base not in {'A', 'C', 'G', 'T'}:
                    continue
                e = float(np.clip(10.0 ** (-int(baseq) / 10.0), 0.001, 0.5))
                p_match_het = 0.5 * (1.0 - e) + 0.5 * (e / 3.0)
                p_err = e / 3.0
                if base == ref_base:
                    log_post[0] += np.log(1.0 - e)      # hom_ref
                    log_post[1] += np.log(p_match_het)   # het
                    log_post[2] += np.log(p_err)         # hom_alt
                elif base == alt_base:
                    log_post[0] += np.log(p_err)         # hom_ref
                    log_post[1] += np.log(p_match_het)   # het
                    log_post[2] += np.log(1.0 - e)       # hom_alt
                else:
                    log_post += np.log(p_err)
                n_valid += 1
            if n_valid == 0:
                out[i] = Haplotyping.heterozygous_prob(int(row.depth), int(row.alt_count), e=0.01)
                continue
            m = np.max(log_post)
            den = np.exp(log_post - m).sum()
            out[i] = np.exp(log_post[1] - m) / den
        return out
    @staticmethod
    def _extract_gene_id_from_isoform_filename(filename):
        match = re.search(r'_(ENSG[^_]+)_(?:isoform_agg(?:_balance|_unbalance)?)\.csv$', os.path.basename(filename))
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
    def _append_isoform_agg_csv(self, file_paths, out_csv):
        if not file_paths:
            return
        if self._should_parallelize(file_paths):
            frames = Parallel(n_jobs=self.n_workers, prefer='threads')(
                delayed(self._load_isoform_agg_frame)(file_path)
                for file_path in file_paths
            )
        else:
            frames = [self._load_isoform_agg_frame(file_path) for file_path in file_paths]
        for idx, df in enumerate(frames):
            df.to_csv(out_csv, mode='w' if idx == 0 else 'a', header=(idx == 0))
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
    def bulk_lrt_allelic_balance_gene(self, df_r, df_pi, em_result):#alpha_hat_adjusted=None
        reads_keep_mask = em_result["reads_keep_mask"]
        #reads_keep_mask_subset = reads_keep_mask if celltype_mask is None else reads_keep_mask & celltype_mask
        # LRT bulk for H0: alpha = 0.5 vs H1: alpha != 0.5
        alpha_hat = float(np.mean(em_result["hat_I"][reads_keep_mask]))
        h_A_hat, h_m_hat = np.asarray(em_result["h_A"]).reshape(-1), np.asarray(em_result["h_m"]).reshape(-1)
        ll_alt = observed_loglikelihood(df_r=df_r.loc[reads_keep_mask], df_pi=df_pi.loc[reads_keep_mask],
                                        alpha=alpha_hat, h_A=h_A_hat, h_m=h_m_hat,
                                        kept_mask=em_result['kept_mask'])
        fit_null = run_em_fixed_alpha(df_r=df_r, df_pi=df_pi,
                                      alpha_fixed=0.5, max_iter=self.max_iter, tol=self.tol,
                                      verbose=self.verbose, seed=self.seed,
                                      heterozygous_priors=(0.4, 0.2, 0.4),
                                      heterozygous_coverage_factor=1, kept_mask=em_result['kept_mask'])
        ll_null = observed_loglikelihood(df_r=df_r.loc[reads_keep_mask], df_pi=df_pi.loc[reads_keep_mask],
                                         alpha=0.5, h_A=fit_null["h_A"], h_m=fit_null["h_m"],
                                         kept_mask=em_result['kept_mask'])
        # LRT statistic and p-value.c Wilks: 2*(ll_alt - ll_null) ~ chi^2_1 (1 df for alpha)
        lrt_stat = max(0.0, 2.0 * (ll_alt - ll_null))
        p_value = chi2.sf(lrt_stat, df=1)
        out = {"alpha_hat": min(alpha_hat, 1 - alpha_hat),
               "ll_alt": float(ll_alt),
               "ll_null": float(ll_null),
               "lrt_stat": float(lrt_stat),
               "p_value": float(p_value)}
        return out
    def ct_lrt_allelic_balance_gene(self, df_r, df_pi, em_result):#alpha_hat_adjusted=None
        reads_keep_mask = em_result["reads_keep_mask"]
        #reads_keep_mask_subset = reads_keep_mask if celltype_mask is None else reads_keep_mask & celltype_mask
        # LRT bulk for H0: alpha = 0.5 vs H1: alpha != 0.5
        alpha_hat = float(np.mean(em_result["hat_I"][reads_keep_mask]))
        h_A_hat, h_m_hat = np.asarray(em_result["h_A"]).reshape(-1), np.asarray(em_result["h_m"]).reshape(-1)
        ll_alt = observed_loglikelihood(df_r=df_r.loc[reads_keep_mask], df_pi=df_pi.loc[reads_keep_mask],
                                        alpha=alpha_hat, h_A=h_A_hat, h_m=h_m_hat,
                                        kept_mask=em_result['kept_mask'])
        ll_null = observed_loglikelihood(df_r=df_r.loc[reads_keep_mask], df_pi=df_pi.loc[reads_keep_mask],
                                         alpha=0.5, h_A=h_A_hat, h_m=h_m_hat,
                                         kept_mask=em_result['kept_mask'])
        # LRT statistic and p-value.c Wilks: 2*(ll_alt - ll_null) ~ chi^2_1 (1 df for alpha)
        lrt_stat = max(0.0, 2.0 * (ll_alt - ll_null))
        p_value = chi2.sf(lrt_stat, df=1)
        out = {"alpha_hat": min(alpha_hat, 1 - alpha_hat),
               "ll_alt": float(ll_alt),
               "ll_null": float(ll_null),
               "lrt_stat": float(lrt_stat),
               "p_value": float(p_value)}
        return out
    def _read_mapping(self):
        mapping_df_dict_list = []
        for read_isoform_mapping_path in self.read_isoform_mapping_path_list:
            pieces = defaultdict(list)
            for chunk in pd.read_csv(read_isoform_mapping_path, sep='\t', chunksize=100000):
                chunk = chunk[chunk['Keep'] == 1][['Read', 'geneName', 'geneID', 'Isoform', 'Cell', 'Umi']]
                chunk['Read'] = chunk['Read'].str.replace(_READNAME_SUFFIX_RE, '', regex=True)
                for gene_id, sub_df in chunk.groupby('geneID', sort=False):
                    pieces[gene_id].append(sub_df)
                del chunk  # release memory for the current chunk
            mapping_df_dict = {
                gene_id: pd.concat(parts, ignore_index=True)
                for gene_id, parts in pieces.items()
            }
            mapping_df_dict_list.append(mapping_df_dict)
            del pieces
            del mapping_df_dict
        return mapping_df_dict_list
    def _get_gene_name(self, geneID):
        for mapping_df_dict in self.mapping_df_dict_list:
            if geneID in mapping_df_dict:
                mapping_df_gene = mapping_df_dict[geneID]
                if not mapping_df_gene.empty:
                    return mapping_df_gene['geneName'].iloc[0]
        return None
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
    @staticmethod
    def _chisq_test(tab):
        out = {"chi2_isoform": None, "df_isoform": None, "p_value_isoform": None}
        O = np.asarray(tab, dtype=float)
        K = tab.shape[0]
        if K >=  2:
            row_tot = O.sum(axis=1, keepdims=True)  # k×1
            col_tot = O.sum(axis=0, keepdims=True)  # 1×2
            grand = col_tot.sum()
            E = row_tot @ (col_tot / grand)  # k×2
            # chi-square statistic (no Yates correction for k>2 rows)
            chi2_stat = float(np.sum((O - E) ** 2 / (E + 1e-12)))
            df = K - 1
            p_asym = float(1.0 - chi2.cdf(chi2_stat, df))
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
        if path is None:
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
                    hits = (idx < len(ag_positions)) & (ag_positions[idx] == query_positions)
                    is_editing[np.flatnonzero(mask)] = hits

            tc_positions = rna_editing_db.get('TC', {}).get(chrom)
            if tc_positions is not None and len(tc_positions) > 0:
                mask = chrom_mask & (refs == 'T') & (alts == 'C')
                if np.any(mask):
                    query_positions = positions[mask]
                    idx = np.searchsorted(tc_positions, query_positions)
                    hits = (idx < len(tc_positions)) & (tc_positions[idx] == query_positions)
                    is_editing[np.flatnonzero(mask)] = hits

        return is_editing

    def run_em_gene(self, geneID, sample_index = None):
        em_input = self.em_input if self.n_samples==1 else os.path.join(self.em_input, self.sample_names[sample_index])
        pileup_path = os.path.join(em_input, f'{geneID}_pileup.csv')
        read_npz_path = os.path.join(em_input, f'{geneID}_read_matrices.npz')
        read_snv_path = os.path.join(em_input, f'{geneID}_read_snv.csv')
        read_pi_path = os.path.join(em_input, f'{geneID}_read_pi.csv')
        df_pileup = pd.read_csv(pileup_path)
        df_read_snv, df_read_pi = _load_em_input(read_npz_path, read_snv_path, read_pi_path)
        # KNOB C — read-level nascent / pre-mRNA filter (high_artifact_mode only).
        # Drops reads whose intronic_aligned_bp / total_aligned_bp > read_intronic_pct_max.
        # Filtered reads disappear from EM input AND from mapping_df_gene below, so they do
        # NOT count as gene coverage, non-phasable, or count-matrix contributions.
        knob_c_blacklist = set()
        if self.high_artifact_mode and self.canonical_exons is not None:
            sidx = sample_index if sample_index is not None else 0
            n_reads_pre_c = int(df_read_snv.shape[0])
            knob_c_blacklist = compute_knob_c_blacklist(
                geneID, sidx, self.bam_path, self.canonical_exons,
                self.read_intronic_pct_max,
                candidate_reads=set(df_read_snv.index.astype(str).tolist()),
                logger=self.logger,
            )
            if knob_c_blacklist:
                keep_mask = ~df_read_snv.index.astype(str).isin(knob_c_blacklist)
                df_read_snv = df_read_snv.loc[keep_mask]
                df_read_pi = df_read_pi.loc[keep_mask]
            n_reads_post_c = int(df_read_snv.shape[0])
            n_dropped_c = n_reads_pre_c - n_reads_post_c
            if n_dropped_c > 0:
                _knob_c_msg = (
                    f'[KnobC] {geneID}: dropped {n_dropped_c} nascent reads '
                    f'({n_reads_pre_c} -> {n_reads_post_c}, '
                    f'cutoff intronic_pct > {self.read_intronic_pct_max})'
                )
                # Print AND log: joblib workers may not inherit the file handler, so
                # print() ensures the line reaches SLURM stdout regardless.
                print(_knob_c_msg)
                if self.logger is not None:
                    self.logger.info(_knob_c_msg)
            if n_reads_post_c == 0:
                _knob_c_skip = f'[KnobC] {geneID}: all reads dropped as nascent; skipping gene'
                print(_knob_c_skip)
                if self.logger is not None:
                    self.logger.info(_knob_c_skip)
                return None, None, None, None, None, None, None, None
        # KNOB D — read truncation filter (standalone, also recommended with haf).
        # Drops reads with fewer than read_sj_min internal splice junctions (CIGAR N
        # ops, sourced from step1.5's read_blocks.pkl). Single-block / 0-SJ reads
        # lack haplotype-distinguishing SNVs across junctions and get assigned to
        # the major hap by prior, inflating apparent allelic imbalance.
        if self.read_sj_min > 0:
            read_blocks_path = os.path.join(self.target, 'variant_align1',
                                            'variants_by_gene',
                                            f'{geneID}_read_blocks.pkl')
            read_blocks_pkl = load_pickle(read_blocks_path)
            if isinstance(read_blocks_pkl, dict):
                n_reads_pre_d = int(df_read_snv.shape[0])
                read_index = df_read_snv.index.astype(str)
                # Strict: missing pkl entry == 0 SJ → drop when read_sj_min > 0.
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
        df_pileup_filtered = df_pileup[
            (df_pileup.alt_count > self.n_alt) & (df_pileup.depth >= self.depth)].reset_index(drop=True)
        _n_before = len(df_pileup); _n_after_depth = len(df_pileup_filtered)  # DIAG
        if len(df_pileup_filtered)==0:
            _depth_max = df_pileup['depth'].max() if len(df_pileup) > 0 else None  # DIAG
            _alt_max = df_pileup['alt_count'].max() if len(df_pileup) > 0 else None  # DIAG
            _msg = (f'[DIAG] {geneID}: {_n_before} raw → 0 after depth/alt filter '  # DIAG
                    f'(n_alt={self.n_alt}, depth={self.depth}, '  # DIAG
                    f'max_depth={_depth_max}, max_alt={_alt_max}, '  # DIAG
                    f'dtypes={df_pileup[["depth","alt_count"]].dtypes.to_dict()}, '  # DIAG
                    f'pileup={pileup_path})')  # DIAG
            print(_msg) if self.logger is None else self.logger.info(_msg)  # DIAG
            return None, None, None, None, None, None, None, None
        df_pileup_filtered["ID"] = df_pileup_filtered["chrom"].astype(str) + "_" + df_pileup_filtered["pos"].astype(
            int).astype(str) + "_" + df_pileup_filtered["ref"].astype(str)
        # Load site_reads.pkl early (shared between het_prob and classifier features).
        # Schema is 7-tuple per utils.py:736; missing file → uniform-e fallback.
        _site_reads_dict = None
        _variants_dir = os.path.join(self.target, 'variant_align1', 'variants_by_gene')
        _site_reads_path = os.path.join(_variants_dir, f'{geneID}_site_reads.pkl')
        if os.path.exists(_site_reads_path):
            _site_reads_dict = load_pickle(_site_reads_path)
        if _site_reads_dict is not None:
            df_pileup_filtered['het_prob'] = self.heterozygous_prob_per_read_bq(
                _site_reads_dict, df_pileup_filtered)
        else:
            df_pileup_filtered['het_prob'] = self.heterozygous_prob_vec(
                df_pileup_filtered['depth'].values, df_pileup_filtered['alt_count'].values, e=0.01)
        if self.snv_confidence is not None:
            merge_keys = ['chrom', 'pos']
            if 'ref' in self.snv_confidence.columns:
                merge_keys.append('ref')
            snv_confidence_cols = merge_keys + [
                col for col in self.snv_confidence.columns
                if col not in merge_keys and col not in df_pileup_filtered.columns
            ]
            df_pileup_filtered = df_pileup_filtered.merge(
                self.snv_confidence.loc[:, snv_confidence_cols], how='inner', on=merge_keys
            )
        if self.heterozygous_filter >= 0 and self.snv_confidence is None:
            _n_pre_het = len(df_pileup_filtered)  # DIAG
            # exp(0.1) -- m exp(0.1+0.91) (m+sd) variants per 1000bp;
            geneInfo, exonInfo, _ = self.geneStructureInformation[geneID]  # short reads
            keep_n = math.ceil(6.6 * sum([b-a for a, b in exonInfo])/1000)
            n = min(keep_n, len(df_pileup_filtered))
            if n == 0:
                df_pileup_filtered = df_pileup_filtered.iloc[0:0]
            else:
                if self.het_fallback:
                    threshold = self.heterozygous_filter
                    df_candidates = df_pileup_filtered.iloc[0:0]
                    while threshold >= 0.5:
                        df_candidates = df_pileup_filtered[
                            df_pileup_filtered["het_prob"] >= threshold
                        ]
                        if len(df_candidates) > 0:
                            break
                        threshold = round(threshold - 0.05, 10)
                    if len(df_candidates) > 0:
                        df_pileup_filtered = df_candidates.nlargest(n, "het_prob").reset_index(drop=True)
                    else:
                        df_pileup_filtered = df_pileup_filtered.iloc[0:0]
                else:
                    df_pileup_filtered = df_pileup_filtered[
                        df_pileup_filtered["het_prob"] >= self.heterozygous_filter
                    ].reset_index(drop=True)
            _n_post_het = len(df_pileup_filtered)  # DIAG
        else:
            _n_pre_het = _n_post_het = len(df_pileup_filtered)  # DIAG
        #filter by homo-polymer stretch
        if len(df_pileup_filtered) > 0 and self.snv_confidence is None and self.fasta_handle is not None:
            df_pileup_filtered = df_pileup_filtered.sort_values(by=['pos']).reset_index(drop=True)
            chrom = df_pileup_filtered.chrom[0]
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
                    seq, 5, snv_pos_in_seq, max_kmer=self.repeat_filter_kmer
                )
            df_pileup_filtered['is_stretch'] = is_stretch
            df_pileup_filtered = df_pileup_filtered[
                (df_pileup_filtered['is_stretch'] == 0) | (df_pileup_filtered['alt_count'] >= self.alt_stretch_filter)]
            df_pileup_filtered = df_pileup_filtered.drop(columns=['is_stretch'])
        _n_post_stretch = len(df_pileup_filtered)  # DIAG
        # filter known RNA editing sites (canonical A-to-I only)
        if len(df_pileup_filtered) > 0 and self.snv_confidence is None and self.rna_editing_db is not None:
            df_pileup_filtered['is_rna_editing'] = self._is_known_rna_editing(
                df_pileup_filtered, self.rna_editing_db
            )
            df_pileup_filtered = df_pileup_filtered[
                ~df_pileup_filtered['is_rna_editing']
            ].reset_index(drop=True)
            df_pileup_filtered = df_pileup_filtered.drop(columns=['is_rna_editing'])
        _n_post_editing = len(df_pileup_filtered)  # DIAG

        #filter by variant cluster
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
        _n_post_cluster = len(df_pileup_filtered)  # DIAG
        _msg = (f'[DIAG] {geneID}: {_n_before} raw → {_n_after_depth} depth/alt(n_alt={self.n_alt},depth={self.depth}) → '  # DIAG
                f'{_n_post_het} het → {_n_post_stretch} stretch → '  # DIAG
                f'{_n_post_editing} editing → {_n_post_cluster} cluster')  # DIAG
        print(_msg) if self.logger is None else self.logger.info(_msg)  # DIAG
        # KNOB B — gene-level nascent-leak SNV mask (high_artifact_mode only, "filter 9").
        # If intron_filled_pct[geneID] > novel_exon_pct_max, drop SNVs falling in SCOTCH-novel-only
        # sub-exon intervals. Reads losing all markers will be flagged non-phasable by EM downstream
        # (semantically correct: those reads have no trustworthy phasing marker).
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
                    # Print AND log: joblib workers may not inherit the file handler, so
                    # print() ensures the line reaches SLURM stdout regardless.
                    print(_knob_b_msg)
                    if self.logger is not None:
                        self.logger.info(_knob_b_msg)
        if len(df_pileup_filtered) == 0:
            return None, None, None, None, None, None, None, None
        snv_list = df_pileup_filtered["ID"].tolist()
        df_read_snv_filtered = df_read_snv.loc[:, snv_list]  # r
        df_read_pi_filtered = df_read_pi.loc[:, snv_list]  # pi
        df_r, df_pi = df_read_snv_filtered, df_read_pi_filtered
        n_reads, n_snvs = df_r.shape
        gamma = float((df_r.to_numpy() == EM_MISSING_CODE).sum()) / (n_reads * n_snvs)
        # SNV classifier: hard filter removes confident artifacts
        _clf_prob_surviving = None  # clf_prob for SNVs surviving hard filter (for --clf_init)
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
                    # Hard filter: remove SNVs the classifier is confident are artifacts
                    hard_mask = clf_prob >= self.clf_hard_threshold
                    n_hard_removed = (~hard_mask).sum()
                    # Hard filter: remove SNVs below threshold (0 = no hard filter)
                    if self.clf_hard_threshold > 0 and n_hard_removed > 0:
                        df_pileup_filtered = df_pileup_filtered[hard_mask].reset_index(drop=True)
                        clf_prob = clf_prob[hard_mask]
                        mes = f'[DIAG] {geneID}: classifier hard filter removed {n_hard_removed} SNVs (clf_prob < {self.clf_hard_threshold}), {len(df_pileup_filtered)} remain'
                        print(mes) if self.logger is None else self.logger.info(mes)
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
        # h_m initialization: use clf_prob if --clf_init, otherwise compute_P_het (default)
        h_m_init = None
        if self.clf_init and _clf_prob_surviving is not None:
            h_m_init = _clf_prob_surviving
        # Iterative pruning: remove lowest-scoring SNVs until <=clf_pruning_frac have clf_prob < clf_pruning_threshold
        if h_m_init is not None and self.clf_pruning_frac < 1.0 and len(h_m_init) > 0:
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
                print(mes) if self.logger is None else self.logger.info(mes)
        em_snv_filter = bool(self.em_snv_filter and self.snv_confidence is None)
        results = run_em(df_r, df_pi, max_iter = self.max_iter, tol=self.tol, verbose=self.verbose, seed = self.seed,
                         heterozygous_priors=(0.4, 0.2, 0.4), heterozygous_coverage_factor = 1,
                         h_m_filter=em_snv_filter, filter_reads=True, h_m_init=h_m_init,
                         gap_tau=self.gap_tau)
        if np.sum(results['h_m']>0.5)==0:
            return None, None, None, None, None, None, None, None
        alpha_hat = results['alpha']
        df_pileup_filtered['h_A'] = results['h_A']
        df_pileup_filtered['h_m'] = results['h_m']
        df_pileup_filtered['hat_Z_binary'] = results['hat_Z_binary']
        read_hap_df = pd.DataFrame({'Read': df_r.index.tolist(),'reads_phasable': results['reads_keep_mask']+0, 'hat_I': results['hat_I']})
        read_hap_df["hat_I"] = read_hap_df["hat_I"]
        read_hap_df["hat_I_B"] = 1 - read_hap_df["hat_I"]
        rp, s = read_hap_df['reads_phasable'].eq(1).astype(int), read_hap_df['hat_I'] #alpha = s.mean()
        alpha_low, alpha_high = (rp * s).mean(), np.where(rp.eq(0), 1, s).mean()
        lrt_test = self.bulk_lrt_allelic_balance_gene(df_r, df_pi, results)
        mapping_df_gene = self.mapping_df_dict_list[sample_index][geneID].reset_index(drop=True)
        # Apply Knob C blacklist so nascent reads are absent from gene coverage, count matrix,
        # phasability metric, and per-cell-type aggregations downstream.
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
        else:
            alpha_hat_low, alpha_hat_high = alpha_low, alpha_high
        read_hap_df["hat_I"] = read_hap_df["hat_I"].round(3)
        read_hap_df["hat_I_B"] = read_hap_df["hat_I_B"].round(3)
        result_dict = {'geneID': geneID, 'geneName': geneName, 'gamma': gamma,
                       'n_reads': n_reads, 'n_reads_phasable': read_hap_df.reads_phasable.sum(),
                       'n_snvs': n_snvs, 'alpha_hat': alpha_hat,
                       'alpha_hat_low': alpha_hat_low, 'alpha_hat_high': alpha_hat_high,
                       'major_hap': major_hap_bulk,
                       'll_alt': lrt_test['ll_alt'], 'll_null': lrt_test['ll_null'],
                       'lrt_stat': lrt_test['lrt_stat'], 'p_value': lrt_test['p_value'],
                       'CellType': 'Bulk'}
        count_matrix_hapA.columns = [geneName + '_' + col + '_hapA' for col in count_matrix_hapA.columns.tolist()]
        count_matrix_hapB.columns = [geneName + '_' + col + '_hapB' for col in count_matrix_hapB.columns.tolist()]
        count_matrix_hapA_phasable.columns = [geneName + '_' + col + '_hapA' for col in count_matrix_hapA_phasable.columns.tolist()]
        count_matrix_hapB_phasable.columns = [geneName + '_' + col + '_hapB' for col in count_matrix_hapB_phasable.columns.tolist()]
        # cell type
        ct_results_df = None
        if self.cell_type_df_list is not None:
            ct_results_list = [result_dict] # bulk
            # --- celltype summaries ---
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
                    'major_hap': None,
                    'll_alt': None,
                    'll_null': None,
                    'lrt_stat': None,
                    'p_value': None,
                    'CellType': celltype}
                try:
                    df_r_ct = df_r.loc[celltype_mask]
                    df_pi_ct = df_pi.loc[celltype_mask]
                    result_ct = run_em(df_r_ct, df_pi_ct, max_iter=self.max_iter,
                                       tol=self.tol, verbose=self.verbose,
                                       seed=self.seed, heterozygous_priors=(0.4, 0.2, 0.4),
                                       heterozygous_coverage_factor=1,
                                       h_m_filter=self.em_snv_filter, filter_reads=True, results = results)
                    lrt_test_ct = self.ct_lrt_allelic_balance_gene(df_r_ct, df_pi_ct, result_ct)
                    read_hap_df_ct = pd.DataFrame(
                        {'Read': df_r_ct.index.tolist(), 'reads_phasable': result_ct['reads_keep_mask'] + 0,
                         'hat_I': result_ct['hat_I']})
                    read_hap_df_ct["hat_I"] = read_hap_df_ct["hat_I"]
                    read_hap_df_ct["hat_I_B"] = 1 - read_hap_df_ct["hat_I"]
                    rp, s = read_hap_df_ct['reads_phasable'].eq(1).astype(int), read_hap_df_ct['hat_I']  # alpha = s.mean()
                    alpha_ct_low, alpha_ct_high = (rp * s).mean(), np.where(rp.eq(0), 1, s).mean()
                    alpha_ct = result_ct['hat_I'].mean()
                    major_hap_ct = 'B' if alpha_ct <= 0.5 else 'A'
                    if major_hap_ct == 'A':
                        alpha_hat_low, alpha_hat_high = 1 - alpha_ct_high, 1 - alpha_ct_low
                    else:
                        alpha_hat_low = alpha_ct_low
                        alpha_hat_high = alpha_ct_high
                    read_hap_df_ct["hat_I"] = read_hap_df_ct["hat_I"].round(3)
                    read_hap_df_ct["hat_I_B"] = read_hap_df_ct["hat_I_B"].round(3)
                    row.update({
                        'n_reads_phasable': int(read_hap_df_ct.reads_phasable.sum()),
                        'alpha_hat': min(alpha_ct, 1 - alpha_ct),
                        'alpha_hat_low': alpha_hat_low,
                        'alpha_hat_high': alpha_hat_high,
                        'major_hap': major_hap_ct,
                        'll_alt': lrt_test_ct.get('ll_alt'),
                        'll_null': lrt_test_ct.get('ll_null'),
                        'lrt_stat': lrt_test_ct.get('lrt_stat'),
                        'p_value': lrt_test_ct.get('p_value')})
                except Exception as e:
                    if getattr(self, 'verbose', False):
                        print(f"LRT failed for {geneID}/{celltype}: {e}")
                ct_results_list.append(row)
            ct_results_df = pd.DataFrame(ct_results_list)
        return count_matrix_hapA, count_matrix_hapB, count_matrix_hapA_phasable, count_matrix_hapB_phasable, result_dict, df_pileup_filtered, read_hap_df, ct_results_df  # isoform_phasability_dict, unphasable_read_counts_by_isoform
    def generate_count_hap_gene(self, geneID, sample_index):
        geneName = self._get_gene_name(geneID)
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
        if os.path.exists(em_result_file_path) == False or self.cover_existing: # skip the gene if existed or cover_existing
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
                    read_hap_df.to_csv(read_hap_path)
                    tab = self._aggregate_isoform_hap_table(count_matrix_hapA_phasable, count_matrix_hapB_phasable,
                                                            min_frac=self.chi_min_frac, group_novel = self.chi_group_novel)
                    tab_balance = self._aggregate_isoform_hap_table(count_matrix_hapA, count_matrix_hapB,
                                                                    min_frac=self.chi_min_frac,
                                                                    group_novel=self.chi_group_novel)  # conservative
                    tab_imbalance = self._most_imbalanced_tab(tab, tab_balance)
                    tab.to_csv(isoform_agg_file_path)
                    tab_balance.to_csv(isoform_agg_balance_file_path)
                    tab_imbalance.to_csv(isoform_agg_imbalance_file_path)
                    out, out_imbalance, out_balance = self._chisq_test(tab), self._chisq_test(tab_imbalance), self._chisq_test(tab_balance)
                    pvals = [out_balance.get('p_value_isoform'), out_imbalance.get('p_value_isoform'), out.get('p_value_isoform')]
                    pvals = [v for v in pvals if v is not None and not pd.isna(v)]
                    out['p_value_isoform_high'] = max(pvals) if pvals else None
                    out['p_value_isoform_low'] = min(pvals) if pvals else None
                    df1 = pd.DataFrame([result_dict])
                    df2 = pd.DataFrame([out])
                    df = pd.concat([df1, df2], axis=1)
                    df["CellType"] = 'Bulk'
                    ##CELL TYPE SPECIFIC
                    dfs_ct = []
                    if self.cell_type_df_list is not None: #cell type
                        celltype_cells = self.cell_type_df_list[sample_index].groupby("CellType")["Cell"].apply(list).to_dict()
                        for ct, cells in celltype_cells.items():
                            # subset count matrices
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
                            tab_imbalance_ct = self._most_imbalanced_tab(tab_ct, tab_balance_ct)
                            # Save cell-type-specific isoform tables
                            safe_ct = ct.replace('/', '_').replace(' ', '_')
                            ct_iso_dir = os.path.join(self.count_hap_folder_path, 'ct_isoform_separate', safe_ct)
                            tab_ct.to_csv(os.path.join(ct_iso_dir, f'{geneName}_{geneID}_isoform_agg.csv'))
                            tab_balance_ct.to_csv(os.path.join(ct_iso_dir, f'{geneName}_{geneID}_isoform_agg_balance.csv'))
                            tab_imbalance_ct.to_csv(os.path.join(ct_iso_dir, f'{geneName}_{geneID}_isoform_agg_unbalance.csv'))
                            out_ct = self._chisq_test(tab_ct)
                            out_ct_imbalance = self._chisq_test(tab_imbalance_ct)
                            out_ct_balance = self._chisq_test(tab_balance_ct)
                            pvals_ct = [out_ct_balance.get("p_value_isoform"),
                                        out_ct_imbalance.get("p_value_isoform"),
                                        out_ct.get("p_value_isoform")]
                            pvals_ct = [v for v in pvals_ct if v is not None and not pd.isna(v)]
                            out_ct["p_value_isoform_high"] = max(pvals_ct) if pvals_ct else None
                            out_ct["p_value_isoform_low"] = min(pvals_ct) if pvals_ct else None
                            df_ct = pd.concat([pd.DataFrame([result_dict]), pd.DataFrame([out_ct])], axis=1)
                            df_ct["CellType"] = ct
                            dfs_ct.append(df_ct)
                        df = pd.concat([df] + dfs_ct, ignore_index=True)
                        df_ = df[['chi2_isoform','df_isoform','p_value_isoform','p_value_isoform_low','p_value_isoform_high','CellType']]
                        df = ct_results_df.merge(df_, on = 'CellType')
                        df['CellType'] = df.pop('CellType')
                    df.to_csv(em_result_file_path, index=False)
                    _msg = f"Results Saved for {geneID}"  # DIAG
                    print(_msg) if self.logger is None else self.logger.info(_msg)  # DIAG
                else:
                    _msg = f'Empty results for {geneID}'  # DIAG
                    print(_msg) if self.logger is None else self.logger.info(_msg)  # DIAG
            except Exception as e:
                import traceback  # DIAG
                _emsg = f"Error processing {geneID}: {e}\n{traceback.format_exc()}"  # DIAG
                print(_emsg) if self.logger is None else self.logger.error(_emsg)  # DIAG
    def _generate_count_hap_gene_safe(self, geneID, sample_index):
        with warnings.catch_warnings():
             warnings.simplefilter("ignore")  # hide Python warnings here
             with np.errstate(all="ignore"):
                 self.generate_count_hap_gene(geneID, sample_index)
    def generate_count_hap_genes(self):
        mes = f'Perform haplotype phasing for {self.n_samples} sample in total'
        print(mes) if self.logger is None else self.logger.info(mes)
        mes = f'Load SCOTCH read mapping information'
        print(mes) if self.logger is None else self.logger.info(mes)
        self.mapping_df_dict_list = self._read_mapping()
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
            mes = f'{len(geneIDs_job)} genes in this job for sample index {i}'
            print(mes) if self.logger is None else self.logger.info(mes)
            Parallel(n_jobs=self.n_workers)(delayed(self._generate_count_hap_gene_safe)(geneID, i) for geneID in geneIDs_job)
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
    def generate_count_matrix(self):
        for i in self._iter_sample_indices():
            self._get_paths(sample_index = i)
            all_genes_dir = os.path.join(self.count_hap_folder_path, 'all_genes')
            all_genes_sep_isoform_dir = os.path.join(self.count_hap_folder_path, 'all_genes_isoform_separate')
            os.makedirs(all_genes_dir, exist_ok=True)
            #isoform aggregation
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
            iso_agg_csv = os.path.join(all_genes_dir, 'isoform_agg.csv')
            iso_agg_balance_csv = os.path.join(all_genes_dir, 'isoform_agg_balance.csv')
            iso_agg_unbalance_csv = os.path.join(all_genes_dir, 'isoform_agg_unbalance.csv')
            self._append_isoform_agg_csv(isoform_agg_files, iso_agg_csv)
            self._append_isoform_agg_csv(isoform_agg_balance_files, iso_agg_balance_csv)
            self._append_isoform_agg_csv(isoform_agg_unbalance_files, iso_agg_unbalance_csv)
            # Cell-type-specific isoform aggregation
            ct_iso_sep_dir = os.path.join(self.count_hap_folder_path, 'ct_isoform_separate')
            if os.path.isdir(ct_iso_sep_dir):
                for safe_ct in os.listdir(ct_iso_sep_dir):
                    ct_dir = os.path.join(ct_iso_sep_dir, safe_ct)
                    if not os.path.isdir(ct_dir):
                        continue
                    for tag, suffix in [('isoform_agg', '_isoform_agg.csv'),
                                        ('isoform_agg_balance', '_isoform_agg_balance.csv'),
                                        ('isoform_agg_unbalance', '_isoform_agg_unbalance.csv')]:
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
            triple_transcript_list, triple_gene_list = self._collect_count_triples(hapA_files)
            adata_gene = self._generate_adata(triple_gene_list)
            adata_transcript = self._generate_adata(triple_transcript_list)
            triple_transcript_phasable_list, triple_gene_phasable_list = self._collect_count_triples(hapA_phasable_files)
            adata_gene_phasable = self._generate_adata(triple_gene_phasable_list)
            adata_transcript_phasable = self._generate_adata(triple_transcript_phasable_list)
            if self.mtx:
                # gene level
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
                #transcript level
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
            # Cell-type-specific count matrices
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
            #part1 --- summary
            mes = f'summarizing summary files for sample index {i}'
            print(mes) if self.logger is None else self.logger.info(mes)
            summary_statistics_path = os.path.join(self.summary_statistics_path, 'all_genes_separate')
            files = [os.path.join(summary_statistics_path, f) for f in os.listdir(summary_statistics_path) if f.endswith('summary.csv')]
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
            # part2 --- read - hap
            mes = f'summarizing read-haplotype mapping files for sample index {i}'
            print(mes) if self.logger is None else self.logger.info(mes)
            read_hap_path = os.path.join(self.snv_hap_path, 'all_genes_separate')
            files = [os.path.join(read_hap_path, f) for f in os.listdir(read_hap_path) if f.endswith('read_hap.csv')]
            df_list = []
            for file in files:
                geneName, geneID = os.path.basename(file).split('_')[:2]
                df_ = pd.read_csv(file, index_col=0)
                df_['geneName'] = geneName
                df_['geneID'] = geneID
                df_list.append(df_)
            df = pd.concat(df_list).reset_index(drop=True)
            filepath = os.path.join(self.snv_hap_path, 'read_hap_map.csv')
            df.to_csv(filepath)
            mes = f'read-haplotype mapping file saved for sample index {i} at {filepath}'
            print(mes) if self.logger is None else self.logger.info(mes)
            #part3 --- snv - hap
            mes = f'summarizing snv-haplotype mapping files for sample index {i}'
            print(mes) if self.logger is None else self.logger.info(mes)
            snv_hap_path = os.path.join(self.snv_hap_path, 'all_genes_separate_snv')
            files = [os.path.join(snv_hap_path, f) for f in os.listdir(snv_hap_path) if f.endswith('.csv')]
            df_list = []
            for file in files:
                geneName, geneID = os.path.basename(file).split('_')[:2]
                geneID = geneID.replace('.csv', '')
                df_ = pd.read_csv(file)
                df_['geneName'] = geneName
                df_['geneID'] = geneID
                df_list.append(df_)
            df = pd.concat(df_list).reset_index(drop=True)
            df = _collapse_legacy_merge_suffixes(
                df, lambda mes: print(mes) if self.logger is None else self.logger.warning(mes)
            )
            filepath=os.path.join(self.snv_hap_path, 'snv_hap_map.csv')
            df.to_csv(filepath)
            mes = f'read-haplotype mapping file saved for sample index {i} at {filepath}'
            print(mes) if self.logger is None else self.logger.info(mes)
