import argparse
import logging
import os
import re
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
import src.utils as u
import src.downstream as d
import pandas as pd


parser = argparse.ArgumentParser(description='LongAllele')
parser.add_argument('--task', type=str)  # step1, step1_5, step1_5_merge, step2, step3, step4, step5, check
parser.add_argument('--output_folder', type=str) #a single output folder to store joint-called variants
parser.add_argument('--scotch_target', type=str, nargs='+')
parser.add_argument('--sample_names', type=str, nargs='+')
parser.add_argument('--sample_name_parse', type=str)
parser.add_argument('--n_alt_count', type=int, default=10)
parser.add_argument('--depth', type=int, default=20)
parser.add_argument('--n_jobs', type=int, default=1)
parser.add_argument('--job_index', type=int, default=0)


parser.add_argument('--cover_existing',action='store_true')
parser.add_argument('--cover_existing_false', action='store_false',dest='cover_existing')

#variant calling + haplotyping
parser.add_argument('--ref_fasta_path', type=str, default=None)

#variant calling
parser.add_argument('--bam_path', type=str, nargs='+')
parser.add_argument('--ref_pickle_path', type=str, help='(optional), assign a reference pickle file for variant callilng')

#haplotyping
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--mtx',action='store_true')
parser.add_argument('--csv',action='store_true')
parser.add_argument('--heterozygous_filter', type=float, default=0.99) #set negative: no filter, set 0: auto filter top n (conservative), or set a positive value
parser.add_argument('--het_fallback', action='store_true')
parser.add_argument('--em_snv_filter', action='store_true', default=True) #post-filter SNVs before EM (default on)
parser.add_argument('--no_em_snv_filter', dest='em_snv_filter', action='store_false') #disable em_snv_filter
parser.add_argument('--snv_classifier', type=str, default=None,
                    help='Path to serialized SNV classifier (.joblib) for hard filtering before EM')
parser.add_argument('--clf_hard_threshold', type=float, default=0.05,
                    help='Remove SNVs with classifier score below this (hard filter before EM)')
parser.add_argument('--clf_init', action='store_true',
                    help='Use classifier scores to initialize h_m in EM (after hard filter)')
parser.add_argument('--gap_tau', type=float, default=1.0,
                    help='Gap threshold for adaptive_keep_mask / classifier gap filter (1.0 = disabled, set 0.10 to enable)')
parser.add_argument('--clf_pruning_threshold', type=float, default=0.1,
                    help='clf_prob threshold below which SNVs are considered low-scoring for pruning')
parser.add_argument('--clf_pruning_frac', type=float, default=1.0,
                    help='Max fraction of low-scoring SNVs allowed (1.0 = no pruning)')
parser.add_argument('--var_cluster_window', type=int, default=20)
parser.add_argument('--var_cluster_n', type=int, default=3)
parser.add_argument('--alt_cluster_filter', type=int, default=150)
parser.add_argument('--alt_stretch_filter', type=int, default=50)
parser.add_argument('--repeat_filter_kmer', type=int, default=1)
parser.add_argument('--min_mapq', type=int, default=20)
parser.add_argument('--min_baseq', type=int, default=5)
parser.add_argument('--min_dist_to_end', type=int, default=3)
parser.add_argument('--chi_min_frac', type=float, default=0.1)
parser.add_argument('--chi_group_novel',action='store_true')
parser.add_argument('--prefix',type=str)
#predefined snv set
parser.add_argument('--snv_confidence_path', type=str)
parser.add_argument(
    '--rna_editing_db',
    type=str,
    default=os.path.join(os.path.dirname(__file__), 'rna_editing_hg38.npz'),
    help='Path to compact RNA editing DB (.npz). Default: bundled hg38 database.'
)
#gene subset: plain-text file with one geneID per line
parser.add_argument('--gene_subset_path', type=str)
#Cell CellType mappipng df
parser.add_argument('--cell_type_df_path', type=str, nargs='+')
#summary
parser.add_argument('--summary_haplotype',action='store_true')
parser.add_argument('--summary_count',action='store_true')
#downstream (step5)
parser.add_argument('--event_min_reads', type=int, default=10)
parser.add_argument('--snv_event_distance', type=int, default=50)
parser.add_argument('--n_workers', type=int, default=1,
                    help='Number of parallel workers for step5 downstream analysis')
parser.add_argument('--astu_sig_only', action='store_true')
parser.add_argument('--astu_sig_from_bulk', action='store_true',
                    help='When filtering Task 4 by ASTU significance, derive the significant gene set from Bulk rows and reuse it for all cell types.')
parser.add_argument('--astu_sig_threshold', type=float, default=0.05)
parser.add_argument('--event_mode', type=str, default='all_events',
                    choices=['all_events', 'switching_events', 'fdr_events'],
                    help='Event selection mode: all_events (default, test all), '
                         'switching_events (isoform-switching boundary events only), '
                         'fdr_events (events passing FDR cutoff)')
parser.add_argument('--fdr_events_value', type=float, default=0.05,
                    help='FDR cutoff for fdr_events mode')
parser.add_argument('--job_array_by_sample', action='store_true',
                    help='When set for step4/step5, process only sample job_index from the full multi-sample input lists.')
# --- high_artifact_mode (Knob B + Knob C; opt-in for lr-snRNA-seq nascent leak) ---
parser.add_argument('--high_artifact_mode', action='store_true',
                    help='Enable Knob B (gene-level SCOTCH-novel SNV mask) + Knob C (read-level '
                         'nascent / pre-mRNA filter) at step3. Designed for high-artifact data such '
                         'as long-read snRNA-seq with nascent contamination. Default OFF preserves '
                         'the standard pipeline byte-for-byte.')
parser.add_argument('--novel_exon_pct_max', type=float, default=0.25,
                    help='Knob B cutoff: per-gene novel_exon_len / base_intron_len. Genes above '
                         'cutoff drop SNVs falling in SCOTCH-novel-only sub-exon intervals. '
                         'Active only with --high_artifact_mode. Default 0.25 (interim, calibrated '
                         'on AD1 distribution).')
parser.add_argument('--read_intronic_pct_max', type=float, default=0.60,
                    help='Knob C cutoff: per-read intronic_aligned_bp / total_aligned_bp against '
                         'GENCODE canonical exon set. Reads above cutoff are dropped from EM '
                         'input AND from gene coverage / count matrix. '
                         'Active only with --high_artifact_mode. Default 0.60 (interim).')
parser.add_argument('--read_sj_min', type=int, default=0,
                    help='Knob D: drop reads with fewer than N internal splice junctions '
                         '(CIGAR N ops) at step3 before EM phasing. Mitigates EM-phasing bias '
                         'from truncated single-block long-read fragments that lack '
                         'haplotype-distinguishing SNVs and get assigned to the major hap by '
                         'prior. Default 0 (no filter, backward compat). Recommended >= 1 for '
                         'any hap-resolved ASE/APA claim. Requires read_blocks.pkl from step1.5 '
                         '(--task step1_5 per-sample array + --task step1_5_merge); datasets '
                         'without it log a warning and skip the filter.')
parser.add_argument('--gsi_base_pkl_path', type=str, default=None,
                    help='Optional explicit path to SCOTCH base (non-augmented) gene structure '
                         'pickle used by --high_artifact_mode. If omitted, auto-resolved from '
                         'scotch_target[0]/reference/.')


def setup_logger(target, task_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_file = os.path.join(target, f'{task_name}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_file



def load_gene_subset(path):
    if path is None:
        return None
    if os.path.isfile(path):
        with open(path) as fh:
            return [line.strip() for line in fh if line.strip()]
    return [g.strip() for g in path.split(',') if g.strip()]


def write_done_marker(output_folder, step, job_index=None):
    """Write a marker file on successful job completion.

    Files land in {output_folder}/job_markers/ and are named
    {step}_job{job_index}.done (or {step}.done for steps without array jobs).
    Check completion with:
        ls job_markers/step3_*.done | wc -l
    """
    import datetime
    marker_dir = os.path.join(output_folder, 'job_markers')
    os.makedirs(marker_dir, exist_ok=True)
    fname = (f'{step}_job{job_index}.done' if job_index is not None
             else f'{step}.done')
    with open(os.path.join(marker_dir, fname), 'w') as fh:
        fh.write(datetime.datetime.now().isoformat() + '\n')


def main():
    global args
    args = parser.parse_args()

    # step1
    def variant_calling():
        logger, _ = setup_logger(args.output_folder, 'step1_variantcalling')
        logger.info('Start running step1: initial variant calling...')
        logger.info(f'total jobs: {args.n_jobs}')
        logger.info(f'this job is: {args.job_index}')
        logger.info(f'Output directory: {args.output_folder}')
        logger.info(f'SCOTCH directory: {args.scotch_target}')
        logger.info(f'bam file paths: {args.bam_path}')
        logger.info(f'ref_fasta_path set as: {args.ref_fasta_path}')
        logger.info(f'ref_pickle_path set as: {args.ref_pickle_path}')
        logger.info(f'n_alt_count set as: {args.n_alt_count}')
        logger.info(f'depth set as: {args.depth}')
        logger.info(f'gene_subset_path set as: {args.gene_subset_path}')
        het_prefilter = args.heterozygous_filter * (0.8 / 0.99) if args.heterozygous_filter > 0 else -1
        logger.info(f'het_prefilter_threshold set as: {het_prefilter}')
        vc = u.VariantCaller(scotch_target=args.scotch_target, bam_path=args.bam_path,
                             ref_fasta_path=args.ref_fasta_path, ref_pickle_path=args.ref_pickle_path,
                             target=args.output_folder,
                             n_jobs=args.n_jobs, job_index=args.job_index,
                             depth=args.depth, n_alt=args.n_alt_count,
                             min_mapq=args.min_mapq, min_baseq=args.min_baseq,
                             min_dist_to_end=args.min_dist_to_end,
                             sample_name_parse=args.sample_name_parse,
                             sample_names=args.sample_names,
                             gene_subset=load_gene_subset(args.gene_subset_path),
                             het_prefilter_threshold=het_prefilter,
                             logger=logger)
        vc.process_genes_round1_1()
        write_done_marker(args.output_folder, 'step1', args.job_index)
        logger.info(f'Finished initial variant calling for job {args.job_index}')

    # step1.5: per-BAM read_blocks dump (decoupled from step1 to avoid the
    # joint-variant-call pileup overhead — see commit notes for AD job 9785242).
    def collect_read_blocks():
        logger, _ = setup_logger(args.output_folder, 'step1_5_readblocks')
        logger.info('Start running step1.5: per-BAM read_blocks collection...')
        logger.info(f'sample_index (job_index): {args.job_index} / n_samples={args.n_jobs}')
        logger.info(f'bam file paths: {args.bam_path}')
        het_prefilter = args.heterozygous_filter * (0.8 / 0.99) if args.heterozygous_filter > 0 else -1
        vc = u.VariantCaller(scotch_target=args.scotch_target, bam_path=args.bam_path,
                             ref_fasta_path=args.ref_fasta_path, ref_pickle_path=args.ref_pickle_path,
                             target=args.output_folder,
                             n_jobs=args.n_jobs, job_index=args.job_index,
                             depth=args.depth, n_alt=args.n_alt_count,
                             min_mapq=args.min_mapq, min_baseq=args.min_baseq,
                             min_dist_to_end=args.min_dist_to_end,
                             sample_name_parse=args.sample_name_parse,
                             sample_names=args.sample_names,
                             gene_subset=load_gene_subset(args.gene_subset_path),
                             het_prefilter_threshold=het_prefilter,
                             logger=logger)
        vc.process_read_blocks_round1_5()
        write_done_marker(args.output_folder, 'step1_5', args.job_index)
        logger.info(f'Finished read_blocks collection for sample {args.job_index}')

    def merge_read_blocks():
        logger, _ = setup_logger(args.output_folder, 'step1_5_merge')
        logger.info('Start running step1.5 merge: combine per-sample intermediate pkls...')
        vc = u.VariantCaller(scotch_target=args.scotch_target, bam_path=args.bam_path,
                             ref_fasta_path=args.ref_fasta_path, ref_pickle_path=args.ref_pickle_path,
                             target=args.output_folder,
                             n_jobs=1, job_index=0,
                             depth=args.depth, n_alt=args.n_alt_count,
                             min_mapq=args.min_mapq, min_baseq=args.min_baseq,
                             min_dist_to_end=args.min_dist_to_end,
                             sample_name_parse=args.sample_name_parse,
                             sample_names=args.sample_names,
                             gene_subset=load_gene_subset(args.gene_subset_path),
                             het_prefilter_threshold=-1,
                             logger=logger)
        vc.merge_read_blocks_round1_5()
        write_done_marker(args.output_folder, 'step1_5_merge', 0)
        logger.info('Finished step1.5 merge')

    #step2
    def generate_em_input():
        logger, _ = setup_logger(args.output_folder, 'step2_eminput')
        logger.info('Start running step2: generating input for em...')
        logger.info(f'total jobs: {args.n_jobs}')
        logger.info(f'this job is: {args.job_index}')
        logger.info(f'Output directory: {args.output_folder}')
        logger.info(f'SCOTCH directory: {args.scotch_target}')
        logger.info(f'ref_pickle_path set as: {args.ref_pickle_path}')
        logger.info(f'gene_subset_path set as: {args.gene_subset_path}')
        vc = u.VariantCaller(scotch_target=args.scotch_target, bam_path=args.bam_path,
                             ref_fasta_path=args.ref_fasta_path, ref_pickle_path=args.ref_pickle_path,
                             target=args.output_folder,
                             n_jobs=args.n_jobs, job_index=args.job_index,
                             depth=args.depth, n_alt=args.n_alt_count,
                             min_mapq=args.min_mapq, min_baseq=args.min_baseq,
                             min_dist_to_end=args.min_dist_to_end,
                             sample_name_parse=args.sample_name_parse,
                             sample_names=args.sample_names,
                             gene_subset=load_gene_subset(args.gene_subset_path),
                             logger=logger)
        vc.process_genes_final()
        write_done_marker(args.output_folder, 'step2', args.job_index)
        logger.info(f'Finished generating em input files for job {args.job_index}')

    # step3
    def haplotyping():
        logger, _ = setup_logger(args.output_folder, 'step3_haplotyping')
        logger.info('Start running step3: haplotyping...')
        logger.info(f'total jobs: {args.n_jobs}')
        logger.info(f'this job is: {args.job_index}')
        logger.info(f'Output directory: {args.output_folder}')
        logger.info(f'SCOTCH directory: {args.scotch_target}')
        logger.info(f'seed set as: {args.seed}')
        logger.info(f'max iteration set as: {args.max_iter}')
        logger.info(f'tol set as: {args.tol}')
        logger.info(f'heterozygous_filter set as: {args.heterozygous_filter}')
        logger.info(f'het_fallback: {args.het_fallback}')
        logger.info(f'em_snv_filter: {args.em_snv_filter}')
        logger.info(f'snv_classifier: {args.snv_classifier}')
        logger.info(f'n_alt_count set as: {args.n_alt_count}')
        logger.info(f'depth set as: {args.depth}')
        logger.info(f'chi_min_frac set as: {args.chi_min_frac}')
        logger.info(f'chi_group_novel set as: {args.chi_group_novel}')
        logger.info(f'alt_stretch_filter set as: {args.alt_stretch_filter}')
        logger.info(f'repeat_filter_kmer set as: {args.repeat_filter_kmer}')
        logger.info(f'alt_cluster_filter set as: {args.alt_cluster_filter}')
        logger.info(f'var_cluster_window set as: {args.var_cluster_window}')
        logger.info(f'var_cluster_n set as: {args.var_cluster_n}')
        logger.info(f'cover_existing set as: {args.cover_existing}')
        logger.info(f'Predefined variant calls: {args.snv_confidence_path}')
        logger.info(f'RNA editing DB: {args.rna_editing_db}')
        logger.info(f'Predefined cell type file: {args.cell_type_df_path}')
        logger.info(f'gene_subset_path set as: {args.gene_subset_path}')
        # high_artifact_mode (Knob B + Knob C) settings — opt-in for lr-snRNA-seq nascent leak
        logger.info(f'high_artifact_mode: {args.high_artifact_mode}')
        if args.high_artifact_mode:
            logger.info(f'  novel_exon_pct_max (Knob B cutoff): {args.novel_exon_pct_max}')
            logger.info(f'  read_intronic_pct_max (Knob C cutoff): {args.read_intronic_pct_max}')
            logger.info(f'  gsi_base_pkl_path (auto-resolved if None): {args.gsi_base_pkl_path}')
        logger.info(f'read_sj_min (Knob D truncation filter): {args.read_sj_min}')
        snv_confidence = None if args.snv_confidence_path is None else pd.read_csv(args.snv_confidence_path, sep='\t')
        ht = u.Haplotyping(scotch_target=args.scotch_target, bam_path=args.bam_path,
                           target=args.output_folder, ref_pickle_path=args.ref_pickle_path,
                           max_iter=args.max_iter, tol=args.tol, verbose=args.verbose, seed=args.seed,
                           mtx=args.mtx, csv=args.csv, n_jobs=args.n_jobs, job_index=args.job_index,
                           n_alt=args.n_alt_count, depth=args.depth,
                           chi_min_frac=args.chi_min_frac, chi_group_novel=args.chi_group_novel,
                           heterozygous_filter=args.heterozygous_filter,alt_stretch_filter = args.alt_stretch_filter,
                           het_fallback=args.het_fallback,
                           repeat_filter_kmer=args.repeat_filter_kmer,
                           alt_cluster_filter = args.alt_cluster_filter, ref_fasta_path=args.ref_fasta_path,
                           var_cluster_window = args.var_cluster_window, var_cluster_n = args.var_cluster_n,
                           sample_name_parse=args.sample_name_parse, prefix=args.prefix,
                           em_snv_filter=args.em_snv_filter, snv_confidence=snv_confidence,
                           snv_classifier=args.snv_classifier,
                           clf_hard_threshold=args.clf_hard_threshold,
                           clf_init=args.clf_init,
                           gap_tau=args.gap_tau,
                           clf_pruning_threshold=args.clf_pruning_threshold,
                           clf_pruning_frac=args.clf_pruning_frac,
                           rna_editing_db=args.rna_editing_db,
                           cell_type_df_path=args.cell_type_df_path, cover_existing = args.cover_existing,
                           gene_subset=load_gene_subset(args.gene_subset_path),
                           n_workers = -1, logger = logger,
                           high_artifact_mode=args.high_artifact_mode,
                           novel_exon_pct_max=args.novel_exon_pct_max,
                           read_intronic_pct_max=args.read_intronic_pct_max,
                           read_sj_min=args.read_sj_min,
                           gsi_base_pkl_path=args.gsi_base_pkl_path)
        ht.generate_count_hap_genes()
        write_done_marker(args.output_folder, 'step3', args.job_index)
        logger.info(f'Finished haplotyping for job {args.job_index}')

    #step 4
    def haplotype_summary(summary = True, count = True):
        logger, _ = setup_logger(args.output_folder, 'step4_haplotype_summary')
        logger.info('Start running step4: haplotype summary...')
        logger.info(f'Output directory: {args.output_folder}')
        logger.info(f'job_array_by_sample: {args.job_array_by_sample}')
        logger.info(f'job_index: {args.job_index}')
        ht = u.Haplotyping(scotch_target=args.scotch_target, target=args.output_folder,
                           ref_pickle_path=args.ref_pickle_path,
                           max_iter=args.max_iter, tol=args.tol, verbose=args.verbose, seed=args.seed,
                           mtx=args.mtx, csv=args.csv, n_jobs=args.n_jobs, job_index=args.job_index,
                           n_alt=args.n_alt_count, depth=args.depth,
                           heterozygous_filter=args.heterozygous_filter,
                           het_fallback=args.het_fallback,
                           repeat_filter_kmer=args.repeat_filter_kmer,
                           sample_name_parse=args.sample_name_parse, prefix=args.prefix,
                           em_snv_filter=args.em_snv_filter, snv_confidence=None,
                           cell_type_df_path=args.cell_type_df_path, cover_existing = args.cover_existing,
                           n_workers = -1, logger = logger,
                           job_array_by_sample=args.job_array_by_sample)
        if summary:
            ht.get_summary_statistics()
            logger.info('Finished summarizing haplotype-aware analysis')
        if count:
            ht.generate_count_matrix()
            logger.info('Finished summarizing count matrix')
        if args.job_array_by_sample:
            write_done_marker(args.output_folder, f'step4_sample{args.job_index}')
        else:
            write_done_marker(args.output_folder, 'step4')


    # step5
    def downstream():
        logger, _ = setup_logger(args.output_folder, 'step5_downstream')
        logger.info('Start running step5: downstream analyses...')
        logger.info(f'Output directory: {args.output_folder}')
        logger.info(f'SCOTCH directory: {args.scotch_target}')
        logger.info(f'event_min_reads: {args.event_min_reads}')
        logger.info(f'snv_event_distance: {args.snv_event_distance}')
        logger.info(f'astu_sig_only: {args.astu_sig_only}')
        logger.info(f'astu_sig_from_bulk: {args.astu_sig_from_bulk}')
        logger.info(f'astu_sig_threshold: {args.astu_sig_threshold}')
        logger.info(f'event_mode: {args.event_mode}')
        logger.info(f'fdr_events_value: {args.fdr_events_value}')
        logger.info(f'n_jobs: {args.n_jobs}')
        logger.info(f'job_index: {args.job_index}')
        logger.info(f'job_array_by_sample: {args.job_array_by_sample}')
        logger.info(f'gene_subset_path set as: {args.gene_subset_path}')
        ds = d.Downstream(
            output_folder=args.output_folder,
            scotch_target=args.scotch_target,
            bam_path=args.bam_path,
            ref_pickle_path=args.ref_pickle_path,
            sample_name_parse=args.sample_name_parse,
            prefix=args.prefix,
            sample_names=args.sample_names,
            cell_type_df_path=args.cell_type_df_path,
            n_workers=args.n_workers,
            astu_sig_only=args.astu_sig_only,
            astu_sig_from_bulk=args.astu_sig_from_bulk,
            astu_sig_threshold=args.astu_sig_threshold,
            n_jobs=args.n_jobs,
            job_index=args.job_index,
            job_array_by_sample=args.job_array_by_sample,
            gene_subset=load_gene_subset(args.gene_subset_path),
            logger=logger,
        )
        ds.run_all(
            event_min_reads=args.event_min_reads,
            snv_event_distance=args.snv_event_distance,
            event_mode=args.event_mode,
            fdr_events_value=args.fdr_events_value,
        )
        write_done_marker(args.output_folder, 'step5')
        logger.info('Finished downstream analysis')

    def check():
        logger, _ = setup_logger(args.output_folder, 'check')
        logger.info('Checking job completion...')

        pfx = args.prefix or ''
        out = args.output_folder

        # ── paths ──────────────────────────────────────────────────────
        variants_dir = os.path.join(out, 'variant_align1', 'variants_by_gene')
        em_input_dir = os.path.join(out, 'em_input')
        summary_sep_dir = os.path.join(
            out,
            f'summary_statistics_{pfx}' if pfx else 'summary_statistics',
            'all_genes_separate')

        # For multi-sample em_input the files live in em_input/{sample_name}/
        if args.scotch_target and len(args.scotch_target) > 1:
            sample_names = (args.sample_names
                            if args.sample_names
                            else [os.path.basename(s) for s in args.scotch_target])
            em_dirs = [os.path.join(em_input_dir, sn) for sn in sample_names]
        else:
            em_dirs = [em_input_dir]

        def gene_ids_by_suffix(directory, suffix):
            """Return set of geneIDs from files ending with suffix in directory."""
            if not os.path.isdir(directory):
                return set()
            return {f.split('_')[0] for f in os.listdir(directory) if f.endswith(suffix)}

        # ── Step 1 ─────────────────────────────────────────────────────
        s1_snvs = gene_ids_by_suffix(variants_dir, '_snvs.csv')
        s1_pkl  = gene_ids_by_suffix(variants_dir, '_site_reads.pkl')
        # Genes where csv was written but job died before pkl = partial failure
        s1_partial = s1_snvs - s1_pkl
        s1_done = s1_snvs & s1_pkl   # both files present = fully complete

        logger.info(f'Step 1 | complete: {len(s1_done)}  '
                    f'partial (csv only): {len(s1_partial)}')
        if s1_partial:
            logger.info(f'  Partial step1 genes will cause step2 errors — '
                        f'rerun step1 for them.')

        # ── Step 1.5 (read_blocks for obs_* / Knob D) ─────────────────
        # Canonical _read_blocks.pkl is the merge output consumed by step5
        # obs_* and step3 Knob D. Intermediate _read_blocks_<N>.pkl files
        # exist between step1_5 and step1_5_merge.
        rb_canonical = gene_ids_by_suffix(variants_dir, '_read_blocks.pkl')
        rb_intermediate = {
            re.match(r'^(.+)_read_blocks_\d+\.pkl$', f).group(1)
            for f in os.listdir(variants_dir)
            if re.match(r'^(.+)_read_blocks_\d+\.pkl$', f)
        } if os.path.isdir(variants_dir) else set()
        s1_5_merge_missing = s1_done - rb_canonical
        s1_5_intermediate_pending = rb_intermediate - rb_canonical

        logger.info(f'Step 1.5 | canonical _read_blocks.pkl: {len(rb_canonical)}  '
                    f'missing (step5 obs_* / step3 Knob D unavailable): {len(s1_5_merge_missing)}  '
                    f'intermediate _read_blocks_<N>.pkl still around: {len(s1_5_intermediate_pending)}')
        if s1_5_intermediate_pending:
            logger.info(f'  step1_5_merge has not finished — run --task step1_5_merge.')
        elif s1_5_merge_missing and not rb_intermediate:
            logger.info(f'  step1_5 + step1_5_merge never ran — obs_* will be no_bam, Knob D inactive.')

        # ── Step 2 ─────────────────────────────────────────────────────
        s2_pile, s2_npz, s2_rsnv, s2_rpi = set(), set(), set(), set()
        for em_dir in em_dirs:
            s2_pile |= gene_ids_by_suffix(em_dir, '_pileup.csv')
            s2_npz  |= gene_ids_by_suffix(em_dir, '_read_matrices.npz')
            s2_rsnv |= gene_ids_by_suffix(em_dir, '_read_snv.csv')
            s2_rpi  |= gene_ids_by_suffix(em_dir, '_read_pi.csv')
        # Complete if pileup + (npz OR both legacy csvs)
        s2_has_em = s2_npz | (s2_rsnv & s2_rpi)
        s2_done = s2_pile & s2_has_em
        # Step 2 partial: some but not all files written
        s2_any  = s2_pile | s2_npz | s2_rsnv | s2_rpi
        s2_partial = s2_any - s2_done
        # Expected: genes that completed step 1
        s2_expected = s1_done
        s2_missing  = s2_expected - s2_done - s2_partial

        logger.info(f'Step 2 | complete: {len(s2_done)} / {len(s2_expected)}  '
                    f'partial: {len(s2_partial)}  '
                    f'not started: {len(s2_missing)}')

        # ── Step 3 ─────────────────────────────────────────────────────
        s3_done = set()
        if os.path.isdir(summary_sep_dir):
            # filename: {geneName}_{geneID}_summary.csv
            # geneID has no underscores (Ensembl IDs), so split('_')[-2] is safe
            s3_done = {f.split('_')[-2]
                       for f in os.listdir(summary_sep_dir)
                       if f.endswith('_summary.csv')}
        s3_expected = s2_done
        s3_missing  = s3_expected - s3_done

        logger.info(f'Step 3 | complete: {len(s3_done)} / {len(s3_expected)}  '
                    f'missing: {len(s3_missing)}')

        # ── Write missing-gene files ────────────────────────────────────
        report = {
            'step1_partial': s1_partial,
            'step2_missing': s2_missing | s2_partial,
            'step3_missing': s3_missing,
        }
        any_missing = False
        for tag, gene_set in report.items():
            if gene_set:
                any_missing = True
                path = os.path.join(out, f'missing_genes_{tag}.txt')
                with open(path, 'w') as fh:
                    for g in sorted(gene_set):
                        fh.write(g + '\n')
                logger.info(f'  {tag}: {len(gene_set)} genes → {path}')
                logger.info(f'    Resubmit: --gene_subset_path {path}')
        if not any_missing:
            logger.info('All steps complete — no missing genes detected.')

    if args.task=='step1':
        variant_calling()
    if args.task=='step1_5':
        collect_read_blocks()
    if args.task=='step1_5_merge':
        merge_read_blocks()
    if args.task=='step2':
        generate_em_input()
    if args.task=='step3':
        haplotyping()
    if args.task=='step4':
        haplotype_summary(args.summary_haplotype, args.summary_count)
    if args.task=='step5':
        downstream()
    if args.task=='check':
        check()


if __name__ == "__main__":
    main()
