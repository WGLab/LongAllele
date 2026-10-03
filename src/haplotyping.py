import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
import src.utils as u
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='LongAllele Haplotyping', allow_abbrev=False)
parser.add_argument('--task', type=str)
parser.add_argument('--output_folder', type=str, default=None)
parser.add_argument('--scotch_target', type=str, nargs='+')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_jobs', type=int, default=1)
parser.add_argument('--job_index', type=int, default=0)
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--mtx',action='store_true')
parser.add_argument('--csv',action='store_true')
parser.add_argument('--platform', type=str, default=None,
                    choices=['ont-cdna', 'ont-drna', 'hifi-isoseq', 'hifi-masseq'],
                    help='measured calling preset: hardware gates plus the '
                         'library-matched SNV classifier shipped in src/models/ '
                         '(hifi-masseq is pinned to no classifier). Explicit flags '
                         'override; pass --snv_classifier "" to disable it.')

parser.add_argument('--snv_classifier', type=str, default=None,
                    help='path to a joblib SNV classifier; empty string disables '
                         'the one a --platform preset would supply')
parser.add_argument('--clf_hard_threshold', type=float, default=0.05,
                    help='drop candidate SNVs whose classifier probability is '
                         'below this value')
parser.add_argument('--min_alt_frac', type=float, default=0.0,
                    help='AF side of the candidate gate (0 = pure absolute, legacy)')
parser.add_argument('--ignore_gate_mismatch', action='store_true',
                    help='downgrade the step1-vs-step3 gate mismatch check to a warning')
parser.add_argument('--n_alt_count', type=int, default=10)
parser.add_argument('--depth', type=int, default=20)
parser.add_argument('--chi_min_frac', type=float, default=0.1)
parser.add_argument('--chi_group_novel',action='store_true')
parser.add_argument('--ref_pickle_path', type=str)
parser.add_argument('--var_cluster_window', type=int, default=20)
parser.add_argument('--var_cluster_n', type=int, default=3)
parser.add_argument('--heterozygous_filter', type=float, default=0.95)
parser.add_argument('--alt_stretch_filter', type=int, default=20)
parser.add_argument('--alt_stretch_len', type=int, default=5,
                    help='homopolymer run length at which a site is flagged; '
                         'the run must COVER the site. NOT the same knob as '
                         '--alt_stretch_filter, which is the alt-count escape '
                         'clause. Default 5 = historical behaviour; <=0 turns the gate OFF '
                         '(do NOT expect 0 to mean "no filtering" by accident -- '
                         'it is handled explicitly, see utils)')
parser.add_argument('--init_link_min_agreement', type=float, default=0.0,
                    help='clf-free Tier 1: before the spectral init, drop '
                         'marker-pair edges whose co-read agreement '
                         '|W|/n_shared is below this. 0.0 = OFF (default)')
parser.add_argument('--init_link_min_shared', type=int, default=3,
                    help='minimum shared reads for a marker-pair edge to be '
                         'considered at all by --init_link_min_agreement')
parser.add_argument('--h_m_init_from', type=str, default='clf',
                    choices=['clf', 'linkage', 'linkage_lr', 'none'],
                    help="clf-free Tier 2: source of h_m_init. 'clf' "
                         '(default) = current behaviour, needs --clf_init and '
                         "a model; 'linkage' = per-marker co-read agreement, "
                         'no model required, and it re-enables the iterative '
                         'pruning that was previously reachable only through '
                         "a classifier; 'none' = no h_m_init")
parser.add_argument('--repeat_filter_kmer', type=int, default=1)
parser.add_argument('--alt_cluster_filter', type=int, default=20)
parser.add_argument('--ref_fasta_path', type=str)
parser.add_argument('--em_snv_filter',action='store_true')
parser.add_argument('--sample_name_parse',type=str)
parser.add_argument('--prefix',type=str)


parser.add_argument('--snv_confidence_path', type=str)
parser.add_argument(
    '--rna_editing_db',
    type=str,
    help='Path to compact RNA editing DB (.npz, 0-based positions, keys like AG__chr1 / TC__chr1).'
)

parser.add_argument('--cell_type_df_path', type=str)


parser.add_argument('--high_artifact_mode', action='store_true',
                    help='Enable Knob B (gene-level SCOTCH-novel SNV mask) + Knob C (read-level '
                         'nascent / pre-mRNA filter) at step3. Default OFF preserves the standard '
                         'pipeline byte-for-byte.')
parser.add_argument('--novel_exon_pct_max', type=float, default=0.25,
                    help='Knob B cutoff. Only active with --high_artifact_mode.')
parser.add_argument('--read_intronic_pct_max', type=float, default=0.60,
                    help='Knob C cutoff. Only active with --high_artifact_mode.')
parser.add_argument('--read_sj_min', type=int, default=0,
                    help='Knob D: drop reads with fewer than N internal splice junctions '
                         'at step3 EM phasing. Default 0 (no filter). Recommended >=1 for '
                         'hap-resolved analysis. Requires read_blocks.pkl from step1.5 '
                         '(--task step1_5 + --task step1_5_merge in longallele.py).')
parser.add_argument('--gsi_base_pkl_path', type=str, default=None,
                    help='Optional explicit path to SCOTCH base gene structure pickle for '
                         '--high_artifact_mode (auto-resolved from scotch_target[0]/reference/ '
                         'if omitted).')

parser.add_argument('--editing_exempt_affinity', type=float, default=2.0,
                    help='RNA-editing exemption (value IS the switch; >1 = off, '
                         'the default). A REDIportal-listed site is KEPT as an '
                         'ordinary SNV when its co-read linkage strength |W|/n_shared with a '
                         'NON-listed candidate exceeds this. Measured operating '
                         'point: 0.95 (54 true SNVs rescued vs 2 edits leaked on '
                         '913 real genes; TP +52, precision +0.0005, phasing '
                         '+0.0006). Everything unproven stays deleted.')
parser.add_argument('--editing_exempt_min_reads', type=int, default=10,
                    help='minimum shared reads for a co-read affinity pair to '
                         'count toward the exemption (guards small-sample noise)')
parser.add_argument('--gene_guard_depth', type=int, default=None,
                    help='(preset surface only; consumed by step1)')
parser.add_argument('--light_ambiguity_ratio', type=float, default=2.0,
                    help='(preset surface only; consumed by light_prep)')
parser.add_argument('--max_baseq', type=int, default=None,
                    help='do not trust base qualities above this; step 2 caps pi here and step 3 floors the het-filter error rate to match')
parser.add_argument('--em_init', type=str, default='signed',
                    choices=['signed', 'concurrence'],
                    help="EM haplotype init: 'signed' (default) or 'concurrence' (legacy)")
parser.add_argument('--no_phase_flip', dest='phase_flip', action='store_false',
                    help='disable the post-EM guarded suffix-flip (on by default)')


def main():
    global args
    args = parser.parse_args()
    u.refuse_retired_entry_point('haplotyping.py', args.task)
    u.apply_platform_preset(args)
    snv_confidence = None if args.snv_confidence_path is None else pd.read_csv(args.snv_confidence_path, sep = '\t')
    ht = u.Haplotyping(scotch_target=args.scotch_target, target=args.output_folder,
                       max_iter=args.max_iter, tol=args.tol, verbose=args.verbose, seed=args.seed,
                       mtx=args.mtx, csv=args.csv,n_jobs=args.n_jobs, job_index=args.job_index,
                       n_alt=args.n_alt_count, min_alt_frac=args.min_alt_frac, depth=args.depth,
                       chi_min_frac = args.chi_min_frac, chi_group_novel = args.chi_group_novel,
                       heterozygous_filter=args.heterozygous_filter,alt_stretch_filter = args.alt_stretch_filter,alt_stretch_len=args.alt_stretch_len,init_link_min_agreement=args.init_link_min_agreement,init_link_min_shared=args.init_link_min_shared,h_m_init_from=args.h_m_init_from,
                       editing_exempt_affinity=args.editing_exempt_affinity,
                           editing_exempt_min_reads=args.editing_exempt_min_reads,
                           em_init_method=args.em_init, phase_flip=args.phase_flip,
                       repeat_filter_kmer=args.repeat_filter_kmer,
                       alt_cluster_filter = args.alt_cluster_filter,
                       var_cluster_window=args.var_cluster_window, var_cluster_n=args.var_cluster_n,
                       sample_name_parse=args.sample_name_parse,prefix=args.prefix,
                       em_snv_filter = args.em_snv_filter, snv_confidence = snv_confidence,
                       max_baseq=args.max_baseq,
                       snv_classifier=args.snv_classifier,
                       clf_hard_threshold=args.clf_hard_threshold,
                       rna_editing_db = args.rna_editing_db,
                       ref_pickle_path = args.ref_pickle_path, cell_type_df_path = args.cell_type_df_path,
                       high_artifact_mode = args.high_artifact_mode,
                       novel_exon_pct_max = args.novel_exon_pct_max,
                       read_intronic_pct_max = args.read_intronic_pct_max,
                       read_sj_min = args.read_sj_min,
                       gsi_base_pkl_path = args.gsi_base_pkl_path)


    if args.task == 'haplotyping':
        u.check_gate_config(args.output_folder, args.n_alt_count, args.depth,
                            args.min_alt_frac,
                            strict=not args.ignore_gate_mismatch)
    if args.task == 'haplotyping':
        ht.generate_count_hap_genes()
    if args.task == 'summary':
        print('summarising results')
        ht.get_summary_statistics()
        print('generating count matrix')
        ht.generate_count_matrix()


if __name__ == "__main__":
    main()
