import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
import src.utils as u
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='LongAllele Haplotyping')
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
parser.add_argument('--n_alt_count', type=int, default=10)
parser.add_argument('--depth', type=int, default=20)
parser.add_argument('--chi_min_frac', type=float, default=0.1)
parser.add_argument('--chi_group_novel',action='store_true')
parser.add_argument('--ref_pickle_path', type=str)
parser.add_argument('--var_cluster_window', type=int, default=20)
parser.add_argument('--var_cluster_n', type=int, default=3)
parser.add_argument('--heterozygous_filter', type=float, default=0.95) #set negative: no filter, set 0: auto filter top n (conservative), or set a positive value
parser.add_argument('--alt_stretch_filter', type=int, default=20)
parser.add_argument('--repeat_filter_kmer', type=int, default=1)
parser.add_argument('--alt_cluster_filter', type=int, default=20)
parser.add_argument('--ref_fasta_path', type=str)
parser.add_argument('--em_snv_filter',action='store_true') #whether to post-filter
parser.add_argument('--sample_name_parse',type=str)
parser.add_argument('--prefix',type=str)

#predefined snv set
parser.add_argument('--snv_confidence_path', type=str)
parser.add_argument(
    '--rna_editing_db',
    type=str,
    help='Path to compact RNA editing DB (.npz, 0-based positions, keys like AG__chr1 / TC__chr1).'
)
#Cell CellType mappipng df
parser.add_argument('--cell_type_df_path', type=str)

# --- high_artifact_mode (Knob B + Knob C; opt-in for lr-snRNA-seq nascent leak) ---
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

#the order is haplotyping, summary

def main():
    global args
    args = parser.parse_args()
    snv_confidence = None if args.snv_confidence_path is None else pd.read_csv(args.snv_confidence_path, sep = '\t')
    ht = u.Haplotyping(scotch_target=args.scotch_target, target=args.output_folder,
                       max_iter=args.max_iter, tol=args.tol, verbose=args.verbose, seed=args.seed,
                       mtx=args.mtx, csv=args.csv,n_jobs=args.n_jobs, job_index=args.job_index,
                       n_alt=args.n_alt_count, depth=args.depth,
                       chi_min_frac = args.chi_min_frac, chi_group_novel = args.chi_group_novel,
                       heterozygous_filter=args.heterozygous_filter,alt_stretch_filter = args.alt_stretch_filter,
                       repeat_filter_kmer=args.repeat_filter_kmer,
                       alt_cluster_filter = args.alt_cluster_filter,
                       var_cluster_window=args.var_cluster_window, var_cluster_n=args.var_cluster_n,
                       sample_name_parse=args.sample_name_parse,prefix=args.prefix,
                       em_snv_filter = args.em_snv_filter, snv_confidence = snv_confidence,
                       rna_editing_db = args.rna_editing_db,
                       ref_pickle_path = args.ref_pickle_path, cell_type_df_path = args.cell_type_df_path,
                       high_artifact_mode = args.high_artifact_mode,
                       novel_exon_pct_max = args.novel_exon_pct_max,
                       read_intronic_pct_max = args.read_intronic_pct_max,
                       read_sj_min = args.read_sj_min,
                       gsi_base_pkl_path = args.gsi_base_pkl_path)
    if args.task == 'haplotyping':
        ht.generate_count_hap_genes()
    if args.task == 'summary':
        print('summarising results')
        ht.get_summary_statistics()
        print('generating count matrix')
        ht.generate_count_matrix()



if __name__ == "__main__":
    main()
