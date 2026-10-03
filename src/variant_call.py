import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
import src.utils as u
import argparse


parser = argparse.ArgumentParser(description='Variant Call', allow_abbrev=False)
parser.add_argument('--task', type=str)
parser.add_argument('--output_folder', type=str)
parser.add_argument('--scotch_target', type=str, nargs='+')
parser.add_argument('--sample_names', type=str, nargs='+')
parser.add_argument('--bam_path', type=str, nargs='+')
parser.add_argument('--ref_fasta_path', type=str, default=None)
parser.add_argument('--ref_pickle_path', type=str)
parser.add_argument('--sample_name_parse', type=str)
parser.add_argument('--n_jobs', type=int, default=1)
parser.add_argument('--job_index', type=int, default=0)
parser.add_argument('--platform', type=str, default=None,
                    choices=['ont-cdna', 'ont-drna', 'hifi-isoseq', 'hifi-masseq'],
                    help='measured calling preset (explicit flags override; presets '
                         'set hardware params only, never the classifier)')
parser.add_argument('--min_alt_frac', type=float, default=0.0,
                    help='AF side of the candidate gate (0 = pure absolute, legacy)')
parser.add_argument('--min_baseq', type=int, default=5,
                    help='minimum base quality for pileup counting')
parser.add_argument('--max_baseq', type=int, default=None,
                    help='base-quality ceiling before pi conversion')
parser.add_argument('--gene_guard_depth', type=int, default=None,
                    help='minimum whitelisted reads for a gene to be scanned')
parser.add_argument('--light_ambiguity_ratio', type=float, default=2.0,
                    help='(preset surface only; consumed by light_prep)')
parser.add_argument('--min_dist_to_end', type=int, default=3,
                    help='minimum ALT distance to read end')
parser.add_argument('--ignore_gate_mismatch', action='store_true',
                    help='downgrade the candidate-gate agreement check to a warning')
parser.add_argument('--n_alt_count', type=int, default=1)
parser.add_argument('--depth', type=int, default=5)


def main():
    global args
    args = parser.parse_args()
    u.refuse_retired_entry_point('variant_call.py', args.task)
    u.apply_platform_preset(args)

    vc = u.VariantCaller(scotch_target = args.scotch_target, bam_path= args.bam_path,
                         ref_fasta_path = args.ref_fasta_path, ref_pickle_path = args.ref_pickle_path,
                         target = args.output_folder,
                         n_jobs = args.n_jobs, job_index = args.job_index,
                         depth = args.depth, n_alt = args.n_alt_count, min_alt_frac = args.min_alt_frac,
                         min_baseq = args.min_baseq, max_baseq = args.max_baseq,
                         gene_guard_depth = args.gene_guard_depth,
                         min_dist_to_end = args.min_dist_to_end,
                         sample_name_parse = args.sample_name_parse,
                         sample_names = args.sample_names)


    if args.task == 'initial call':
        u.ensure_gate_config(args.output_folder, args.n_alt_count, args.depth,
                             args.min_alt_frac,
                             strict=not args.ignore_gate_mismatch)
    elif args.task == 'generate input':
        u.check_gate_config(args.output_folder, args.n_alt_count, args.depth,
                            args.min_alt_frac,
                            strict=not args.ignore_gate_mismatch)
    if args.task=='initial call':
        vc.process_genes_round1_1()
    if args.task=="generate input":
        vc.process_genes_final()


if __name__ == "__main__":
    main()
