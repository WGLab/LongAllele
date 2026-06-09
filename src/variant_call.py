import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
import src.utils as u
import argparse



parser = argparse.ArgumentParser(description='Variant Call')
parser.add_argument('--task', type=str)
parser.add_argument('--output_folder', type=str) #a single output folder to store joint-called variants
parser.add_argument('--scotch_target', type=str, nargs='+')
parser.add_argument('--sample_names', type=str, nargs='+')
parser.add_argument('--bam_path', type=str, nargs='+')
parser.add_argument('--ref_fasta_path', type=str, default=None)
parser.add_argument('--ref_pickle_path', type=str)
parser.add_argument('--sample_name_parse', type=str)
parser.add_argument('--n_jobs', type=int, default=1)
parser.add_argument('--job_index', type=int, default=0)
parser.add_argument('--n_alt_count', type=int, default=1)
parser.add_argument('--depth', type=int, default=5)


def main():
    global args
    args = parser.parse_args()
    #realignment = True if args.realignment==1 else False
    vc = u.VariantCaller(scotch_target = args.scotch_target, bam_path= args.bam_path,
                         ref_fasta_path = args.ref_fasta_path, ref_pickle_path = args.ref_pickle_path,
                         target = args.output_folder,
                         n_jobs = args.n_jobs, job_index = args.job_index,
                         depth = args.depth, n_alt = args.n_alt_count,
                         sample_name_parse = args.sample_name_parse,
                         sample_names = args.sample_names)
    if args.task=='initial call':
        vc.process_genes_round1_1() # job array
    if args.task=="generate input":
        vc.process_genes_final()  # job array



if __name__ == "__main__":
    main()