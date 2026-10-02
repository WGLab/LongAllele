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


parser = argparse.ArgumentParser(description='LongAllele', allow_abbrev=False)
parser.add_argument('--task', type=str)
parser.add_argument('--output_folder', type=str)
parser.add_argument('--scotch_target', type=str, nargs='+')
parser.add_argument('--sample_names', type=str, nargs='+')
parser.add_argument('--sample_name_parse', type=str)
parser.add_argument('--platform', type=str, default=None,
                    choices=['ont-cdna', 'ont-drna', 'hifi-isoseq', 'hifi-masseq', 'other'],
                    help='Apply the measured calling preset for this platform: '
                         'hardware gates (n_alt_count / min_alt_frac / min_baseq / '
                         'max_baseq / min_dist_to_end) plus the matching SNV '
                         'classifier, which ships in src/models/. The classifier is '
                         'keyed by LIBRARY, not chemistry: Iso-Seq and MAS-Seq share '
                         'HiFi hardware, yet one shared model measures 9.5:1 benefit '
                         'on Iso-Seq and 0.1:1 (a net loss) on MAS-Seq, so hifi-masseq '
                         'is pinned to NO classifier. Explicit flags always override '
                         'the preset (pass --snv_classifier to substitute your own, or '
                         'an empty string to disable it); omit --platform and nothing '
                         'changes.')
parser.add_argument('--ignore_gate_mismatch', action='store_true',
                    help='downgrade the step1-vs-step3 candidate-gate mismatch '
                         'check from an error to a warning (use only when you '
                         'deliberately re-gate an existing step1 output)')
parser.add_argument('--min_alt_frac', type=float, default=0.0,
                    help='AF side of the candidate gate: additionally require '
                         'alt_count >= ceil(min_alt_frac * depth). Default 0 = the '
                         'pure absolute gate (existing behavior byte-for-byte). The '
                         'absolute side keeps its strict > semantics, i.e. '
                         '--n_alt_count 2 means alt >= 3.')
parser.add_argument('--n_alt_count', type=int, default=10)
parser.add_argument('--depth', type=int, default=20,
                    help='minimum SNV-site depth. NOTE this is measured after '
                         'mapq/baseq filtering AND gene assignment, so it is a '
                         'smaller number than a raw pileup depth; the presets '
                         'use longcallR-aligned values (hifi 6 / ont 10)')
parser.add_argument('--gene_guard_depth', type=int, default=None,
                    help='minimum whitelisted reads for a gene to be scanned '
                         'at all (default: follow --depth). Decoupled so the '
                         'site gate can be lowered without changing which '
                         'genes are scanned')
parser.add_argument('--n_jobs', type=int, default=1)
parser.add_argument('--job_index', type=int, default=0)


parser.add_argument('--cover_existing',action='store_true')
parser.add_argument('--cover_existing_false', action='store_false',dest='cover_existing')


parser.add_argument('--ref_fasta_path', type=str, default=None)


parser.add_argument('--bam_path', type=str, nargs='+')
parser.add_argument('--ref_pickle_path', type=str, help='(optional), assign a reference pickle file for variant callilng')


parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--mtx',action='store_true')
parser.add_argument('--csv',action='store_true')

class _ParserDefault(float):
    __slots__ = ()


HET_STEP1_DEFAULT = _ParserDefault(0.8)
HET_STEP3_DEFAULT = _ParserDefault(0.99)


def _was_given(v):
    return v is not None and not isinstance(v, _ParserDefault)


parser.add_argument('--het_prob_step1', type=float, default=HET_STEP1_DEFAULT,
                    help="step1's het_prob cutoff, when candidates are called "
                         "(default 0.8, the decided value)")
parser.add_argument('--het_prob_step3', type=float, default=HET_STEP3_DEFAULT,
                    help="step3's het_prob cutoff, before phasing (default 0.99, "
                         "the decided value). Negative = no filter; 0 = "
                         "auto-filter the top n (conservative)")


parser.add_argument('--heterozygous_filter', type=float, default=None,
                    help='DEPRECATED alias for --het_prob_step3')
parser.add_argument('--em_snv_filter', action='store_true', default=True)
parser.add_argument('--no_em_snv_filter', dest='em_snv_filter', action='store_false')
parser.add_argument('--snv_classifier', type=str, default=None,
                    help='Path to serialized SNV classifier (.joblib) for hard filtering before EM')
parser.add_argument('--clf_hard_threshold', type=float, default=0.05,
                    help='Remove SNVs with classifier score below this (hard filter before EM)')
parser.add_argument('--setting', type=str, default=None,
                    choices=['clf', 'clf-free'],
                    help='which half of the platform\'s decided configuration to run. '
                         'clf = the headline setting (classifier on, EM started from '
                         'its scores); clf-free = the second REPORTED setting '
                         '(classifier off, EM started from marker linkage). Default: '
                         'clf when --platform names a platform that has a classifier, '
                         'clf-free when it does not. Needs --platform.')
parser.add_argument('--clf_init', action='store_true',
                    help='DEPRECATED and ignored -- superseded by --h_m_init_from clf, '
                         'which is the default whenever a classifier is available. It '
                         'used to be an extra switch the preset could not set, so '
                         '--platform alone did not reproduce the decided configuration. '
                         'Passing it now warns and changes nothing.')
parser.add_argument('--em_max_reads', type=int, default=20000,
                    help='Ultra-deep gene cap for step3 phasing: learn SNV '
                         'selection + haplotype markers on this many randomly '
                         'subsampled reads, then assign ALL reads by the '
                         'learned markers (alpha re-estimated on all reads). '
                         '0 disables. Default 20000 = ON ('
                         ''
                         '. just keep pbmc that is, for the rest, open '
                         'them, asked about AD specifically, ). '
                         'Why the default moved back ON:it was pinned '
                         'OFF because the '
                         'uncapped fit had become fast enough after the init fix '
                         '(45 s / 24 s / 9 s on the deepest cdna genes; '
                         ', job 21450860). The real-data reruns of '
                         '/12 (GTEx, human brain, AD) all passed '
                         '--em_max_reads 20000 explicitly, so the preset said OFF '
                         'while every production run was capped — the value and '
                         'the record disagreed. resolved it toward what ran. '
                         'PBMC is the one cohort that ran uncapped and stays that '
                         'way; a rerun of PBMC must '
                         'pass --em_max_reads 0 explicitly. NOTE: the cap CHANGES '
                         'results on ultra-deep genes (1,688 sites / 69 genes on '
                         'masseq), so a comparison across the boundary must state '
                         'which side it ran on.')
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
parser.add_argument('--alt_stretch_len', type=int, default=5,
                    help='homopolymer run length at which a site is flagged; '
                         'the run must COVER the site. NOT the same knob as '
                         '--alt_stretch_filter, which is the alt-count escape '
                         'clause. Default 5 = historical behaviour; <=0 turns the gate OFF '
                         '(do NOT expect 0 to mean "no filtering" by accident -- '
                         'it is handled explicitly, see utils)')
parser.add_argument('--phase_block_min_shared', type=int, default=1,
                    help='reads informative at BOTH markers before they are called '
                         'phased together (phase_block / PS ids). Default 1 = the '
                         'historical behaviour, so this flag changes nothing until '
                         'it is raised. Raising it errs toward SPLITTING, which '
                         'costs interval width; the low setting errs toward MERGING, '
                         'which costs correctness -- a merged pair gets an anchored '
                         'orientation nothing supports and the interval does not '
                         'widen for it.')
parser.add_argument('--phase_block_min_agreement', type=float, default=0.0,
                    help='how one-sided the shared reads must be about the two '
                         'markers (max(concordant, discordant) / shared) before the '
                         'link is accepted. 0.0 = off, the historical behaviour.')
parser.add_argument('--init_link_min_agreement', type=float, default=0.0,
                    help='clf-free Tier 1: before the spectral init, drop '
                         'marker-pair edges whose co-read agreement '
                         '|W|/n_shared is below this. 0.0 = OFF (default)')
parser.add_argument('--init_link_min_shared', type=int, default=3,
                    help='minimum shared reads for a marker-pair edge to be '
                         'considered at all by --init_link_min_agreement')
parser.add_argument('--h_m_init_from', type=str, default='clf',
                    choices=['clf', 'linkage', 'linkage_lr', 'none'],
                    help="where the EM's starting belief about each SNV comes from. "
                         "'clf' (default) = classifier scores, used whenever a model "
                         'is loaded and falling back to the three-genotype posterior '
                         "when none is; 'linkage' = per-marker co-read agreement, no "
                         'model required, and the decided clf-free setting on all four '
                         "platforms; 'linkage_lr' = the same evidence as a binomial "
                         "likelihood ratio; 'none' = no h_m_init.")
parser.add_argument('--het_prefilter_threshold', type=float, default=None,
                    help='DEPRECATED alias for --het_prob_step1')
parser.add_argument('--het_beta', type=str, default=None,
                    help="heterozygous likelihood: unset (default) assumes the ALT "
                         "fraction is exactly 0.5, which is allelic BALANCE, not "
                         "heterozygosity -- at alpha=0.10 a true het site shows 0.10 "
                         "or 0.90 and is scored as homozygous. Give 'a,b' (e.g. "
                         "'2,2') to integrate the fraction out under Beta(a,b) "
                         "instead. Measured at alpha=0.10: alpha bias +0.087 -> "
                         "+0.020, recall 0.011 -> 0.154, precision 0.64 -> 0.85; "
                         "nothing moves at alpha=0.3/0.5. DEFAULT IS THE OLD "
                         "BEHAVIOUR until sim and benchmark real are both re-run.")
parser.add_argument('--shrink_denominator', type=str, default='gene_reads',
                    choices=['gene_reads', 'deepest_site'],
                    help="what the evidence tempering is measured against. "
                         "gene_reads (default) = the gene's total reads, which "
                         "falls as the gene gets LONGER -- a perfectly ordinary "
                         "site in a 20 kb gene is tempered to 5%%. deepest_site = "
                         "the gene's own deepest site, which asks how cold a corner "
                         "of THIS gene the site is in. Measured: recall 0.319 -> "
                         "0.365 at alpha=0.5, precision 0.981 -> 0.959.")
parser.add_argument('--coverage_factor', type=float, default=1.0,
                    help='exponent on the coverage fraction. 1.0 (default) is '
                         'linear; 0 disables tempering entirely. 0.5 was measured '
                         'and NOT adopted -- it buys 3.4 points of recall for 6.8 '
                         'of precision.')
parser.add_argument('--repeat_filter_kmer', type=int, default=1)
parser.add_argument('--min_mapq', type=int, default=20)
parser.add_argument('--min_baseq', type=int, default=5,
                    help='minimum base quality to count a base (0 = count '
                         'every base and let the model weight it, which is '
                         'what the presets do)')
parser.add_argument('--max_baseq', type=int, default=None,
                    help='base-quality CEILING before converting to an error '
                         'probability (presets: 30). baseq knows nothing about '
                         'alignment or reference error, so an uncapped Q93 '
                         'becomes pi=5e-10 and lets one mis-aligned read '
                         'dominate the EM. Default None = uncapped (legacy)')
parser.add_argument('--min_dist_to_end', type=int, default=3)
parser.add_argument('--chi_min_frac', type=float, default=0.1)
parser.add_argument('--chi_group_novel',action='store_true')
parser.add_argument('--prefix',type=str)

parser.add_argument('--snv_confidence_path', type=str, nargs='+',
                    help='known heterozygous sites (TSV: chrom, pos, ref, alt). One '
                         'file PER BAM, in the same order as --bam_path, space '
                         'separated -- two BAMs cannot share one genotype list. '
                         'Which steps use it is --genotype_stage.')


parser.add_argument('--genotype_stage', type=str, default=None,
                    choices=['step1', 'step3', 'both'],
                    help='which steps use --snv_confidence_path. both (DEFAULT) = '
                         'given sites all the way, which is the actual '
                         'genotype-guided method. step3 = the pre-'
                         'default, kept as a CONTROL: step1 calls variants '
                         'normally and step3 intersects its survivors with the '
                         'given list. step1 = the other control: step1 restricts '
                         'to the given sites with NO filter and step3 still runs '
                         'its own filters. The two controls isolate which step '
                         'loses what; neither is the method.')


GENOTYPE_STAGE_DEFAULT = 'both'


def resolve_genotype_stage(args):
    if getattr(args, 'genotype_stage', None) is None:
        return GENOTYPE_STAGE_DEFAULT, 'default'
    return args.genotype_stage, 'user override'
parser.add_argument(
    '--rna_editing_db',
    type=str,
    default=os.path.join(os.path.dirname(__file__), 'rna_editing_hg38.npz'),
    help="Path to compact RNA editing DB (.npz). Default: bundled hg38 database. Pass 'none' (or empty) to disable RNA editing filtering."
)

parser.add_argument('--gene_subset_path', type=str)

parser.add_argument('--cell_type_df_path', type=str, nargs='+')


parser.add_argument('--summary_haplotype', action='store_true',
                    help='accepted for compatibility; summarising is the default')
parser.add_argument('--summary_count', action='store_true',
                    help='accepted for compatibility; the count matrix is the default')
parser.add_argument('--no_summary_haplotype', action='store_true',
                    help='skip the haplotype summary in step4')
parser.add_argument('--no_summary_count', action='store_true',
                    help='skip the count matrix in step4')

parser.add_argument('--event_min_reads', type=int, default=10)


parser.add_argument('--actv', action=argparse.BooleanOptionalAction, default=None,
                    help='step5: per-gene cross-cell-type ACTV table '
                         '(per-CT plain deltas, cross-CT range, permutation '
                         'p by shuffling cell-type labels — no EM rerun; '
                         'gate flags, never filters). '
                         'authorizing order. Default = AUTO: on '
                         'whenever the run has cell types to compare -- a '
                         '--cell_type_df_path, or --same_individual (tissue '
                         'names become cell types) -- and off otherwise ('
                         '; '
                         '). --actv / --no-actv override.')
parser.add_argument('--actv_permutations', type=int, default=300,
                    help='number of valid label permutations per gene for ACTV (B)')
parser.add_argument('--actv_unit', choices=['cell', 'read'], default=None,
                    help='ACTV permutation unit: cell (single-cell) or read '
                         '(bulk/pooled trees, e.g. GTEx two-tissue — shuffles '
                         'reads\' tissue labels; ). Default '
                         '= read under --same_individual, cell otherwise.')
parser.add_argument('--actv_min_cells', type=int, default=10,
                    help='minimum expressing cells per qualifying ACTV cell type; '
                         'ignored in read mode')
parser.add_argument('--actv_min_phasable_reads', type=int, default=20,
                    help='minimum phasable reads per qualifying ACTV context')
parser.add_argument('--actv_max_attempts', type=int, default=None,
                    help='maximum ACTV permutation attempts (default: 10 times B)')


parser.add_argument('--actv_min_phasable_frac', type=float, default=0.6,
                    help='ACTV credibility flag: every participating context must have '
                         'phasable reads / total reads >= this (the weakest context '
                         'decides); flag column pass_phasable_frac, rows are not dropped')
parser.add_argument('--actv_min_actv', type=float, default=0.3,
                    help='ACTV effect-size flag: actv >= this; flag column pass_min_actv, '
                         'rows are not dropped')
parser.add_argument('--snv_event_distance', type=int, default=50)
parser.add_argument('--n_workers', type=int, default=1,
                    help='Number of parallel workers for step5 downstream analysis')
parser.add_argument('--astu_sig_only', action='store_true')
parser.add_argument('--astu_sig_from_bulk', action='store_true',
                    help='When filtering Task 4 by ASTU significance, derive the significant gene set from Bulk rows and reuse it for all cell types.')
parser.add_argument('--astu_sig_threshold', type=float, default=0.05)
parser.add_argument('--ase_call_margin', type=float, default=0.095,
                    help='step5: ASE_call door 2. ASE_call is 1 '
                         'when gene p_adj <= 0.05 AND (door 1: the whole alpha '
                         'interval lies on one side of 0.5, i.e. alpha_hat_high < 0.5 '
                         'or alpha_hat_low > 0.5; OR door 2: |(low+high)/2 - 0.5| >= '
                         'this margin, i.e. half the unphasable reads allocated '
                         'adversarially still leaves 40.5/59.5). -1 when p_adj > '
                         '0.05; 0 = significant but neither door. Default 0.095 = '
                         'loosest margin letting <=5%% of sim trap genes through '
                         '.')
parser.add_argument('--conf_nonphasable_astu', type=float, default=1.0,
                    help='step5: ASTU accountability gate '
                         '— the read-isoform constraint '
                         'caps interception below 95%%, so theta takes its supremum: '
                         'conf_astu >= 1 <=> still significant under the TRUE '
                         'worst-case allocation, coinciding with the three-table '
                         'conservative corner).')
parser.add_argument('--conf_nonphasable', type=float, default=None,
                    help='DEPRECATED spelling: overrides --conf_nonphasable_astu when '
                         'given, so-era command lines behave. (Its ASE '
                         'half, --conf_nonphasable_ase / the conf_ase column, was '
                         'deleted—')
parser.add_argument('--event_mode', type=str, default='all_events',
                    choices=['all_events', 'switching_events', 'fdr_events'],
                    help='Event selection mode: all_events (default, test all), '
                         'switching_events (isoform-switching boundary events only), '
                         'fdr_events (events passing FDR cutoff)')
parser.add_argument('--fdr_events_value', type=float, default=0.05,
                    help='FDR cutoff for fdr_events mode')
parser.add_argument('--same_individual', action='store_true',
                    help='multi-sample lists are N samples of the SAME person '
                         ': '
                         'evidence is pooled into ONE variant call and ONE EM, '
                         'and each sample name becomes a CellType (tissues as '
                         'cell types — the GTEx donor semantics). WITHOUT this '
                         'flag a multi-sample list means N DIFFERENT samples, '
                         'processed fully independently, one output tree per '
                         'sample under output_folder/<sample>/. The old '
                         'implicit behavior (pooled step1 + separate EMs) is '
                         'RETIRED — it silently pooled different individuals.')
parser.add_argument('--job_array_by_sample', action='store_true',
                    help='When set for step4/step5, process only sample job_index from the full multi-sample input lists.')

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


parser.add_argument('--gtf_path', type=str, default=None,
                    help='Reference annotation GTF for lightweight mode (light_prep/light_merge). '
                         'Replaces the SCOTCH dependency: gene structures and read->gene '
                         'assignment are derived from this GTF + the BAM directly.')


parser.add_argument('--isoquant_dir', type=str, default=None,
                    help='isoquant_prep: the IsoQuant output directory (its '
                         'read_assignments / transcript_model_reads / extended_annotation '
                         'are found by suffix, under --isoquant_prefix when given)')
parser.add_argument('--isoquant_prefix', type=str, default=None,
                    help='isoquant_prep: the --prefix IsoQuant was run with (subdirectory)')
parser.add_argument('--isoquant_keep_policy', type=str, default='default',
                    choices=['default', 'strict'],
                    help="isoquant_prep: which reads count (INPUT_CONTRACT.2). 'default' "
                         "= IsoQuant's unique assignments + same-gene ambiguous reads at the "
                         "gene level + supported novel models; 'strict' = unique assignments only")
parser.add_argument('--novel_min_support_reads', type=int, default=2,
                    help='isoquant_prep: an IsoQuant novel transcript model counts as an '
                         'isoform when at least this many reads support it ('
                         'novel isoforms count; the number is an engineering default)')
parser.add_argument('--isoquant_force_version', action='store_true',
                    help='isoquant_prep: run on an IsoQuant version outside the tested set')
parser.add_argument('--isoquant_no_model_construction', action='store_true',
                    help='isoquant_prep: IsoQuant was run without model construction (no '
                         'transcript_model_reads table): novel isoforms are not used. Without '
                         'this flag a missing table is an error, not a silent downgrade')
parser.add_argument('--barcode_cell', type=str, default='CB',
                    help='Cell barcode BAM tag for light_prep / isoquant_prep (default CB)')
parser.add_argument('--barcode_umi', type=str, default='UB',
                    help='UMI BAM tag for light_prep (default UB)')
parser.add_argument('--light_min_exonic_bp', type=int, default=0,
                    help='light_prep adjudication: minimum exonic-overlap bp for a read to be '
                         'assigned to its best gene. The DECIDED value is 0 ('
                         'FINAL: light mode is gene_range, where an exonic floor could only '
                         'discard the intronic reads the mode exists to admit); tracked home '
                         'is DECIDED_READ_GATES, and this default is kept EQUAL to it '
                         'deliberately -- a bare default that differs from the decided value '
                         'is a silent fallback, which cost a run on. Explicit '
                         'exonic scoring must pass a floor >= 1 (validated).')
parser.add_argument('--light_assign_by', type=str, default='gene_range',
                    choices=['exonic', 'gene_range'],
                    help="light_prep read->gene scoring. 'gene_range' (default, FINAL "
                         ", clf arm 4/4 platforms): "
                         "overlap with the whole gene span. 'exonic' (the pre-"
                         "shipped rule, kept for reproducing older results): overlap with "
                         "the gene's exon union -- a purely intronic read scores 0 and is "
                         "never assigned; requires an explicit floor >= 1. "
                         "'gene_range': overlap with the whole gene span, introns "
                         "included -- longcallR's rule, and the only one under which an "
                         "intronic read can be assigned. Measured on isoseq "
                         "98.8%% of the reads the >=30bp exonic gate discarded had ZERO "
                         "exonic overlap, which is why lowering that gate to 1 bought "
                         "+0.21%%. gene_range requires --light_min_exonic_bp 0. "
                         "⚠️ It admits reads, it does not prove they are assigned to the "
                         "right gene: judge it on the endpoint metric.")
parser.add_argument('--light_ambiguity_ratio', type=float, default=1.0,
                    help='light_prep adjudication: best gene must have >= this ratio x the '
                         'second-best exonic overlap, else the read is ambiguous. Default 1.0, '
                         'the decided value: at 1.0 the test never fires (the winner always ties '
                         'or beats the runner-up), so reads go to the gene with the most exonic '
                         'overlap and none is discarded for ambiguity. It read 2.0 here and 1.0 '
                         'in the presets until, when a script that omitted --platform '
                         'silently got 2.0 and lost 132k isoseq reads.')
parser.add_argument('--target_units', type=int, default=None,
                    help='light_prep sharding granularity: number of read-balanced work '
                         'units to aim for (default max(64, 8*n_jobs)). Raise (e.g. 512) '
                         'when one dense window still dominates a shard. All array jobs '
                         'of one run must use the SAME value (fingerprint-enforced).')
parser.add_argument('--fast_pileup_min_frac', type=float, default=0.05,
                    help='step1 prescreen gate (DEFAULT ON, ratifiedafter '
                         'real-data audit): minimum whitelist-aligned alt fraction for a '
                         'column to enter the exact per-read pileup. The VALUE is the '
                         'control: 0.05 default sits 3x below the het-prefilter band '
                         '(~15%%) and above sequencing-error columns; set 0 to disable '
                         'the prescreen and run the legacy full-interval scan.')
parser.add_argument('--fast_pileup_raw_frac', type=float, default=0.02,
                    help='step1 stage-0 gate (A1, default 0.02): RAW-count alt-fraction '
                         'prescreen in pure C (no per-read whitelist callback — the '
                         'whitelist is enforced with full rigor at the exact sweep). '
                         'Wider than the het band by ~7x; dilution tolerance ~25x. '
                         'Set 0 for the strict whitelist-aligned prescreen at '
                         '--fast_pileup_min_frac (slower on huge genes).')
parser.add_argument('--marker_span_from_reads', action='store_true',
                    help='step1 '
                         '): call candidate '
                         'SNVs over the union span of the gene\'s whitelisted reads '
                         'instead of the annotated gene interval, so every site those '
                         'reads cover can serve as a phasing marker. Read ownership is '
                         'unchanged; step3 tags sites outside the gene interval with '
                         'in_gene_span=0. Default off (experimental arm).')
parser.add_argument('--pileup_engine', type=str, default='walk',
                    choices=['column', 'walk'],
                    help="step1 exact-sweep engine: 'walk' (default; one fetch per gene "
                         "+ manual CIGAR walk, O(reads x (ops+hits)); unpaired/long-read "
                         "BAMs only) or 'column' (per-candidate truncated pileups, "
                         "legacy fallback and the only option for paired BAMs)")
parser.add_argument('--no_fast_pileup', action='store_true',
                    help='step1: force the legacy full-interval pileup (equivalent to '
                         '--fast_pileup_min_frac 0)')
parser.add_argument('--fast_pileup', action='store_true',
                    help='(deprecated no-op — the prescreen is on by default; tune or '
                         'disable via --fast_pileup_min_frac / --no_fast_pileup)')
parser.add_argument('--bulk', action='store_true',
                    help='light_prep: bulk (barcode-free) BAM — do not read CB/UB tags; '
                         'every read gets Cell="bulk" and Umi=<query_name> (unique, so no '
                         'spurious dedup). Matches SCOTCH --bulk semantics; per-cell counts '
                         'degrade to per-sample.')
parser.add_argument('--em_init', type=str, default='signed',
                    choices=['signed', 'concurrence'],
                    help="step3 EM haplotype initialization: 'signed' "
                         "(default; non-randomized signed-affinity eigenvector "
                         "2-coloring — 8-27 phasing campaign) or "
                         "'concurrence' (legacy alt-alt spectral clustering)")
parser.add_argument('--no_phase_flip', dest='phase_flip', action='store_false',
                    help='step3: disable the post-EM guarded suffix-flip '
                         '(three-cell adjudication: bridge>=2 reads, net '
                         'allele votes>=2, polished-likelihood improvement '
                         '- on by default)')
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
parser.add_argument('--step3_backend', type=str, default='threads',
                    choices=['threads', 'loky'],
                    help="step3 gene fan-out backend: 'threads' (legacy; GIL caps at "
                         "~2-6 effective cores) or 'loky' (processes; near-linear "
                         "scaling, pins worker BLAS threads to 1; pairs best with the "
                         "geneidx sidecar seek mode)")
parser.add_argument('--skip_ase_test', action='store_true',
                    help='step3: skip the ASE LRT (bulk + per-cell-type allelic-balance '
                         'test); its summary columns stay present but empty. The ASE test '
                         'needs only phasing, so it IS valid in lightweight mode — skip it '
                         'only when you want pure phasing/counts.')
parser.add_argument('--skip_astu_test', action='store_true',
                    help='step3: skip the isoform chi-squared (ASTU) tests; columns stay '
                         'present but empty. Recommended always-on in lightweight mode '
                         '(placeholder isoforms make the test df=0/meaningless).')
parser.add_argument('--skip_tests', action='store_true',
                    help='step3: skip BOTH the ASE LRT and the isoform chi-squared '
                         '(= --skip_ase_test --skip_astu_test). Kept for compatibility; '
                         'prefer the fine-grained flags.')


SAMPLE_TASKS = ('step1', 'step1_5', 'step1_5_merge', 'step2', 'step3',
                'step4', 'step5', 'check')


def _sample_names_of(args):
    if args.sample_names:
        return list(args.sample_names)
    return [os.path.basename(os.path.normpath(st)) for st in args.scotch_target]


def run_with_sample_modes(args, dispatch):
    n = len(args.scotch_target) if args.scotch_target else 1
    if args.task not in SAMPLE_TASKS or n <= 1:


        if (args.task in SAMPLE_TASKS and args.scotch_target
                and args.bam_path and len(args.bam_path) > len(args.scotch_target)):
            raise SystemExit(
                f'{len(args.bam_path)} --bam_path entries for '
                f'{len(args.scotch_target)} --scotch_target: give one BAM per '
                f'sample (extra BAMs used to be silently ignored).')
        dispatch()
        return
    if args.bam_path and len(args.bam_path) != n:
        raise SystemExit(f'{len(args.bam_path)} --bam_path entries for {n} '
                         f'samples: give one per sample.')
    names = _sample_names_of(args)
    if len(names) != n:
        raise SystemExit(f'{len(names)} --sample_names for {n} samples.')
    _given = args.snv_confidence_path
    if isinstance(_given, str):
        _given = [x.strip() for x in _given.split(',') if x.strip()]
    if args.same_individual:
        if _given and len(_given) > 1:
            raise SystemExit(
                '--same_individual takes at most ONE --snv_confidence_path — '
                'one person has one genotype.')


        u.resolve_mapq_policy(args)
        pooled = u.build_same_individual_inputs(
            scotch_targets=args.scotch_target,
            bam_paths=args.bam_path,
            sample_names=names,
            out_dir=args.output_folder,
            cell_type_df_path=args.cell_type_df_path,
            need_bam=args.task in ('step1', 'step1_5', 'step5'))
        args.scotch_target = [pooled['scotch_target']]
        args.bam_path = ([pooled['bam_path']] if pooled.get('bam_path')
                         else (args.bam_path[:1] if args.bam_path else None))
        args.cell_type_df_path = [pooled['cell_type_df_path']]
        args.sample_names = None
        if args.task in ('step1_5', 'step1_5_merge'):

            args.n_jobs, args.job_index = 1, 0
        dispatch()
        return

    if len(set(names)) != len(names):
        raise SystemExit(
            f'multi-sample independent mode needs distinct sample names, got '
            f'{names} — pass --sample_names to disambiguate.')
    idxs = range(n)
    if args.job_array_by_sample:
        if args.job_index < 0 or args.job_index >= n:
            raise SystemExit(f'--job_index {args.job_index} out of range for '
                             f'{n} samples')
        idxs = [args.job_index]
    base = dict(scotch_target=list(args.scotch_target),
                bam_path=list(args.bam_path) if args.bam_path else None,
                cell_type_df_path=(list(args.cell_type_df_path)
                                   if args.cell_type_df_path else None),
                snv_confidence_path=args.snv_confidence_path,
                output_folder=args.output_folder,
                sample_names=args.sample_names,
                job_array_by_sample=args.job_array_by_sample,
                n_jobs=args.n_jobs,
                job_index=args.job_index)


    pristine = {k: (list(v) if isinstance(v, list) else v)
                for k, v in vars(args).items()}

    def _reset_args():
        for k in set(vars(args)) - set(pristine):
            delattr(args, k)
        for k, v in pristine.items():
            setattr(args, k, list(v) if isinstance(v, list) else v)
    given_lists = None
    if _given:
        if len(_given) == n:
            given_lists = list(_given)
        elif len(_given) != 1:
            raise SystemExit(
                f'--snv_confidence_path has {len(_given)} entries for {n} '
                f'samples — give one, or one per sample.')
        else:
            given_lists = [_given[0]] * n
    try:
        for i in idxs:
            _reset_args()
            name = names[i]
            args.scotch_target = [base['scotch_target'][i]]
            args.bam_path = ([base['bam_path'][i]]
                             if base['bam_path'] else None)
            if base['cell_type_df_path']:
                if len(base['cell_type_df_path']) == n:
                    args.cell_type_df_path = [base['cell_type_df_path'][i]]
                else:
                    args.cell_type_df_path = list(base['cell_type_df_path'])
            if given_lists is not None:


                args.snv_confidence_path = [given_lists[i]]
            args.sample_names = [name]
            args.output_folder = os.path.join(base['output_folder'], name)
            os.makedirs(args.output_folder, exist_ok=True)
            args.job_array_by_sample = False
            if base['job_array_by_sample']:


                args.n_jobs, args.job_index = 1, 0
            if args.task in ('step1_5', 'step1_5_merge'):

                args.n_jobs, args.job_index = 1, 0
            print(f"=== independent multi-sample mode: [{i + 1}/{n}] "
                  f"{name} -> {args.output_folder} ===")
            dispatch()
    finally:
        _reset_args()


def _log_mapq_policy(logger):
    pol = getattr(args, '_mapq_policy', None)
    if pol is None:
        return
    if pol['auto_fallback'] or pol['kept_user_flags']:
        logger.warning(f"⚠️ MAPQ unavailable in the BAM: automatic fallback "
                       f"changed {pol['changed']} ; user flags kept "
                       f"{pol['kept_user_flags']} ")
    else:
        logger.info(f"mapq: {pol['verdict']} "
                    + '; '.join(f"{pr['bam']}: 0 {pr['zero']}/{pr['sampled']}, "
                                f"255 {pr['v255']}/{pr['sampled']}, "
                                f"1..254 {pr['valid']}/{pr['sampled']}"
                                for pr in pol['probe']))


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


def parse_het_beta(raw):
    if not raw:
        return None
    parts = [x.strip() for x in str(raw).split(',')]
    if len(parts) != 2:
        raise SystemExit(f"--het_beta must be 'a,b' (e.g. '2,2'), got {raw!r}")
    hb = (float(parts[0]), float(parts[1]))
    if hb[0] <= 0 or hb[1] <= 0:
        raise SystemExit(f"--het_beta needs positive shape parameters, got {hb}")
    return hb


def resolve_het_thresholds(args, logger=None):


    def _pick(new_v, old_v, new_name, old_name, default):
        if not _was_given(old_v):
            return default if not _was_given(new_v) else new_v
        if _was_given(new_v):
            raise SystemExit(f'{new_name} and {old_name} were both given '
                             f'({new_v} vs {old_v}). They are the same knob; '
                             f'pass one.')
        mes = (f'[DEPRECATED] {old_name} is now {new_name}; using {old_v}. '
               f'The old spelling still works and will keep working.')
        print(mes) if logger is None else logger.warning(mes)
        return old_v

    step1 = _pick(args.het_prob_step1, getattr(args, 'het_prefilter_threshold', None),
                  '--het_prob_step1', '--het_prefilter_threshold',
                  float(HET_STEP1_DEFAULT))
    step3 = _pick(args.het_prob_step3, getattr(args, 'heterozygous_filter', None),
                  '--het_prob_step3', '--heterozygous_filter',
                  float(HET_STEP3_DEFAULT))
    if step1 >= 0 and step3 >= 0 and step1 > step3:


        mes = (f'[WARN] --het_prob_step1 {step1} is numerically above '
               f'--het_prob_step3 {step3}. That is usually a wiring mistake -- '
               f'but the two are not the same statistic (step1: aggregate '
               f'counts, P(het)=0.6; step3: per-read qualities, P(het)=0.1), so '
               f'it is not necessarily redundant and is not refused.\n'
               f' Note the two cutoffs are INDEPENDENT as of '
               f'step1 no longer follows step3, so set both if you meant to.')
        print(mes) if logger is None else logger.warning(mes)

    args.heterozygous_filter = step3
    return step1, step3


def load_gene_subset(path):
    if path is None:
        return None
    if os.path.isfile(path):
        with open(path) as fh:
            return [line.strip() for line in fh if line.strip()]
    return [g.strip() for g in path.split(',') if g.strip()]


def write_done_marker(output_folder, step, job_index=None):
    import datetime
    marker_dir = os.path.join(output_folder, 'job_markers')
    os.makedirs(marker_dir, exist_ok=True)
    fname = (f'{step}_job{job_index}.done' if job_index is not None
             else f'{step}.done')
    with open(os.path.join(marker_dir, fname), 'w') as fh:
        fh.write(datetime.datetime.now().isoformat() + '\n')


def _load_given_snv(paths, bam_path, stage, want, logger=None):
    if not paths:
        return None
    if stage != 'both' and stage != want:
        return None
    bams = list(bam_path or [])
    if len(paths) != len(bams):
        raise SystemExit(
            f'--snv_confidence_path has {len(paths)} file(s) but --bam_path has '
            f'{len(bams)} BAM(s). They are matched by position, one genotype list '
            f'per sample; a shared list would be wrong for every sample but one.')
    out = []
    for i, (pth, bam) in enumerate(zip(paths, bams)):
        df = pd.read_csv(pth, sep='\t')
        need = {'chrom', 'pos'}
        missing = need - set(df.columns)
        if missing:
            raise SystemExit(f'{pth}: missing column(s) {sorted(missing)}')
        msg = (f'[genotype] sample{i}: {len(df)} supplied site(s) from '
               f'{os.path.basename(pth)} for {os.path.basename(str(bam))}')
        print(msg) if logger is None else logger.info(msg)
        out.append(df)
    return out

def main():
    global args
    args = parser.parse_args()
    if args.clf_init:
        print('[deprecated] --clf_init is ignored: h_m initialization is now '
              'selected by --h_m_init_from (default clf, which uses classifier '
              'scores whenever a model is loaded). Remove the flag.')
    u.apply_platform_preset(args)
    u.check_gate_consistency(args.min_alt_frac, args.fast_pileup_min_frac,
                             fast_pileup_raw_frac=args.fast_pileup_raw_frac)


    def light_prep():
        import src.light_upstream as lu
        os.makedirs(args.output_folder, exist_ok=True)
        logger, _ = setup_logger(args.output_folder, 'light_prep')
        logger.info('Start running light_prep: GTF-direct read->gene assignment...')
        if not args.gtf_path or not args.bam_path:
            raise SystemExit('--task light_prep requires --gtf_path and --bam_path')
        if len(args.bam_path) != 1:
            raise SystemExit('light_prep takes exactly one --bam_path per output '
                             'folder; run one lightweight directory per sample')
        if args.sample_name_parse:
            raise SystemExit('--sample_name_parse is incompatible with lightweight '
                             'mode: the light directory uses the root-level '
                             'auxillary/ layout (one directory per sample)')
        logger.info(f'bam: {args.bam_path[0]}  gtf: {args.gtf_path}')
        logger.info(f'job {args.job_index}/{args.n_jobs}  bulk: {args.bulk}  tags: '
                    f'{args.barcode_cell}/{args.barcode_umi}  min_mapq: {args.min_mapq}  '
                    f'min_exonic_bp: {args.light_min_exonic_bp}  '
                    f'ambiguity_ratio: {args.light_ambiguity_ratio}  '
                    f'assign_by: {args.light_assign_by}')
        lu.run_light_prep(args.bam_path[0], args.gtf_path, args.output_folder,
                          n_jobs=args.n_jobs, job_index=args.job_index,
                          cell_tag=args.barcode_cell, umi_tag=args.barcode_umi,
                          min_mapq=args.min_mapq,
                          min_exonic_bp=args.light_min_exonic_bp,
                          ambiguity_ratio=args.light_ambiguity_ratio,
                          assign_by=args.light_assign_by,
                          bulk=args.bulk, target_units=args.target_units,
                          logger=logger)
        write_done_marker(args.output_folder, 'light_prep', args.job_index)
        logger.info(f'Finished light_prep job {args.job_index}')

    def light_merge():
        import src.light_upstream as lu
        logger, _ = setup_logger(args.output_folder, 'light_merge')
        logger.info('Start running light_merge: assembling SCOTCH-shaped directory...')
        if not args.gtf_path:
            raise SystemExit('--task light_merge requires --gtf_path')
        if args.sample_name_parse:
            raise SystemExit('--sample_name_parse is incompatible with lightweight '
                             'mode (root-level auxillary/ layout)')
        light_dir = lu.run_light_merge(
            args.gtf_path, args.output_folder, n_jobs=args.n_jobs, logger=logger)
        write_done_marker(args.output_folder, 'light_merge')
        logger.info(f'Finished light_merge; run steps 1-4 with '
                    f'--scotch_target {light_dir}')


    def isoquant_prep():
        import src.isoquant2longallele as iq
        os.makedirs(args.output_folder, exist_ok=True)
        logger, _ = setup_logger(args.output_folder, 'isoquant_prep')
        logger.info('Start running isoquant_prep: IsoQuant output -> upstream directory...')
        if not args.isoquant_dir or not args.gtf_path or not args.bam_path:
            raise SystemExit('--task isoquant_prep requires --isoquant_dir, --gtf_path '
                             '(the reference GTF IsoQuant was given) and --bam_path')
        if len(args.bam_path) != 1:
            raise SystemExit('isoquant_prep takes exactly one --bam_path per output '
                             'folder; run one IsoQuant directory per sample')
        found = iq.find_isoquant_outputs(args.isoquant_dir, args.isoquant_prefix)
        if 'read_assignments' not in found or 'extended_gtf' not in found:
            raise SystemExit(f'no IsoQuant read_assignments / extended_annotation under '
                             f'{args.isoquant_dir} (prefix {args.isoquant_prefix}); found {found}')
        out_dir = os.path.join(args.output_folder, 'isoquant_upstream')
        if found.get('transcript_model_reads') is None and not args.isoquant_no_model_construction:
            raise SystemExit(f'no transcript_model_reads table under {args.isoquant_dir}: rerun '
                             f'IsoQuant with --large_output read_assignments read2transcripts '
                             f'(novel isoforms come from that table), or pass '
                             f'--isoquant_no_model_construction to proceed without novel isoforms')
        config = iq.AdapterConfig(
            read_assignments=found['read_assignments'], extended_gtf=found['extended_gtf'],
            reference_gtf=args.gtf_path, bam=args.bam_path[0], out_dir=out_dir,
            transcript_model_reads=found.get('transcript_model_reads'),
            model_construction_enabled=not args.isoquant_no_model_construction,
            keep_policy=iq.KeepPolicy(args.isoquant_keep_policy),
            cell_tag=args.barcode_cell, umi_tag=args.barcode_umi, bulk_mode=args.bulk,
            novel_min_support_reads=args.novel_min_support_reads,
            force_version=args.isoquant_force_version)
        if not config.model_construction_enabled:
            logger.warning('--isoquant_no_model_construction: IsoQuant novel models are NOT used')
        logger.info(f'isoquant: {found}  gtf: {args.gtf_path}  bam: {args.bam_path[0]}  '
                    f'policy: {config.keep_policy.value}  novel_min_support_reads: '
                    f'{config.novel_min_support_reads}  bulk: {args.bulk}  tags: '
                    f'{args.barcode_cell}/{args.barcode_umi}')
        iq.run_adapter(config, logger=logger)
        write_done_marker(args.output_folder, 'isoquant_prep')
        logger.info(f'Finished isoquant_prep; run steps 1-5 with --scotch_target {out_dir}')


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
        _log_mapq_policy(logger)
        het_prefilter, _ = resolve_het_thresholds(args)
        logger.info(f'het_prefilter_threshold set as: {het_prefilter}')


        logger.info(f'het_beta set as: {parse_het_beta(args.het_beta)} '
                    f'({"fixed p=0.5, the historical model" if not args.het_beta else "marginal over the ALT fraction"})')
        logger.info(f'fast_pileup prescreen: '
                    f'{"OFF (legacy scan)" if args.no_fast_pileup or args.fast_pileup_min_frac == 0 else f"ON @ min_frac {args.fast_pileup_min_frac}"}')
        _stage, _stage_src = resolve_genotype_stage(args)
        logger.info(f'genotype_stage={_stage} ({_stage_src})')
        _given = _load_given_snv(args.snv_confidence_path, args.bam_path,
                                 _stage, 'step1', logger=logger)
        if _given is not None:
            logger.info('[genotype] step1 restricts to the supplied sites and runs '
                        'NO candidate filter (no depth / alt / AF / het_prob gate)')
        vc = u.VariantCaller(scotch_target=args.scotch_target, bam_path=args.bam_path,
                             ref_fasta_path=args.ref_fasta_path, ref_pickle_path=args.ref_pickle_path,
                             target=args.output_folder, given_snv=_given,
                             n_jobs=args.n_jobs, job_index=args.job_index,
                             depth=args.depth, n_alt=args.n_alt_count, min_alt_frac=args.min_alt_frac, max_baseq=args.max_baseq,
                             gene_guard_depth=args.gene_guard_depth,
                             min_mapq=args.min_mapq, min_baseq=args.min_baseq,
                             min_dist_to_end=args.min_dist_to_end,
                             sample_name_parse=args.sample_name_parse,
                             sample_names=args.sample_names,
                             gene_subset=load_gene_subset(args.gene_subset_path),
                             het_prefilter_threshold=het_prefilter,
                             het_beta=parse_het_beta(args.het_beta),
                             fast_pileup=(False if args.no_fast_pileup else None),
                             fast_pileup_min_frac=args.fast_pileup_min_frac,
                             fast_pileup_raw_frac=args.fast_pileup_raw_frac,
                             pileup_engine=args.pileup_engine,
                             marker_span_from_reads=args.marker_span_from_reads,
                             logger=logger)


        u.ensure_gate_config(args.output_folder, args.n_alt_count, args.depth,
                             args.min_alt_frac, logger=logger,
                             strict=not args.ignore_gate_mismatch,
                             het_beta=parse_het_beta(args.het_beta),
                             het_prob_step1=het_prefilter,
                             setting=getattr(args, 'setting', None),
                             mapq_policy=getattr(args, '_mapq_policy', None))
        vc.process_genes_round1_1()
        write_done_marker(args.output_folder, 'step1', args.job_index)
        logger.info(f'Finished initial variant calling for job {args.job_index}')


    def collect_read_blocks():
        logger, _ = setup_logger(args.output_folder, 'step1_5_readblocks')
        logger.info('Start running step1.5: per-BAM read_blocks collection...')
        logger.info(f'sample_index (job_index): {args.job_index} / n_samples={args.n_jobs}')
        logger.info(f'bam file paths: {args.bam_path}')
        het_prefilter, _ = resolve_het_thresholds(args)
        vc = u.VariantCaller(scotch_target=args.scotch_target, bam_path=args.bam_path,
                             ref_fasta_path=args.ref_fasta_path, ref_pickle_path=args.ref_pickle_path,
                             target=args.output_folder,
                             n_jobs=args.n_jobs, job_index=args.job_index,
                             depth=args.depth, n_alt=args.n_alt_count, min_alt_frac=args.min_alt_frac, max_baseq=args.max_baseq,
                             gene_guard_depth=args.gene_guard_depth,
                             min_mapq=args.min_mapq, min_baseq=args.min_baseq,
                             min_dist_to_end=args.min_dist_to_end,
                             sample_name_parse=args.sample_name_parse,
                             sample_names=args.sample_names,
                             gene_subset=load_gene_subset(args.gene_subset_path),
                             het_prefilter_threshold=het_prefilter,
                             het_beta=parse_het_beta(args.het_beta),
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
                             depth=args.depth, n_alt=args.n_alt_count, min_alt_frac=args.min_alt_frac, max_baseq=args.max_baseq,
                             gene_guard_depth=args.gene_guard_depth,
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
                             depth=args.depth, n_alt=args.n_alt_count, min_alt_frac=args.min_alt_frac, max_baseq=args.max_baseq,
                             gene_guard_depth=args.gene_guard_depth,
                             min_mapq=args.min_mapq, min_baseq=args.min_baseq,
                             min_dist_to_end=args.min_dist_to_end,
                             sample_name_parse=args.sample_name_parse,
                             sample_names=args.sample_names,
                             gene_subset=load_gene_subset(args.gene_subset_path),
                             n_workers=args.n_workers,
                             logger=logger)
        vc.process_genes_final()
        write_done_marker(args.output_folder, 'step2', args.job_index)
        logger.info(f'Finished generating em input files for job {args.job_index}')


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
        logger.info(f'marker_span_from_reads: {args.marker_span_from_reads}')
        logger.info(f'Predefined cell type file: {args.cell_type_df_path}')
        logger.info(f'gene_subset_path set as: {args.gene_subset_path}')

        logger.info(f'high_artifact_mode: {args.high_artifact_mode}')
        if args.high_artifact_mode:
            logger.info(f'  novel_exon_pct_max (Knob B cutoff): {args.novel_exon_pct_max}')
            logger.info(f'  read_intronic_pct_max (Knob C cutoff): {args.read_intronic_pct_max}')
            logger.info(f'  gsi_base_pkl_path (auto-resolved if None): {args.gsi_base_pkl_path}')
        logger.info(f'read_sj_min (Knob D truncation filter): {args.read_sj_min}')
        skip_ase = args.skip_ase_test or args.skip_tests
        skip_astu = args.skip_astu_test or args.skip_tests
        logger.info(f'skip_ase_test: {skip_ase}  skip_astu_test: {skip_astu}'
                    f'  (--skip_tests: {args.skip_tests})')
        _stage, _stage_src = resolve_genotype_stage(args)
        logger.info(f'genotype_stage={_stage} ({_stage_src})')
        _given3 = _load_given_snv(args.snv_confidence_path, args.bam_path,
                                  _stage, 'step3', logger=logger)


        if _given3 is None:
            snv_confidence = None
            if args.snv_confidence_path:
                logger.info('[genotype] step3 is NOT using the supplied sites '
                            f'(--genotype_stage {_stage}); its own '
                            'filters run as usual')
        elif len(_given3) == 1:
            snv_confidence = _given3[0]
        elif args.job_array_by_sample:

            snv_confidence = _given3[args.job_index]
            logger.info(f'[genotype] step3 using list #{args.job_index} of '
                        f'{len(_given3)} (one sample per job)')
        else:


            raise SystemExit(
                f'--genotype_stage {_stage} with {len(_given3)} genotype '
                f'lists needs --job_array_by_sample: step3 otherwise processes all '
                f'samples in one pass against a single list, which would apply one '
                f'sample\'s genotype to another\'s reads. Run one job per sample.')
        _hb = parse_het_beta(args.het_beta)
        _t1, _t3 = resolve_het_thresholds(args)
        _log_mapq_policy(logger)
        u.check_gate_config(args.output_folder, args.n_alt_count, args.depth,
                            args.min_alt_frac, het_beta=_hb,
                            het_prob_step1=_t1,
                            setting=getattr(args, 'setting', None),
                            logger=logger,
                            strict=not args.ignore_gate_mismatch,
                            mapq_policy=getattr(args, '_mapq_policy', None))
        logger.info(f'het_prob cutoffs: step1 {_t1}, step3 {_t3}')
        logger.info(f'het_beta set as: {_hb} '
                    f'({"fixed p=0.5, the historical model" if _hb is None else "marginal over the ALT fraction"})')
        logger.info(f'shrink_denominator set as: {args.shrink_denominator}')
        logger.info(f'coverage_factor set as: {args.coverage_factor}')


        import hashlib as _hl
        def _digest(paths):
            out = []
            for pth in (paths or []):
                try:
                    with open(pth, 'rb') as fh:
                        out.append(os.path.basename(str(pth)) + ':' + _hl.md5(fh.read()).hexdigest()[:12])
                except OSError:
                    out.append(os.path.basename(str(pth)) + ':unreadable')
            return out
        _step3_params = {
            'platform': getattr(args, 'platform', None), 'setting': getattr(args, 'setting', None),
            'genotype_stage': _stage,
            'snv_confidence': _digest(args.snv_confidence_path) if snv_confidence is not None else [],
            'het_beta': u._gate_cfg(0, 0, 0.0, het_beta=_hb).get('het_beta'),
            'het_prob_step3': float(_t3) if _t3 is not None else None,
            'seed': args.seed, 'max_iter': args.max_iter, 'tol': args.tol,
            'em_max_reads': int(args.em_max_reads or 0), 'h_m_init_from': args.h_m_init_from,
            'em_init': args.em_init, 'phase_flip': bool(args.phase_flip), 'gap_tau': args.gap_tau,
            'snv_classifier': os.path.basename(str(args.snv_classifier)) if args.snv_classifier else None,
            'clf_hard_threshold': args.clf_hard_threshold,
            'high_artifact_mode': bool(args.high_artifact_mode),
            'alt_stretch_len': args.alt_stretch_len, 'var_cluster_window': args.var_cluster_window,
            'var_cluster_n': args.var_cluster_n,
            'rna_editing_db': os.path.basename(str(args.rna_editing_db)) if args.rna_editing_db else None,
            'editing_exempt_affinity': args.editing_exempt_affinity,
            'shrink_denominator': args.shrink_denominator, 'coverage_factor': args.coverage_factor,
            'chi_min_frac': args.chi_min_frac, 'chi_group_novel': bool(args.chi_group_novel),
            'cell_type_df': [os.path.basename(str(x)) for x in (args.cell_type_df_path or [])],


            **u.mapq_record_fields(getattr(args, '_mapq_policy', None)),
        }
        u.ensure_step3_config(args.output_folder, args.prefix, _step3_params,
                              logger=logger, cover_existing=args.cover_existing)
        ht = u.Haplotyping(scotch_target=args.scotch_target, bam_path=args.bam_path,
                           het_beta=_hb, shrink_denominator=args.shrink_denominator,
                           heterozygous_coverage_factor=args.coverage_factor,
                           target=args.output_folder, ref_pickle_path=args.ref_pickle_path,
                           max_iter=args.max_iter, tol=args.tol, verbose=args.verbose, seed=args.seed,
                           mtx=args.mtx, csv=args.csv, n_jobs=args.n_jobs, job_index=args.job_index,
                           n_alt=args.n_alt_count, min_alt_frac=args.min_alt_frac, depth=args.depth,
                           chi_min_frac=args.chi_min_frac, chi_group_novel=args.chi_group_novel,
                           heterozygous_filter=args.heterozygous_filter,alt_stretch_filter = args.alt_stretch_filter,alt_stretch_len=args.alt_stretch_len,init_link_min_agreement=args.init_link_min_agreement,init_link_min_shared=args.init_link_min_shared,phase_block_min_shared=args.phase_block_min_shared,phase_block_min_agreement=args.phase_block_min_agreement,h_m_init_from=args.h_m_init_from,
                           repeat_filter_kmer=args.repeat_filter_kmer,
                           alt_cluster_filter = args.alt_cluster_filter, ref_fasta_path=args.ref_fasta_path,
                           var_cluster_window = args.var_cluster_window, var_cluster_n = args.var_cluster_n,
                           sample_name_parse=args.sample_name_parse, prefix=args.prefix,
                           em_snv_filter=args.em_snv_filter, snv_confidence=snv_confidence,
                           max_baseq=args.max_baseq,
                           snv_classifier=args.snv_classifier,
                           clf_hard_threshold=args.clf_hard_threshold,
                           clf_init=args.clf_init,
                           gap_tau=args.gap_tau,
                           em_max_reads=args.em_max_reads,
                           clf_pruning_threshold=args.clf_pruning_threshold,
                           clf_pruning_frac=args.clf_pruning_frac,
                           rna_editing_db=args.rna_editing_db,
                           cell_type_df_path=args.cell_type_df_path, cover_existing = args.cover_existing,
                           gene_subset=load_gene_subset(args.gene_subset_path),


                           n_workers = args.n_workers if args.n_workers > 1 else -1,
                           logger = logger,
                           high_artifact_mode=args.high_artifact_mode,
                           novel_exon_pct_max=args.novel_exon_pct_max,
                           read_intronic_pct_max=args.read_intronic_pct_max,
                           read_sj_min=args.read_sj_min,
                           gsi_base_pkl_path=args.gsi_base_pkl_path,
                           skip_ase_test=skip_ase, skip_astu_test=skip_astu,
                           step3_backend=args.step3_backend,
                           editing_exempt_affinity=args.editing_exempt_affinity,
                           editing_exempt_min_reads=args.editing_exempt_min_reads,
                           em_init_method=args.em_init,
                           phase_flip=args.phase_flip)
        ht.generate_count_hap_genes()
        write_done_marker(args.output_folder, 'step3', args.job_index)
        logger.info(f'Finished haplotyping for job {args.job_index}')


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
                           n_alt=args.n_alt_count, min_alt_frac=args.min_alt_frac, depth=args.depth,
                           heterozygous_filter=args.heterozygous_filter,
                           repeat_filter_kmer=args.repeat_filter_kmer,
                           sample_name_parse=args.sample_name_parse, prefix=args.prefix,
                           em_snv_filter=args.em_snv_filter, snv_confidence=None,
                           cell_type_df_path=args.cell_type_df_path, cover_existing = args.cover_existing,
                           n_workers = args.n_workers if args.n_workers > 1 else -1,
                           logger = logger,
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


    def downstream():


        for st in (args.scotch_target or []):
            if os.path.isfile(os.path.join(st, 'light_provenance.json')):
                raise SystemExit(
                    f'step5 unsupported on lightweight upstream {st} '
                    f'(upstream=gtf_direct has no isoform truth); rerun with a '
                    f'real SCOTCH/IsoQuant --scotch_target for isoform analyses')
        logger, _ = setup_logger(args.output_folder, 'step5_downstream')
        logger.info('Start running step5: downstream analyses...')
        logger.info(f'Output directory: {args.output_folder}')
        logger.info(f'SCOTCH directory: {args.scotch_target}')
        logger.info(f'event_min_reads: {args.event_min_reads}')
        logger.info(f'snv_event_distance: {args.snv_event_distance}')
        logger.info(f'astu_sig_only: {args.astu_sig_only}')
        logger.info(f'astu_sig_from_bulk: {args.astu_sig_from_bulk}')
        logger.info(f'astu_sig_threshold: {args.astu_sig_threshold}')
        _cu = args.conf_nonphasable if args.conf_nonphasable is not None else args.conf_nonphasable_astu
        logger.info(f'ase_call_margin (ASE_call door 2; door 1 = interval on one side of 0.5): {args.ase_call_margin}')
        logger.info(f'conf_nonphasable_astu (ASTU call gate): {_cu}')
        logger.info(f'event_mode: {args.event_mode}')


        if args.actv is None:
            args.actv = bool(args.cell_type_df_path)
            _actv_src = ('auto: cell types given' if args.actv
                         else 'auto: no cell types to compare')
        else:
            _actv_src = 'user flag'
        if args.actv_unit is None:
            args.actv_unit = 'read' if args.same_individual else 'cell'
        if args.actv:
            logger.info(f'[decided] actv=ON ({_actv_src}; unit={args.actv_unit}, '
                        f'B={args.actv_permutations}, '
                        f'min_cells={args.actv_min_cells}, '
                        f'min_phasable_reads={args.actv_min_phasable_reads}, '
                        f'min_phasable_frac={args.actv_min_phasable_frac}, '
                        f'min_actv={args.actv_min_actv}, '
                        f'max_attempts={args.actv_max_attempts if args.actv_max_attempts is not None else 10 * args.actv_permutations}, '
                        f'seed={args.seed})')
            if not args.cell_type_df_path and args.actv_unit == 'cell':
                logger.warning('⚠️ actv=ON but no --cell_type_df_path: the cross-cell-type '
                               'ACTV table cannot be computed for this run (Bulk only) '
                               '-- pass a cell-type table, or --actv_unit read for '
                               'pooled/bulk trees')
        else:
            logger.warning(f'⚠️ [decided] actv=OFF ({_actv_src}) -- no ACTV table will be '
                           f'written for this run')
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
            conf_nonphasable_astu=args.conf_nonphasable_astu,
            conf_nonphasable=args.conf_nonphasable,
            ase_call_margin=args.ase_call_margin,
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
            actv=args.actv,
            actv_permutations=args.actv_permutations,
            actv_min_cells=args.actv_min_cells,
            actv_min_phasable_reads=args.actv_min_phasable_reads,
            actv_max_attempts=args.actv_max_attempts,
            actv_min_phasable_frac=args.actv_min_phasable_frac,
            actv_min_actv=args.actv_min_actv,
            actv_seed=args.seed if args.seed is not None else 42,
            actv_unit=args.actv_unit,
        )
        write_done_marker(args.output_folder, 'step5')
        logger.info('Finished downstream analysis')

    def check():
        logger, _ = setup_logger(args.output_folder, 'check')
        logger.info('Checking job completion...')

        pfx = args.prefix or ''
        out = args.output_folder


        variants_dir = os.path.join(out, 'variant_align1', 'variants_by_gene')
        em_input_dir = os.path.join(out, 'em_input')
        summary_sep_dir = os.path.join(
            out,
            f'summary_statistics_{pfx}' if pfx else 'summary_statistics',
            'all_genes_separate')


        if args.scotch_target and len(args.scotch_target) > 1:
            sample_names = (args.sample_names
                            if args.sample_names
                            else [os.path.basename(s) for s in args.scotch_target])
            em_dirs = [os.path.join(em_input_dir, sn) for sn in sample_names]
        else:
            em_dirs = [em_input_dir]

        def gene_ids_by_suffix(directory, suffix):
            if not os.path.isdir(directory):
                return set()
            return {f.split('_')[0] for f in os.listdir(directory) if f.endswith(suffix)}


        s1_snvs = gene_ids_by_suffix(variants_dir, '_snvs.csv')
        s1_pkl  = gene_ids_by_suffix(variants_dir, '_site_reads.pkl')

        s1_partial = s1_snvs - s1_pkl
        s1_done = s1_snvs & s1_pkl

        logger.info(f'Step 1 | complete: {len(s1_done)}  '
                    f'partial (csv only): {len(s1_partial)}')
        if s1_partial:
            logger.info(f'  Partial step1 genes will cause step2 errors — '
                        f'rerun step1 for them.')


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


        s2_pile, s2_npz, s2_rsnv, s2_rpi = set(), set(), set(), set()
        for em_dir in em_dirs:
            s2_pile |= gene_ids_by_suffix(em_dir, '_pileup.csv')
            s2_npz  |= gene_ids_by_suffix(em_dir, '_read_matrices.npz')
            s2_rsnv |= gene_ids_by_suffix(em_dir, '_read_snv.csv')
            s2_rpi  |= gene_ids_by_suffix(em_dir, '_read_pi.csv')

        s2_has_em = s2_npz | (s2_rsnv & s2_rpi)
        s2_done = s2_pile & s2_has_em

        s2_any  = s2_pile | s2_npz | s2_rsnv | s2_rpi
        s2_partial = s2_any - s2_done

        s2_expected = s1_done
        s2_missing  = s2_expected - s2_done - s2_partial

        logger.info(f'Step 2 | complete: {len(s2_done)} / {len(s2_expected)}  '
                    f'partial: {len(s2_partial)}  '
                    f'not started: {len(s2_missing)}')


        s3_done = set()
        if os.path.isdir(summary_sep_dir):


            s3_done = {f.split('_')[-2]
                       for f in os.listdir(summary_sep_dir)
                       if f.endswith('_summary.csv')}
        s3_expected = s2_done
        s3_missing  = s3_expected - s3_done

        logger.info(f'Step 3 | complete: {len(s3_done)} / {len(s3_expected)}  '
                    f'missing: {len(s3_missing)}')


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

    def _dispatch():


        u.resolve_mapq_policy(args)
        if args.task=='light_prep':
            light_prep()
        if args.task=='isoquant_prep':
            isoquant_prep()
            return
        if args.task=='light_merge':
            light_merge()
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
            haplotype_summary(not args.no_summary_haplotype,
                              not args.no_summary_count)
        if args.task=='step5':
            downstream()
        if args.task=='check':
            check()

    run_with_sample_modes(args, _dispatch)


if __name__ == "__main__":
    main()
