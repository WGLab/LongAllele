#!/bin/bash
# ============================================================================
# LongAllele demo — 10 genes on chr22 from a public PBMC ONT long-read sample
# (Xu et al., Nature Communications 2026:
#  https://www.nature.com/articles/s41467-026-72665-5).
# Runs the full pipeline step1 -> step5.
#
# Usage:   cd examples/pbmc_demo && bash run_demo.sh
# Output:  ./demo_output/   (compare against ./expected_output/)
# Runtime: a few minutes on a laptop.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

LA=../../src/longallele.py
CLF=../../src/models/snv_classifier_ont_hg001_17feat.joblib
SCOTCH=scotch_target
BAM=demo.bam
REF=ref/chr22.fa.gz
CT=sample7_celltype.csv
OUT=demo_output
PREFIX=demo

COMMON="--scotch_target $SCOTCH --bam_path $BAM --ref_fasta_path $REF \
        --output_folder $OUT --prefix $PREFIX --seed 42"
# variant-calling / classifier / filter settings (match the paper run)
CALL="--depth 20 --n_alt_count 10 --min_mapq 20 --min_baseq 5 --min_dist_to_end 3 \
      --heterozygous_filter 0.99 --snv_classifier $CLF --clf_init \
      --clf_hard_threshold 0.05 --gap_tau 1.0"

mkdir -p $OUT

echo "### step1  — variant calling (pileup -> SNV candidates)"
python $LA --task step1 $COMMON $CALL --n_jobs 1 --job_index 0

echo "### step1.5 — per-BAM read_blocks (enables step5 raw-read validation)"
python $LA --task step1_5 $COMMON $CALL --n_jobs 1 --job_index 0
python $LA --task step1_5_merge $COMMON $CALL

echo "### step2  — build EM input matrices"
python $LA --task step2 $COMMON $CALL

echo "### step3  — EM haplotyping"
python $LA --task step3 $COMMON $CALL --cell_type_df_path $CT

echo "### step4  — summary statistics + count matrices"
python $LA --task step4 $COMMON --cell_type_df_path $CT \
        --summary_haplotype --summary_count --csv

echo "### step5  — downstream: ASE / ASTU effect sizes + haplotype-event tests"
python $LA --task step5 $COMMON --cell_type_df_path $CT \
        --event_min_reads 10 --snv_event_distance 50 --event_mode all_events --n_workers 2

echo "### done — results in $OUT/"
