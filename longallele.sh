#!/usr/bin/env bash
# longallele.sh — submit the LongAllele pipeline as a chain of SLURM jobs.
#
# Usage:
#   cp config_template.sh my_run.sh   # fill in paths and settings
#   bash longallele.sh my_run.sh
#
# Job dependency graph:
#   step1 ──┬──→ step1_5 ──→ step1_5_merge ─┐
#           └──→ step2 ──→ step3 ──→ step4 ─┴──→ step5

set -euo pipefail

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG="${1:-}"
if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
    echo "Usage: bash longallele.sh <config.sh>"
    exit 1
fi
source "$CONFIG"

# ── Validate required variables ───────────────────────────────────────────────
for var in SCOTCH_TARGET BAM_PATH REF_FASTA OUTPUT_DIR N_GENE_JOBS N_SAMPLES \
           PARTITION MEM_12 TIME_12 CPUS_12 MEM_3 TIME_3 CPUS_3 MEM_4 TIME_4 CPUS_4 \
           MEM_5 TIME_5 CPUS_5; do
    if [[ -z "${!var:-}" ]]; then
        echo "Error: '$var' is not set in $CONFIG"
        exit 1
    fi
done

# ── Build optional argument fragments ─────────────────────────────────────────
CELL_OPT="${CELL_TYPE_DF:+--cell_type_df_path $CELL_TYPE_DF}"
PREFIX_OPT="${PREFIX:+--prefix $PREFIX}"

# Default MEM_15/TIME_15/CPUS_15 from the steps-1-2 settings if not set in config
MEM_15="${MEM_15:-$MEM_12}"
TIME_15="${TIME_15:-$TIME_12}"
CPUS_15="${CPUS_15:-$CPUS_12}"

slurm_base() { echo --partition="$PARTITION" --mem="$1" --time="$2" --cpus-per-task="$3"; }

mkdir -p "$OUTPUT_DIR" logs

ARRAY_END_GENE=$((N_GENE_JOBS - 1))
ARRAY_END_SAMPLE=$((N_SAMPLES - 1))

echo "=== LongAllele SLURM submission ==="
echo "Output: $OUTPUT_DIR"
echo "Gene jobs: $N_GENE_JOBS  |  Samples: $N_SAMPLES"
echo ""

# ── Step 1: variant calling (gene array) ─────────────────────────────────────
JID1=$(sbatch $(slurm_base "$MEM_12" "$TIME_12" "$CPUS_12") \
    --array=0-${ARRAY_END_GENE} \
    --job-name=la_step1 \
    --output=logs/la_step1_%A_%a.out \
    --error=logs/la_step1_%A_%a.err \
    --parsable \
    --wrap="python src/longallele.py --task step1 \
        --scotch_target $SCOTCH_TARGET \
        --bam_path $BAM_PATH \
        --ref_fasta_path $REF_FASTA \
        --output_folder $OUTPUT_DIR \
        --n_jobs $N_GENE_JOBS --job_index \$SLURM_ARRAY_TASK_ID \
        $PREFIX_OPT")
echo "Step 1   variant calling    → array job $JID1  (${N_GENE_JOBS} tasks)"

# ── Step 1.5: read-block collection (sample array, parallel with step 2) ──────
JID15=$(sbatch $(slurm_base "$MEM_15" "$TIME_15" "$CPUS_15") \
    --array=0-${ARRAY_END_SAMPLE} \
    --job-name=la_step1_5 \
    --output=logs/la_step1_5_%A_%a.out \
    --error=logs/la_step1_5_%A_%a.err \
    --dependency=afterok:$JID1 \
    --parsable \
    --wrap="python src/longallele.py --task step1_5 \
        --scotch_target $SCOTCH_TARGET \
        --bam_path $BAM_PATH \
        --ref_fasta_path $REF_FASTA \
        --output_folder $OUTPUT_DIR \
        --n_jobs $N_SAMPLES --job_index \$SLURM_ARRAY_TASK_ID \
        $PREFIX_OPT")
echo "Step 1.5 read-block collect → array job $JID15  (${N_SAMPLES} tasks, parallel with step 2)"

# ── Step 2: EM input generation (gene array) ──────────────────────────────────
JID2=$(sbatch $(slurm_base "$MEM_12" "$TIME_12" "$CPUS_12") \
    --array=0-${ARRAY_END_GENE} \
    --job-name=la_step2 \
    --output=logs/la_step2_%A_%a.out \
    --error=logs/la_step2_%A_%a.err \
    --dependency=afterok:$JID1 \
    --parsable \
    --wrap="python src/longallele.py --task step2 \
        --scotch_target $SCOTCH_TARGET \
        --bam_path $BAM_PATH \
        --ref_fasta_path $REF_FASTA \
        --output_folder $OUTPUT_DIR \
        --n_jobs $N_GENE_JOBS --job_index \$SLURM_ARRAY_TASK_ID \
        $PREFIX_OPT")
echo "Step 2   EM input           → array job $JID2  (${N_GENE_JOBS} tasks)"

# ── Step 1.5 merge: union per-sample pkls ─────────────────────────────────────
JID15M=$(sbatch $(slurm_base "$MEM_12" "$TIME_12" "$CPUS_12") \
    --job-name=la_step1_5m \
    --output=logs/la_step1_5m_%j.out \
    --error=logs/la_step1_5m_%j.err \
    --dependency=afterok:$JID15 \
    --parsable \
    --wrap="python src/longallele.py --task step1_5_merge \
        --scotch_target $SCOTCH_TARGET \
        --output_folder $OUTPUT_DIR")
echo "Step 1.5 read-block merge   → job      $JID15M"

# ── Step 3: EM haplotyping (gene array) ───────────────────────────────────────
JID3=$(sbatch $(slurm_base "$MEM_3" "$TIME_3" "$CPUS_3") \
    --array=0-${ARRAY_END_GENE} \
    --job-name=la_step3 \
    --output=logs/la_step3_%A_%a.out \
    --error=logs/la_step3_%A_%a.err \
    --dependency=afterok:$JID2 \
    --parsable \
    --wrap="python src/longallele.py --task step3 \
        --scotch_target $SCOTCH_TARGET \
        --bam_path $BAM_PATH \
        --ref_fasta_path $REF_FASTA \
        --output_folder $OUTPUT_DIR \
        --n_jobs $N_GENE_JOBS --job_index \$SLURM_ARRAY_TASK_ID \
        --clf_init \
        $CELL_OPT $PREFIX_OPT")
echo "Step 3   EM haplotyping     → array job $JID3  (${N_GENE_JOBS} tasks)"

# ── Step 4: summary statistics + count matrix ─────────────────────────────────
if [[ "$N_SAMPLES" -gt 1 ]]; then
    JID4=$(sbatch $(slurm_base "$MEM_4" "$TIME_4" "$CPUS_4") \
        --array=0-${ARRAY_END_SAMPLE} \
        --job-name=la_step4 \
        --output=logs/la_step4_%A_%a.out \
        --error=logs/la_step4_%A_%a.err \
        --dependency=afterok:$JID3 \
        --parsable \
        --wrap="python src/longallele.py --task step4 \
            --scotch_target $SCOTCH_TARGET \
            --output_folder $OUTPUT_DIR \
            --summary_haplotype --summary_count \
            --job_array_by_sample --job_index \$SLURM_ARRAY_TASK_ID \
            $CELL_OPT $PREFIX_OPT")
    echo "Step 4   summary + counts   → array job $JID4  (${N_SAMPLES} tasks)"
else
    JID4=$(sbatch $(slurm_base "$MEM_4" "$TIME_4" "$CPUS_4") \
        --job-name=la_step4 \
        --output=logs/la_step4_%j.out \
        --error=logs/la_step4_%j.err \
        --dependency=afterok:$JID3 \
        --parsable \
        --wrap="python src/longallele.py --task step4 \
            --scotch_target $SCOTCH_TARGET \
            --output_folder $OUTPUT_DIR \
            --summary_haplotype --summary_count \
            $CELL_OPT $PREFIX_OPT")
    echo "Step 4   summary + counts   → job      $JID4"
fi

# ── Step 5: downstream analysis (waits for step 4 AND step 1.5 merge) ─────────
if [[ "$N_SAMPLES" -gt 1 ]]; then
    JID5=$(sbatch $(slurm_base "$MEM_5" "$TIME_5" "$CPUS_5") \
        --array=0-${ARRAY_END_SAMPLE} \
        --job-name=la_step5 \
        --output=logs/la_step5_%A_%a.out \
        --error=logs/la_step5_%A_%a.err \
        --dependency=afterok:$JID4:$JID15M \
        --parsable \
        --wrap="python src/longallele.py --task step5 \
            --scotch_target $SCOTCH_TARGET \
            --bam_path $BAM_PATH \
            --output_folder $OUTPUT_DIR \
            --job_array_by_sample --job_index \$SLURM_ARRAY_TASK_ID \
            $CELL_OPT $PREFIX_OPT")
    echo "Step 5   downstream         → array job $JID5  (${N_SAMPLES} tasks)"
else
    JID5=$(sbatch $(slurm_base "$MEM_5" "$TIME_5" "$CPUS_5") \
        --job-name=la_step5 \
        --output=logs/la_step5_%j.out \
        --error=logs/la_step5_%j.err \
        --dependency=afterok:$JID4:$JID15M \
        --parsable \
        --wrap="python src/longallele.py --task step5 \
            --scotch_target $SCOTCH_TARGET \
            --bam_path $BAM_PATH \
            --output_folder $OUTPUT_DIR \
            $CELL_OPT $PREFIX_OPT")
    echo "Step 5   downstream         → job      $JID5"
fi

echo ""
echo "=== All jobs submitted ==="
echo "step1($JID1) ──┬──→ step1_5($JID15) ──→ step1_5_merge($JID15M) ─┐"
echo "               └──→ step2($JID2) ──→ step3($JID3) ──→ step4($JID4) ─┴──→ step5($JID5)"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     $OUTPUT_DIR/../logs/"
