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

# Resolve the pipeline entry point from this script's own location, so the
# submitted jobs find it no matter which directory the user launches from.
# (logs/ is still created relative to the launch directory, on purpose.)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LA_PY="$REPO_ROOT/src/longallele.py"
if [[ ! -f "$LA_PY" ]]; then
    echo "Error: cannot find src/longallele.py next to longallele.sh (looked in $REPO_ROOT)"
    exit 1
fi

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG="${1:-}"
if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
    echo "Usage: bash longallele.sh <config.sh>"
    exit 1
fi
source "$CONFIG"

# ── Validate required variables ───────────────────────────────────────────────
# Every variable here must carry a value: an empty one would silently fall back
# to the argparse default in src/longallele.py, which is exactly the failure
# mode this script is meant to avoid. Variables that are legitimately optional
# (CELL_TYPE_DF, PREFIX, SNV_CLASSIFIER, RNA_EDITING_DB) are handled below.
for var in SCOTCH_TARGET BAM_PATH REF_FASTA OUTPUT_DIR N_GENE_JOBS N_SAMPLES \
           DEPTH N_ALT_COUNT MIN_MAPQ MIN_BASEQ MIN_DIST_TO_END HET_FILTER \
           SEED MAX_ITER TOL GAP_TAU \
           EVENT_MODE SNV_EVENT_DISTANCE EVENT_MIN_READS N_WORKERS \
           PARTITION MEM_12 TIME_12 CPUS_12 MEM_3 TIME_3 CPUS_3 MEM_4 TIME_4 CPUS_4 \
           MEM_5 TIME_5 CPUS_5; do
    if [[ -z "${!var:-}" ]]; then
        echo "Error: '$var' is not set in $CONFIG"
        exit 1
    fi
done

# ── Build argument fragments ──────────────────────────────────────────────────
CELL_OPT="${CELL_TYPE_DF:+--cell_type_df_path $CELL_TYPE_DF}"
PREFIX_OPT="${PREFIX:+--prefix \"$PREFIX\"}"

# Booleans arrive from the config as the strings "true"/"false". They must be
# compared explicitly — "${VAR:+--flag}" would expand for "false" too, since a
# non-empty string is a non-empty string either way. Anything that is neither
# spelling is rejected rather than quietly treated as false.
check_bool() {
    if [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "Error: '$1' must be exactly 'true' or 'false' in $CONFIG (got: '$2')"
        exit 1
    fi
}
check_bool HIGH_ARTIFACT_MODE "${HIGH_ARTIFACT_MODE:-false}"
check_bool ASTU_SIG_ONLY "${ASTU_SIG_ONLY:-false}"

HAF_OPT=""
[[ "${HIGH_ARTIFACT_MODE:-false}" == "true" ]] && HAF_OPT="--high_artifact_mode"
ASTU_OPT=""
[[ "${ASTU_SIG_ONLY:-false}" == "true" ]] && ASTU_OPT="--astu_sig_only"

# Variant calling / SNV filtering, applied to steps 1, 1.5, 2 and 3 — the same
# grouping used by examples/pbmc_demo/run_demo.sh.
CALL_OPTS="--depth $DEPTH --n_alt_count $N_ALT_COUNT --min_mapq $MIN_MAPQ \
           --min_baseq $MIN_BASEQ --min_dist_to_end $MIN_DIST_TO_END \
           --heterozygous_filter $HET_FILTER"

# SNV classifier. --clf_init only takes effect once a classifier is loaded, so
# the two flags are emitted as a pair or not at all; an empty SNV_CLASSIFIER
# disables the classifier entirely.
if [[ -n "${SNV_CLASSIFIER:-}" ]]; then
    # The shipped default is repo-relative, so fall back to the repo copy when
    # the path does not resolve against the launch directory.
    if [[ "$SNV_CLASSIFIER" != /* && ! -f "$SNV_CLASSIFIER" && -f "$REPO_ROOT/$SNV_CLASSIFIER" ]]; then
        SNV_CLASSIFIER="$REPO_ROOT/$SNV_CLASSIFIER"
    fi
    if [[ ! -f "$SNV_CLASSIFIER" ]]; then
        echo "Error: SNV_CLASSIFIER is set but no file exists at: $SNV_CLASSIFIER"
        echo "       Paths are resolved from the directory you launch longallele.sh in."
        echo "       Set SNV_CLASSIFIER=\"\" in $CONFIG to run without the classifier."
        exit 1
    fi
    CLF_OPTS="--snv_classifier \"$SNV_CLASSIFIER\" --clf_init"
else
    CLF_OPTS=""
fi

# EM settings (step 3 only)
EM_OPTS="--seed $SEED --max_iter $MAX_ITER --tol $TOL --gap_tau $GAP_TAU"
RNA_OPT="${RNA_EDITING_DB:+--rna_editing_db \"$RNA_EDITING_DB\"}"

# Downstream settings (step 5 only)
DS_OPTS="--event_mode $EVENT_MODE --snv_event_distance $SNV_EVENT_DISTANCE \
         --event_min_reads $EVENT_MIN_READS --n_workers $N_WORKERS $ASTU_OPT"

# DRY_RUN=1 prints each sbatch invocation instead of submitting it, so the
# generated command lines can be checked without a scheduler.
DRY_RUN="${DRY_RUN:-0}"
submit() {
    if [[ "$DRY_RUN" == "1" ]]; then
        # %q keeps each argument's boundaries intact, so the printed line can be
        # pasted back into a shell and reproduces this exact submission.
        printf 'sbatch' >&2
        printf ' %q' "$@" >&2
        printf '\n' >&2
        echo "000000"
    else
        sbatch "$@"
    fi
}

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
JID1=$(submit $(slurm_base "$MEM_12" "$TIME_12" "$CPUS_12") \
    --array=0-${ARRAY_END_GENE} \
    --job-name=la_step1 \
    --output=logs/la_step1_%A_%a.out \
    --error=logs/la_step1_%A_%a.err \
    --parsable \
    --wrap="python \"$LA_PY\" --task step1 \
        --scotch_target $SCOTCH_TARGET \
        --bam_path $BAM_PATH \
        --ref_fasta_path \"$REF_FASTA\" \
        --output_folder \"$OUTPUT_DIR\" \
        --n_jobs $N_GENE_JOBS --job_index \$SLURM_ARRAY_TASK_ID \
        $CALL_OPTS $PREFIX_OPT")
echo "Step 1   variant calling    → array job $JID1  (${N_GENE_JOBS} tasks)"

# ── Step 1.5: read-block collection (sample array, parallel with step 2) ──────
JID15=$(submit $(slurm_base "$MEM_15" "$TIME_15" "$CPUS_15") \
    --array=0-${ARRAY_END_SAMPLE} \
    --job-name=la_step1_5 \
    --output=logs/la_step1_5_%A_%a.out \
    --error=logs/la_step1_5_%A_%a.err \
    --dependency=afterok:$JID1 \
    --parsable \
    --wrap="python \"$LA_PY\" --task step1_5 \
        --scotch_target $SCOTCH_TARGET \
        --bam_path $BAM_PATH \
        --ref_fasta_path \"$REF_FASTA\" \
        --output_folder \"$OUTPUT_DIR\" \
        --n_jobs $N_SAMPLES --job_index \$SLURM_ARRAY_TASK_ID \
        $CALL_OPTS $PREFIX_OPT")
echo "Step 1.5 read-block collect → array job $JID15  (${N_SAMPLES} tasks, parallel with step 2)"

# ── Step 2: EM input generation (gene array) ──────────────────────────────────
JID2=$(submit $(slurm_base "$MEM_12" "$TIME_12" "$CPUS_12") \
    --array=0-${ARRAY_END_GENE} \
    --job-name=la_step2 \
    --output=logs/la_step2_%A_%a.out \
    --error=logs/la_step2_%A_%a.err \
    --dependency=afterok:$JID1 \
    --parsable \
    --wrap="python \"$LA_PY\" --task step2 \
        --scotch_target $SCOTCH_TARGET \
        --bam_path $BAM_PATH \
        --ref_fasta_path \"$REF_FASTA\" \
        --output_folder \"$OUTPUT_DIR\" \
        --n_jobs $N_GENE_JOBS --job_index \$SLURM_ARRAY_TASK_ID \
        $CALL_OPTS $PREFIX_OPT")
echo "Step 2   EM input           → array job $JID2  (${N_GENE_JOBS} tasks)"

# ── Step 1.5 merge: union per-sample pkls ─────────────────────────────────────
JID15M=$(submit $(slurm_base "$MEM_12" "$TIME_12" "$CPUS_12") \
    --job-name=la_step1_5m \
    --output=logs/la_step1_5m_%j.out \
    --error=logs/la_step1_5m_%j.err \
    --dependency=afterok:$JID15 \
    --parsable \
    --wrap="python \"$LA_PY\" --task step1_5_merge \
        --scotch_target $SCOTCH_TARGET \
        --output_folder \"$OUTPUT_DIR\"")
echo "Step 1.5 read-block merge   → job      $JID15M"

# ── Step 3: EM haplotyping (gene array) ───────────────────────────────────────
JID3=$(submit $(slurm_base "$MEM_3" "$TIME_3" "$CPUS_3") \
    --array=0-${ARRAY_END_GENE} \
    --job-name=la_step3 \
    --output=logs/la_step3_%A_%a.out \
    --error=logs/la_step3_%A_%a.err \
    --dependency=afterok:$JID2 \
    --parsable \
    --wrap="python \"$LA_PY\" --task step3 \
        --scotch_target $SCOTCH_TARGET \
        --bam_path $BAM_PATH \
        --ref_fasta_path \"$REF_FASTA\" \
        --output_folder \"$OUTPUT_DIR\" \
        --n_jobs $N_GENE_JOBS --job_index \$SLURM_ARRAY_TASK_ID \
        $CALL_OPTS $EM_OPTS $CLF_OPTS $RNA_OPT $HAF_OPT \
        $CELL_OPT $PREFIX_OPT")
echo "Step 3   EM haplotyping     → array job $JID3  (${N_GENE_JOBS} tasks)"

# ── Step 4: summary statistics + count matrix ─────────────────────────────────
if [[ "$N_SAMPLES" -gt 1 ]]; then
    JID4=$(submit $(slurm_base "$MEM_4" "$TIME_4" "$CPUS_4") \
        --array=0-${ARRAY_END_SAMPLE} \
        --job-name=la_step4 \
        --output=logs/la_step4_%A_%a.out \
        --error=logs/la_step4_%A_%a.err \
        --dependency=afterok:$JID3 \
        --parsable \
        --wrap="python \"$LA_PY\" --task step4 \
            --scotch_target $SCOTCH_TARGET \
            --output_folder \"$OUTPUT_DIR\" \
            --summary_haplotype --summary_count \
            --job_array_by_sample --job_index \$SLURM_ARRAY_TASK_ID \
            $CELL_OPT $PREFIX_OPT")
    echo "Step 4   summary + counts   → array job $JID4  (${N_SAMPLES} tasks)"
else
    JID4=$(submit $(slurm_base "$MEM_4" "$TIME_4" "$CPUS_4") \
        --job-name=la_step4 \
        --output=logs/la_step4_%j.out \
        --error=logs/la_step4_%j.err \
        --dependency=afterok:$JID3 \
        --parsable \
        --wrap="python \"$LA_PY\" --task step4 \
            --scotch_target $SCOTCH_TARGET \
            --output_folder \"$OUTPUT_DIR\" \
            --summary_haplotype --summary_count \
            $CELL_OPT $PREFIX_OPT")
    echo "Step 4   summary + counts   → job      $JID4"
fi

# ── Step 5: downstream analysis (waits for step 4 AND step 1.5 merge) ─────────
if [[ "$N_SAMPLES" -gt 1 ]]; then
    JID5=$(submit $(slurm_base "$MEM_5" "$TIME_5" "$CPUS_5") \
        --array=0-${ARRAY_END_SAMPLE} \
        --job-name=la_step5 \
        --output=logs/la_step5_%A_%a.out \
        --error=logs/la_step5_%A_%a.err \
        --dependency=afterok:$JID4:$JID15M \
        --parsable \
        --wrap="python \"$LA_PY\" --task step5 \
            --scotch_target $SCOTCH_TARGET \
            --bam_path $BAM_PATH \
            --output_folder \"$OUTPUT_DIR\" \
            --job_array_by_sample --job_index \$SLURM_ARRAY_TASK_ID \
            $DS_OPTS $CELL_OPT $PREFIX_OPT")
    echo "Step 5   downstream         → array job $JID5  (${N_SAMPLES} tasks)"
else
    JID5=$(submit $(slurm_base "$MEM_5" "$TIME_5" "$CPUS_5") \
        --job-name=la_step5 \
        --output=logs/la_step5_%j.out \
        --error=logs/la_step5_%j.err \
        --dependency=afterok:$JID4:$JID15M \
        --parsable \
        --wrap="python \"$LA_PY\" --task step5 \
            --scotch_target $SCOTCH_TARGET \
            --bam_path $BAM_PATH \
            --output_folder \"$OUTPUT_DIR\" \
            $DS_OPTS $CELL_OPT $PREFIX_OPT")
    echo "Step 5   downstream         → job      $JID5"
fi

echo ""
echo "=== All jobs submitted ==="
echo "Monitor:  squeue -u \$USER"
echo "Logs:     logs/"
