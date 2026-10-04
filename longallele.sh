#!/usr/bin/env bash
# longallele.sh — run the LongAllele pipeline from one config file, either as a
# chain of SLURM jobs (RUNNER=slurm) or in order on this machine (RUNNER=local).
#
# Usage:
#   cp config_template.sh my_run.sh   # fill in paths and settings
#   bash longallele.sh my_run.sh
#
# Step graph (both runners):
#   step1 ──┬──→ step1_5 ──→ step1_5_merge ─┐
#           └──→ step2 ──→ step3 ──→ step4 ─┴──→ step5
#
# Every step's command line is built ONCE (the args_* functions below) and
# handed to whichever runner is selected, so the two ways of running cannot
# drift apart in what they pass to src/longallele.py.

set -euo pipefail

# Resolve the pipeline entry point from this script's own location, so the
# jobs find it no matter which directory the user launches from.
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

# ── Validate ──────────────────────────────────────────────────────────────────
# Every variable here must carry a value: an empty one would silently fall back
# to the argparse default in src/longallele.py, which is exactly the failure
# mode this script is meant to avoid. Legitimately optional variables
# (CELL_TYPE_DF, PREFIX, RNA_EDITING_DB, SR_BAM, EXTRA_OPTS) are handled below.
RUNNER="${RUNNER:-slurm}"
INPUT="${INPUT:-scotch}"
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
[[ "$INPUT" == "isoquant" || "$INPUT" == "light" ]] && SCOTCH_TARGET="${SCOTCH_TARGET:-$INPUT}"   # set for real below
for var in SCOTCH_TARGET BAM_PATH REF_FASTA OUTPUT_DIR N_GENE_JOBS N_SAMPLES \
           PLATFORM SEED MAX_ITER TOL GAP_TAU \
           EVENT_MODE SNV_EVENT_DISTANCE EVENT_MIN_READS; do
    if [[ -z "${!var:-}" ]]; then
        echo "Error: '$var' is not set in $CONFIG"
        exit 1
    fi
done
case "$RUNNER" in
    slurm)
        for var in PARTITION MEM_12 TIME_12 CPUS_12 MEM_3 TIME_3 CPUS_3 \
                   MEM_4 TIME_4 CPUS_4 MEM_5 TIME_5 CPUS_5; do
            if [[ -z "${!var:-}" ]]; then
                echo "Error: '$var' is not set in $CONFIG (needed for RUNNER=slurm)"
                exit 1
            fi
        done
        command -v sbatch >/dev/null 2>&1 || [[ "${DRY_RUN:-0}" == "1" ]] || {
            echo "Error: RUNNER=slurm but 'sbatch' is not on PATH; set RUNNER=local to run here"
            exit 1
        } ;;
    local)
        CORES="${CORES:-1}"
        [[ "$CORES" =~ ^[0-9]+$ && "$CORES" -ge 1 ]] || {
            echo "Error: CORES must be a positive integer in $CONFIG (got: '${CORES:-}')"
            exit 1
        } ;;
    *)  echo "Error: RUNNER must be 'slurm' or 'local' in $CONFIG (got: '$RUNNER')"
        exit 1 ;;
esac
case "$INPUT" in
    scotch) ;;
    isoquant)
        if [[ "$N_SAMPLES" -ne 1 ]]; then
            echo "Error: INPUT=isoquant converts one IsoQuant run = one BAM; set N_SAMPLES=1 (run the driver once per sample)"
            exit 1
        fi
        for var in ISOQUANT_DIR GTF; do
            if [[ -z "${!var:-}" ]]; then
                echo "Error: '$var' is not set in $CONFIG (needed for INPUT=isoquant)"
                exit 1
            fi
        done
        # steps 1-5 read the directory isoquant_prep writes
        SCOTCH_TARGET="$OUTPUT_DIR/isoquant_upstream" ;;
    light)
        if [[ "$N_SAMPLES" -ne 1 ]]; then
            echo "Error: INPUT=light builds one upstream directory per BAM; set N_SAMPLES=1 (run the driver once per sample)"
            exit 1
        fi
        if [[ -z "${GTF:-}" ]]; then
            echo "Error: 'GTF' is not set in $CONFIG (needed for INPUT=light)"
            exit 1
        fi
        # steps 1-4 read the directory light_merge writes; no isoforms => no step 1.5 / step 5
        SCOTCH_TARGET="$OUTPUT_DIR/light_upstream" ;;
    *)  echo "Error: INPUT must be 'scotch', 'isoquant' or 'light' in $CONFIG (got: '$INPUT')"
        exit 1 ;;
esac
check_bool BULK "${BULK:-false}"
BULK_OPT=""
[[ "${BULK:-false}" == "true" ]] && BULK_OPT="--bulk"
# lightweight: placeholder isoforms make the isoform chi-squared meaningless
LIGHT_STEP3_OPT=""
[[ "$INPUT" == "light" ]] && LIGHT_STEP3_OPT="--skip_astu_test"
# reference annotation for the exon/intron SNV labels in step 3 (all inputs)
GTF_OPT=""
if [[ -n "${GTF:-}" ]]; then
    GTF_OPT="--gtf_path \"$GTF\""
else
    echo "Warning: 'GTF' is not set in $CONFIG; step 3 cannot label SNVs exonic/intronic (snv_region_label will be unknown)"
fi
case "$PLATFORM" in
    ont-cdna|ont-drna|hifi-isoseq|hifi-masseq|other) ;;
    *)  echo "Error: PLATFORM must be one of ont-cdna, ont-drna, hifi-isoseq, hifi-masseq, other in $CONFIG (got: '$PLATFORM')"
        exit 1 ;;
esac

check_bool HIGH_ARTIFACT_MODE "${HIGH_ARTIFACT_MODE:-false}"
check_bool ASTU_SIG_ONLY "${ASTU_SIG_ONLY:-false}"
check_bool SAME_INDIVIDUAL "${SAME_INDIVIDUAL:-false}"

if [[ "${SAME_INDIVIDUAL:-false}" == "true" && "$N_SAMPLES" -lt 2 ]]; then
    echo "Error: SAME_INDIVIDUAL=true needs several BAMs of one person (N_SAMPLES >= 2)"
    exit 1
fi

# ── Argument fragments ────────────────────────────────────────────────────────
IO_OPTS="--scotch_target $SCOTCH_TARGET --bam_path $BAM_PATH \
         --ref_fasta_path \"$REF_FASTA\" --output_folder \"$OUTPUT_DIR\""
# The platform preset resolves the calling parameters and the SNV classifier
# for that library (README "Platform presets"); every resolved value is echoed
# in the step logs with its source. 'other' ships no classifier and resolves
# to the clf-free configuration by itself.
CALL_OPTS="--platform $PLATFORM"
SI_OPT=""
[[ "${SAME_INDIVIDUAL:-false}" == "true" ]] && SI_OPT="--same_individual"
# ACTV needs no switch: step 5 turns it on when the run has cell types to
# compare (a cell-type table, or --same_individual's tissues) and off otherwise.
CELL_OPT="${CELL_TYPE_DF:+--cell_type_df_path $CELL_TYPE_DF}"
PREFIX_OPT="${PREFIX:+--prefix \"$PREFIX\"}"
HAF_OPT=""
[[ "${HIGH_ARTIFACT_MODE:-false}" == "true" ]] && HAF_OPT="--high_artifact_mode"
ASTU_OPT=""
[[ "${ASTU_SIG_ONLY:-false}" == "true" ]] && ASTU_OPT="--astu_sig_only"
EM_OPTS="--seed $SEED --max_iter $MAX_ITER --tol $TOL --gap_tau $GAP_TAU"
RNA_OPT="${RNA_EDITING_DB:+--rna_editing_db \"$RNA_EDITING_DB\"}"
SR_OPT=""
if [[ -n "${SR_BAM:-}" ]]; then
    read -r -a SR_BAMS <<< "$SR_BAM"
    if [[ "${#SR_BAMS[@]}" -ne 1 && "${#SR_BAMS[@]}" -ne "$N_SAMPLES" ]]; then
        echo "Error: SR_BAM must contain one shared BAM or N_SAMPLES=$N_SAMPLES paths in BAM_PATH order"
        exit 1
    fi
    SR_MIN_DEPTH="${SR_MIN_DEPTH:-30}"
    SR_MAX_ALT="${SR_MAX_ALT:-1}"
    SR_MIN_MAPQ="${SR_MIN_MAPQ:-20}"
    SR_MIN_BASEQ="${SR_MIN_BASEQ:-20}"
    for var in SR_MIN_DEPTH SR_MAX_ALT SR_MIN_MAPQ SR_MIN_BASEQ; do
        if [[ ! "${!var}" =~ ^[0-9]+$ ]]; then
            echo "Error: $var must be a non-negative integer in $CONFIG"
            exit 1
        fi
    done
    if [[ ! "$SR_MIN_DEPTH" =~ [1-9] ]]; then
        echo "Error: SR_MIN_DEPTH must be at least 1 in $CONFIG"
        exit 1
    fi
    # Quote each path for both the local eval and SLURM's --wrap shell.
    printf -v SR_OPT ' %q' "${SR_BAMS[@]}"
    SR_OPT="--sr_bam$SR_OPT --sr_min_depth $SR_MIN_DEPTH --sr_max_alt $SR_MAX_ALT --sr_min_mapq $SR_MIN_MAPQ --sr_min_baseq $SR_MIN_BASEQ"
fi
EXTRA="${EXTRA_OPTS:-}"
# Everything that is common to every step.
COMMON="$CALL_OPTS $SI_OPT $PREFIX_OPT $EXTRA"

# Per-BAM step 1.5 fans out over samples; one pooled person is one job.
N_15=$N_SAMPLES
[[ "${SAME_INDIVIDUAL:-false}" == "true" ]] && N_15=1
# Steps 4 and 5 run one job per sample only for N independent samples.
PER_SAMPLE_45=0
[[ "$N_SAMPLES" -gt 1 && "${SAME_INDIVIDUAL:-false}" != "true" ]] && PER_SAMPLE_45=1
if [[ "$RUNNER" == "local" ]]; then N_WORKERS=$CORES; else N_WORKERS=$CPUS_5; fi
DS_OPTS="--event_mode $EVENT_MODE --snv_event_distance $SNV_EVENT_DISTANCE \
         --event_min_reads $EVENT_MIN_READS --n_workers $N_WORKERS $ASTU_OPT"

# ── One command line per step (shared by both runners) ────────────────────────
# $1 = shard / sample index: a number under RUNNER=local, the literal
# '$SLURM_ARRAY_TASK_ID' (left for the job to expand) under RUNNER=slurm.
args_isoquant_prep(){ echo "--task isoquant_prep --isoquant_dir \"$ISOQUANT_DIR\" ${ISOQUANT_PREFIX:+--isoquant_prefix \"$ISOQUANT_PREFIX\"} --gtf_path \"$GTF\" --bam_path $BAM_PATH --output_folder \"$OUTPUT_DIR\" $BULK_OPT $PREFIX_OPT $EXTRA"; }
args_light_prep()   { echo "--task light_prep --gtf_path \"$GTF\" --bam_path $BAM_PATH --output_folder \"$OUTPUT_DIR\" --n_jobs $N_GENE_JOBS --job_index $1 $BULK_OPT $COMMON"; }
args_light_merge()  { echo "--task light_merge --gtf_path \"$GTF\" --output_folder \"$OUTPUT_DIR\" --n_jobs $N_GENE_JOBS $BULK_OPT $COMMON"; }
args_step1()        { echo "--task step1 $IO_OPTS --n_jobs $N_GENE_JOBS --job_index $1 $COMMON"; }
args_step1_5()      { echo "--task step1_5 $IO_OPTS --n_jobs $N_15 --job_index $1 $COMMON"; }
args_step1_5_merge(){ echo "--task step1_5_merge $IO_OPTS $COMMON"; }
args_step2()        { echo "--task step2 $IO_OPTS --n_jobs $N_GENE_JOBS --job_index $1 $COMMON"; }
args_step3()        { echo "--task step3 $IO_OPTS --n_jobs $N_GENE_JOBS --job_index $1 $COMMON $EM_OPTS $RNA_OPT $SR_OPT $HAF_OPT $CELL_OPT $LIGHT_STEP3_OPT $GTF_OPT"; }
args_step4()        { local s=""; [[ "$PER_SAMPLE_45" == 1 ]] && s="--job_array_by_sample --job_index $1"
                      echo "--task step4 --scotch_target $SCOTCH_TARGET --output_folder \"$OUTPUT_DIR\" $s $CELL_OPT $COMMON"; }
args_step5()        { local s=""; [[ "$PER_SAMPLE_45" == 1 ]] && s="--job_array_by_sample --job_index $1"
                      echo "--task step5 --scotch_target $SCOTCH_TARGET --bam_path $BAM_PATH --output_folder \"$OUTPUT_DIR\" $s $DS_OPTS $CELL_OPT $COMMON"; }

mkdir -p "$OUTPUT_DIR" logs

echo "=== LongAllele ($RUNNER) ==="
echo "Output: $OUTPUT_DIR"
echo "Input: $INPUT  |  Platform: $PLATFORM  |  Gene shards: $N_GENE_JOBS  |  Samples: $N_SAMPLES$( [[ -n "$SI_OPT" ]] && echo ' (one individual)')"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# RUNNER=slurm — dependency-chained job graph
# ═════════════════════════════════════════════════════════════════════════════
run_slurm() {
    # DRY_RUN=1 prints each sbatch invocation instead of submitting it, so the
    # generated command lines can be checked without a scheduler.
    local DRY_RUN="${DRY_RUN:-0}"
    submit() {
        if [[ "$DRY_RUN" == "1" ]]; then
            # %q keeps each argument's boundaries intact, so the printed line can
            # be pasted back into a shell and reproduces this exact submission.
            printf 'sbatch' >&2; printf ' %q' "$@" >&2; printf '\n' >&2
            echo "000000"
        else
            sbatch "$@"
        fi
    }
    slurm_base() { echo --partition="$PARTITION" --mem="$1" --time="$2" --cpus-per-task="$3"; }
    local MEM_15="${MEM_15:-$MEM_12}" TIME_15="${TIME_15:-$TIME_12}" CPUS_15="${CPUS_15:-$CPUS_12}"
    local IDX='$SLURM_ARRAY_TASK_ID'
    local JID0 JID1 JID15 JID2 JID15M JID3 JID4 JID5 DEP0=""

    if [[ "$INPUT" == "isoquant" ]]; then
        JID0=$(submit $(slurm_base "$MEM_4" "$TIME_12" "$CPUS_12") \
            --job-name=la_isoquant_prep --output=logs/la_isoquant_prep_%j.out --error=logs/la_isoquant_prep_%j.err \
            --parsable --wrap="python \"$LA_PY\" $(args_isoquant_prep)")
        echo "Step 0   IsoQuant -> upstream  → job      $JID0"
        DEP0="--dependency=afterok:$JID0"
    elif [[ "$INPUT" == "light" ]]; then
        local JIDLP
        JIDLP=$(submit $(slurm_base "$MEM_12" "$TIME_12" "$CPUS_12") --array=0-$((N_GENE_JOBS-1)) \
            --job-name=la_light_prep --output=logs/la_light_prep_%A_%a.out --error=logs/la_light_prep_%A_%a.err \
            --parsable --wrap="python \"$LA_PY\" $(args_light_prep "$IDX")")
        echo "Step 0a  read->gene (GTF)     → array job $JIDLP  ($N_GENE_JOBS tasks)"
        JID0=$(submit $(slurm_base "$MEM_4" "$TIME_12" "$CPUS_12") \
            --job-name=la_light_merge --output=logs/la_light_merge_%j.out --error=logs/la_light_merge_%j.err \
            --dependency=afterok:$JIDLP --parsable --wrap="python \"$LA_PY\" $(args_light_merge)")
        echo "Step 0b  merge -> upstream     → job      $JID0"
        DEP0="--dependency=afterok:$JID0"
    fi

    JID1=$(submit $(slurm_base "$MEM_12" "$TIME_12" "$CPUS_12") --array=0-$((N_GENE_JOBS-1)) \
        --job-name=la_step1 --output=logs/la_step1_%A_%a.out --error=logs/la_step1_%A_%a.err --parsable $DEP0 \
        --wrap="python \"$LA_PY\" $(args_step1 "$IDX")")
    echo "Step 1   variant calling    → array job $JID1  ($N_GENE_JOBS tasks)"

    if [[ "$INPUT" != "light" ]]; then
    JID15=$(submit $(slurm_base "$MEM_15" "$TIME_15" "$CPUS_15") --array=0-$((N_15-1)) \
        --job-name=la_step1_5 --output=logs/la_step1_5_%A_%a.out --error=logs/la_step1_5_%A_%a.err \
        --dependency=afterok:$JID1 --parsable \
        --wrap="python \"$LA_PY\" $(args_step1_5 "$IDX")")
    echo "Step 1.5 read-block collect → array job $JID15  ($N_15 tasks, parallel with step 2)"
    fi

    JID2=$(submit $(slurm_base "$MEM_12" "$TIME_12" "$CPUS_12") --array=0-$((N_GENE_JOBS-1)) \
        --job-name=la_step2 --output=logs/la_step2_%A_%a.out --error=logs/la_step2_%A_%a.err \
        --dependency=afterok:$JID1 --parsable \
        --wrap="python \"$LA_PY\" $(args_step2 "$IDX")")
    echo "Step 2   EM input           → array job $JID2  ($N_GENE_JOBS tasks)"

    if [[ "$INPUT" != "light" ]]; then
    JID15M=$(submit $(slurm_base "$MEM_12" "$TIME_12" "$CPUS_12") \
        --job-name=la_step1_5m --output=logs/la_step1_5m_%j.out --error=logs/la_step1_5m_%j.err \
        --dependency=afterok:$JID15 --parsable \
        --wrap="python \"$LA_PY\" $(args_step1_5_merge)")
    echo "Step 1.5 read-block merge   → job      $JID15M"
    fi

    JID3=$(submit $(slurm_base "$MEM_3" "$TIME_3" "$CPUS_3") --array=0-$((N_GENE_JOBS-1)) \
        --job-name=la_step3 --output=logs/la_step3_%A_%a.out --error=logs/la_step3_%A_%a.err \
        --dependency=afterok:$JID2 --parsable \
        --wrap="python \"$LA_PY\" $(args_step3 "$IDX")")
    echo "Step 3   EM haplotyping     → array job $JID3  ($N_GENE_JOBS tasks)"

    if [[ "$INPUT" == "light" ]]; then
        JID4=$(submit $(slurm_base "$MEM_4" "$TIME_4" "$CPUS_4") \
            --job-name=la_step4 --output=logs/la_step4_%j.out --error=logs/la_step4_%j.err \
            --dependency=afterok:$JID3 --parsable --wrap="python \"$LA_PY\" $(args_step4 0)")
        echo "Step 4   summary + counts   → job      $JID4"
        echo "(lightweight: no step 1.5 / step 5 — no isoform information)"
    elif [[ "$PER_SAMPLE_45" == 1 ]]; then
        JID4=$(submit $(slurm_base "$MEM_4" "$TIME_4" "$CPUS_4") --array=0-$((N_SAMPLES-1)) \
            --job-name=la_step4 --output=logs/la_step4_%A_%a.out --error=logs/la_step4_%A_%a.err \
            --dependency=afterok:$JID3 --parsable --wrap="python \"$LA_PY\" $(args_step4 "$IDX")")
        echo "Step 4   summary + counts   → array job $JID4  ($N_SAMPLES tasks)"
        JID5=$(submit $(slurm_base "$MEM_5" "$TIME_5" "$CPUS_5") --array=0-$((N_SAMPLES-1)) \
            --job-name=la_step5 --output=logs/la_step5_%A_%a.out --error=logs/la_step5_%A_%a.err \
            --dependency=afterok:$JID4:$JID15M --parsable --wrap="python \"$LA_PY\" $(args_step5 "$IDX")")
        echo "Step 5   downstream         → array job $JID5  ($N_SAMPLES tasks)"
    else
        JID4=$(submit $(slurm_base "$MEM_4" "$TIME_4" "$CPUS_4") \
            --job-name=la_step4 --output=logs/la_step4_%j.out --error=logs/la_step4_%j.err \
            --dependency=afterok:$JID3 --parsable --wrap="python \"$LA_PY\" $(args_step4 0)")
        echo "Step 4   summary + counts   → job      $JID4"
        JID5=$(submit $(slurm_base "$MEM_5" "$TIME_5" "$CPUS_5") \
            --job-name=la_step5 --output=logs/la_step5_%j.out --error=logs/la_step5_%j.err \
            --dependency=afterok:$JID4:$JID15M --parsable --wrap="python \"$LA_PY\" $(args_step5 0)")
        echo "Step 5   downstream         → job      $JID5"
    fi

    echo ""
    echo "=== All jobs submitted ==="
    echo "Monitor:  squeue -u \$USER"
    echo "Logs:     logs/"
}

# ═════════════════════════════════════════════════════════════════════════════
# RUNNER=local — same steps, same order, on this machine
# ═════════════════════════════════════════════════════════════════════════════
# run_shards NAME N FN: run FN(0..N-1) as separate python processes, at most
# CORES at a time; each shard logs to logs/NAME_<i>.log. Any failing shard
# fails the step, and a failed step stops the pipeline — a later step would
# otherwise silently run on a partial upstream.
run_shards() {
    local name=$1 n=$2 fn=$3 i failed=0
    local -a pids=() idxs=()
    for ((i = 0; i < n; i++)); do
        while (( $(jobs -rp | wc -l) >= CORES )); do sleep 1; done
        eval "python \"$LA_PY\" $($fn "$i")" > "logs/${name}_${i}.log" 2>&1 &
        pids+=($!); idxs+=("$i")
    done
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "  ✗ $name shard ${idxs[$i]} failed — see logs/${name}_${idxs[$i]}.log"
            failed=1
        fi
    done
    if [[ "$failed" == 1 ]]; then
        echo "Error: $name did not finish; stopping before the next step."
        exit 1
    fi
}

run_local() {
    local t0=$SECONDS
    if [[ "$INPUT" == "isoquant" ]]; then
        echo "Step 0   IsoQuant -> upstream"
        run_shards isoquant_prep 1 args_isoquant_prep
    elif [[ "$INPUT" == "light" ]]; then
        echo "Step 0a  read->gene (GTF)     ($N_GENE_JOBS shards)"
        run_shards light_prep "$N_GENE_JOBS" args_light_prep
        echo "Step 0b  merge -> upstream"
        run_shards light_merge 1 args_light_merge
    fi
    echo "Step 1   variant calling    ($N_GENE_JOBS shards, $CORES at a time)"
    run_shards step1 "$N_GENE_JOBS" args_step1
    if [[ "$INPUT" != "light" ]]; then
        echo "Step 1.5 read-block collect ($N_15 shards)"
        run_shards step1_5 "$N_15" args_step1_5
        echo "Step 1.5 read-block merge"
        run_shards step1_5_merge 1 args_step1_5_merge
    fi
    echo "Step 2   EM input           ($N_GENE_JOBS shards)"
    run_shards step2 "$N_GENE_JOBS" args_step2
    echo "Step 3   EM haplotyping     ($N_GENE_JOBS shards)"
    run_shards step3 "$N_GENE_JOBS" args_step3
    if [[ "$INPUT" == "light" ]]; then
        echo "Step 4   summary + counts"
        run_shards step4 1 args_step4
        echo "(lightweight: no step 1.5 / step 5 — no isoform information)"
    elif [[ "$PER_SAMPLE_45" == 1 ]]; then
        echo "Step 4   summary + counts   ($N_SAMPLES samples)"
        run_shards step4 "$N_SAMPLES" args_step4
        echo "Step 5   downstream         ($N_SAMPLES samples)"
        run_shards step5 "$N_SAMPLES" args_step5
    else
        echo "Step 4   summary + counts"
        run_shards step4 1 args_step4
        echo "Step 5   downstream         ($N_WORKERS workers)"
        run_shards step5 1 args_step5
    fi
    echo ""
    echo "=== Done in $((SECONDS - t0)) s — results in $OUTPUT_DIR, logs in logs/ ==="
}

if [[ "$RUNNER" == "slurm" ]]; then run_slurm; else run_local; fi
