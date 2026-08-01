# LongAllele pipeline configuration
# Usage: bash longallele.sh config_template.sh
# Copy this file, fill in your paths and settings, then run.
#
# Every setting below is forwarded to src/longallele.py by longallele.sh.
# If you need a parameter that is not listed here, run the steps individually
# (see the per-step sections of README.md) — longallele.sh only forwards what
# it finds in this file.

# ──── Required ────────────────────────────────────────────────────────────────
# SCOTCH_TARGET, BAM_PATH and CELL_TYPE_DF may hold several space-separated
# paths for multi-sample analysis. Because the list is split by the shell, an
# individual path in those three must not contain spaces or wildcards. The
# single-valued settings (REF_FASTA, OUTPUT_DIR, PREFIX, SNV_CLASSIFIER,
# RNA_EDITING_DB) are quoted for you and may contain spaces.
SCOTCH_TARGET="/path/to/scotch_output"   # SCOTCH output directory
BAM_PATH="/path/to/aligned.bam"          # aligned BAM file
REF_FASTA="/path/to/genome.fa"           # reference genome FASTA
OUTPUT_DIR="/path/to/results"            # pipeline output directory

# ──── Parallelization ─────────────────────────────────────────────────────────
N_GENE_JOBS=50   # SLURM array size for steps 1–3 (number of gene-parallel tasks)
N_SAMPLES=1      # number of BAM files; set >1 and use space-separated lists above
                 # for multi-sample analysis

# ──── Optional inputs ─────────────────────────────────────────────────────────
CELL_TYPE_DF=""  # path to CSV with Cell/CellType columns; leave empty if not needed,
                 # or give one space-separated path per sample for multi-sample runs
PREFIX=""        # output filename prefix (leave empty for none)

# ──── Variant calling / SNV filtering (steps 1, 1.5, 2, 3) ───────────────────
DEPTH=20            # minimum read depth at SNV position
N_ALT_COUNT=10      # minimum alt-allele read count
MIN_MAPQ=20         # minimum mapping quality
MIN_BASEQ=5         # minimum base quality
MIN_DIST_TO_END=3   # minimum distance from read end
HET_FILTER=0.99     # heterozygosity probability threshold

# ──── SNV classifier (step 3) ─────────────────────────────────────────────────
# The bundled classifier scores every SNV candidate on how likely it is to be a
# real variant rather than a basecalling artifact. When a path is set, those
# scores are used both to drop confident artifacts before the EM and to
# initialize the EM marker probabilities (longallele.sh always sends
# --snv_classifier and --clf_init together, because --clf_init on its own has
# no effect).
#
# To turn the classifier off completely, set SNV_CLASSIFIER="" (or delete the line).
#   - ONT data ........ keep as is (the bundled model was trained on ONT GIAB HG001)
#   - PacBio data ..... set to "" — the model has not been validated on PacBio
#   - simulated data .. set to "" and set GAP_TAU=0.10 below
SNV_CLASSIFIER="src/models/snv_classifier_ont_hg001_17feat.joblib"

# ──── EM haplotyping (step 3) ─────────────────────────────────────────────────
SEED=42                    # random seed
MAX_ITER=50                # maximum EM iterations per gene
TOL=1e-3                   # convergence tolerance
GAP_TAU=1.0                # marker-selection elbow threshold: 1.0 disables the
                           # elbow rule, 0.10 enables it (used for simulated data)
RNA_EDITING_DB=""          # override the bundled hg38 A-to-I database; leave empty
                           # to use the bundled one, or set to "none" to disable
                           # the RNA-editing filter entirely
HIGH_ARTIFACT_MODE=false   # true enables the nascent-RNA leak filters for snRNA-seq

# ──── Downstream analysis (step 5) ───────────────────────────────────────────
EVENT_MODE="all_events"    # all_events | switching_events | fdr_events
SNV_EVENT_DISTANCE=50      # ±bp exonic distance for SNV–event linking
EVENT_MIN_READS=10         # minimum weighted reads per event test
N_WORKERS=4                # parallel workers for step 5
ASTU_SIG_ONLY=false        # true restricts step 5 to ASTU-significant genes

# ──── SLURM resources ─────────────────────────────────────────────────────────
PARTITION="cpu"   # shared across all steps

# Steps 1, 2, 1.5: lightweight per-gene / per-sample pileup
MEM_12="16G"  ; TIME_12="4:00:00"  ; CPUS_12=2
# Step 3: EM haplotyping — more memory for large genes
MEM_3="32G"   ; TIME_3="8:00:00"   ; CPUS_3=4
# Step 4: aggregation across all genes — single job, high memory
MEM_4="64G"   ; TIME_4="4:00:00"   ; CPUS_4=8
# Step 5: downstream analysis — parallel workers, high memory
MEM_5="64G"   ; TIME_5="6:00:00"   ; CPUS_5=8
