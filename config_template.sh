# LongAllele configuration
# Copy: cp config_template.sh my_run.sh
# Edit the paths and settings below, then run: bash longallele.sh my_run.sh
# Guide: docs/pipeline_steps.md#configuration-options

# ──── Input files ─────────────────────────────────────────────────────────────
BAM_PATH="/path/to/aligned.bam"          # genome-aligned, indexed BAM
REF_FASTA="/path/to/genome.fa"           # matching reference genome FASTA
OUTPUT_DIR="/path/to/results"            # results directory

# Choose one: scotch = SCOTCH results; isoquant = IsoQuant results;
# light = BAM + FASTA + GTF for SNV calling, phasing and ASE.
INPUT="scotch"
SCOTCH_TARGET="/path/to/scotch_output"   # fill in for INPUT=scotch
ISOQUANT_DIR=""                          # fill in for INPUT=isoquant
ISOQUANT_PREFIX=""                       # your IsoQuant run prefix
GTF=""                                   # reference annotation GTF (e.g. GENCODE); used by every input
                                         # to label SNVs exonic/intronic; required for light or IsoQuant

# ──── Data type and platform ──────────────────────────────────────────────────
BULK=false                               # false = single-cell/single-nucleus; true = bulk
CELL_TYPE_DF=""                          # optional CSV with Cell and CellType columns
PLATFORM="ont-cdna"                      # ont-cdna | ont-drna | hifi-isoseq | hifi-masseq | other
                                         # other uses no SNV classifier

# ──── Where to run ────────────────────────────────────────────────────────────
RUNNER="slurm"   # local = this computer/server; slurm = submit to a SLURM cluster
CORES=8          # local runs: tasks allowed at once; reduce if memory is limited
N_GENE_JOBS=50   # total gene-processing jobs; normally leave at this default

# ──── Optional sample settings ────────────────────────────────────────────────
N_SAMPLES=1              # for light or IsoQuant input, run one sample at a time
SAME_INDIVIDUAL=false    # true = compare multiple bulk samples from one individual
PREFIX=""                # optional output filename label, e.g. sample1
HIGH_ARTIFACT_MODE=false # extra filtering for single-nucleus RNA artifacts; unavailable with IsoQuant

# With SCOTCH input, set N_SAMPLES > 1 to run multiple samples together.
# Supply space-separated BAM_PATH, SCOTCH_TARGET and (if used) CELL_TYPE_DF
# paths in matching sample order. These paths must not contain spaces.

# ──── EM haplotyping (step 3) ─────────────────────────────────────────────────
SEED=42                    # random seed
MAX_ITER=50                # maximum EM iterations per gene
TOL=1e-3                   # convergence tolerance
GAP_TAU=1.0                # marker-selection elbow threshold: 1.0 disables the
                           # elbow rule, 0.10 enables it (used for simulated data)
RNA_EDITING_DB=""          # override the bundled hg38 A-to-I database; leave empty
                           # to use the bundled one, or set to "none" to disable
                           # the RNA-editing filter entirely

# Optional short-read evidence: leave SR_BAM empty to disable.
# Use indexed BAM(s) aligned to the same reference. For multiple samples, give
# space-separated paths in BAM_PATH order (no spaces within paths), or one
# shared BAM. SAME_INDIVIDUAL=true pools their evidence.
SR_BAM=""
SR_MIN_DEPTH=30            # remove a candidate only at this depth or higher
SR_MAX_ALT=1               # and with at most this many non-reference bases
SR_MIN_MAPQ=20             # minimum short-read mapping quality
SR_MIN_BASEQ=20            # minimum short-read base quality

# ──── Downstream analysis (step 5) ───────────────────────────────────────────
EVENT_MODE="all_events"    # all_events | switching_events | fdr_events
SNV_EVENT_DISTANCE=50      # ±bp exonic distance for SNV–event linking
EVENT_MIN_READS=10         # distinct reads: isoform inclusion / raw-read include+skip
ASTU_SIG_ONLY=false        # true restricts step 5 to ASTU-significant genes

# ──── Additional command-line options ─────────────────────────────────────────
EXTRA_OPTS=""              # normally leave empty; see docs/pipeline_steps.md

# ──── SLURM resources (RUNNER=slurm only) ─────────────────────────────────────
PARTITION="cpu"   # replace with a partition name on your cluster

# Memory, time limit (hours:minutes:seconds), and CPUs per job.
# Steps 1, 2 and 1.5
MEM_12="16G"  ; TIME_12="4:00:00"  ; CPUS_12=2
# Step 3: haplotyping
MEM_3="32G"   ; TIME_3="8:00:00"   ; CPUS_3=4
# Step 4: summaries and counts
MEM_4="64G"   ; TIME_4="4:00:00"   ; CPUS_4=8
# Step 5: allelic analyses
MEM_5="64G"   ; TIME_5="6:00:00"   ; CPUS_5=8
