# LongAllele pipeline configuration
# Usage: bash longallele.sh config_template.sh
# Copy this file, fill in your paths and settings, then run.
#
# Every setting below is forwarded to src/longallele.py by longallele.sh. For a
# parameter that has no line here, use EXTRA_OPTS at the bottom — it is passed
# verbatim to every step (an explicit flag always wins over the preset).

# ──── Required ────────────────────────────────────────────────────────────────
# SCOTCH_TARGET, BAM_PATH and CELL_TYPE_DF may hold several space-separated
# paths for multi-sample analysis. Because the list is split by the shell, an
# individual path in those three must not contain spaces or wildcards. The
# single-valued settings (REF_FASTA, OUTPUT_DIR, PREFIX, RNA_EDITING_DB) are
# quoted for you and may contain spaces.
INPUT="scotch"                           # scotch   = SCOTCH_TARGET below is a SCOTCH output directory
                                         # isoquant = build the upstream from an IsoQuant run first
                                         #            (ISOQUANT_DIR / ISOQUANT_PREFIX / GTF below)
                                         # light    = no isoform tool at all: build the upstream from
                                         #            the BAM + GTF alone (variant calling + phasing +
                                         #            ASE; no ASTU, no step 1.5 / step 5)
                                         # with isoquant / light, SCOTCH_TARGET is ignored
SCOTCH_TARGET="/path/to/scotch_output"   # SCOTCH output directory (INPUT=scotch)
ISOQUANT_DIR=""                          # INPUT=isoquant: IsoQuant output directory
ISOQUANT_PREFIX=""                       # INPUT=isoquant: the --prefix IsoQuant was run with
GTF=""                                   # INPUT=isoquant / light: the reference annotation GTF
BULK=false                               # INPUT=isoquant / light: true = bulk BAM without CB/UB tags
BAM_PATH="/path/to/aligned.bam"          # aligned BAM file
REF_FASTA="/path/to/genome.fa"           # reference genome FASTA
OUTPUT_DIR="/path/to/results"            # pipeline output directory

# ──── How to run ──────────────────────────────────────────────────────────────
RUNNER="slurm"   # slurm = submit the five steps as a dependency-chained job graph
                 # local = run them in order on this machine, no scheduler needed
CORES=8          # local only: how many processes run at once (gene shards for
                 # steps 1–3, workers for step 5)
N_GENE_JOBS=50   # gene shards for steps 1–3 (slurm: array size; local: shards,
                 # CORES of them at a time)

# ──── Samples ─────────────────────────────────────────────────────────────────
N_SAMPLES=1              # number of BAM files; set >1 and use space-separated lists above
SAME_INDIVIDUAL=false    # true  = the N BAMs belong to ONE person (e.g. several
                         #         tissues of one donor): pooled into one variant
                         #         call and one EM, each sample name becomes a
                         #         cell type, and ACTV compares across them
                         # false = N different samples, each processed on its own
CELL_TYPE_DF=""          # single-cell: CSV with Cell/CellType columns (one
                         # space-separated path per sample for multi-sample runs).
                         # Giving it turns on ACTV across cell types in step 5;
                         # with neither CELL_TYPE_DF nor SAME_INDIVIDUAL there are
                         # no cell types to compare and no ACTV table is written.
PREFIX=""                # output filename prefix (leave empty for none)

# ──── Platform ────────────────────────────────────────────────────────────────
# The preset carries the measured calling parameters AND the SNV classifier
# trained for that library (README "Platform presets"). Do not hand-set those
# parameters here.
#   ont-cdna | ont-drna | hifi-isoseq | hifi-masseq
#   other  = a library none of the shipped classifiers was trained for: the
#            shared read-counting parameters, no classifier, clf-free marker
#            priors (from read linkage)
PLATFORM="ont-cdna"
HIGH_ARTIFACT_MODE=false   # true enables the nascent-RNA leak filters for snRNA-seq

# ──── EM haplotyping (step 3) ─────────────────────────────────────────────────
SEED=42                    # random seed
MAX_ITER=50                # maximum EM iterations per gene
TOL=1e-3                   # convergence tolerance
GAP_TAU=1.0                # marker-selection elbow threshold: 1.0 disables the
                           # elbow rule, 0.10 enables it (used for simulated data)
RNA_EDITING_DB=""          # override the bundled hg38 A-to-I database; leave empty
                           # to use the bundled one, or set to "none" to disable
                           # the RNA-editing filter entirely

# ──── Downstream analysis (step 5) ───────────────────────────────────────────
EVENT_MODE="all_events"    # all_events | switching_events | fdr_events
SNV_EVENT_DISTANCE=50      # ±bp exonic distance for SNV–event linking
EVENT_MIN_READS=10         # minimum weighted reads per event test
ASTU_SIG_ONLY=false        # true restricts step 5 to ASTU-significant genes

# ──── Anything else ───────────────────────────────────────────────────────────
EXTRA_OPTS=""              # forwarded verbatim to every step, e.g.
                           # "--gene_subset_path genes.txt --actv_permutations 1000"

# ──── SLURM resources (RUNNER=slurm only) ─────────────────────────────────────
PARTITION="cpu"   # shared across all steps

# Steps 1, 2, 1.5: lightweight per-gene / per-sample pileup
MEM_12="16G"  ; TIME_12="4:00:00"  ; CPUS_12=2
# Step 3: EM haplotyping — more memory for large genes
MEM_3="32G"   ; TIME_3="8:00:00"   ; CPUS_3=4
# Step 4: aggregation across all genes — single job, high memory
MEM_4="64G"   ; TIME_4="4:00:00"   ; CPUS_4=8
# Step 5: downstream analysis — parallel workers, high memory
MEM_5="64G"   ; TIME_5="6:00:00"   ; CPUS_5=8
