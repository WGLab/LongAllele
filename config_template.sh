# LongAllele pipeline configuration
# Usage: bash longallele.sh config_template.sh
# Copy this file, fill in your paths, and run.

# ── Required ──────────────────────────────────────────────────────────────────
SCOTCH_TARGET="/path/to/scotch_output"   # SCOTCH output directory
BAM_PATH="/path/to/aligned.bam"          # aligned BAM file
REF_FASTA="/path/to/genome.fa"           # reference genome FASTA
OUTPUT_DIR="/path/to/results"            # pipeline output directory

# ── Parallelization ───────────────────────────────────────────────────────────
N_GENE_JOBS=50   # SLURM array size for steps 1–3 (splits work across genes)
N_SAMPLES=1      # number of BAM files (1 for single-sample; set to n_samples for
                 # multi-sample, and pass space-separated lists above)

# ── Optional ──────────────────────────────────────────────────────────────────
CELL_TYPE_DF=""  # path to CSV with Cell/CellType columns (leave empty if not needed)
PREFIX=""        # output filename prefix (leave empty for none)

# ── SLURM resources ───────────────────────────────────────────────────────────
PARTITION="cpu"
MEM="32G"
TIME="12:00:00"
CPUS=4
