# LongAllele configuration reference

## Configuration options

Copy [config_template.sh](../config_template.sh) to `my_run.sh`, edit the settings below, then run:

| Action | Command |
|---|---|
| Run the pipeline | `bash longallele.sh my_run.sh` |
| Preview SLURM submissions without submitting | `DRY_RUN=1 bash longallele.sh my_run.sh` |

Defaults below match the template. Replace the example paths with your own.

### Input files

| Setting | Default | How to set it |
|---|---|---|
| `BAM_PATH` | Set your path | Genome-aligned, indexed BAM, e.g. `"/data/sample.bam"`. |
| `REF_FASTA` | Set your path | Reference genome FASTA matching the BAM alignment, e.g. `"/data/genome.fa"`. |
| `OUTPUT_DIR` | Set your path | [Results](output_schema.md) directory, e.g. `"/data/results/sample1"`; use a new directory when changing inputs/settings. Logs go to `logs/` in the launch directory. |
| `INPUT` | `"scotch"` | `"scotch"`, `"isoquant"`, or `"light"` (BAM + FASTA + GTF). See [input preparation](upstream_input_contract.md) for each interface's required and optional inputs. |
| `SCOTCH_TARGET` | Set for SCOTCH input | SCOTCH output directory. Required when `INPUT="scotch"`; otherwise leave unchanged. |
| `ISOQUANT_DIR` | `""` | IsoQuant output directory. Required when `INPUT="isoquant"`. |
| `ISOQUANT_PREFIX` | `""` | The prefix used for your IsoQuant run, e.g. `"sample1"`. Leave empty to search the specified IsoQuant directory automatically. |
| `GTF` | `""` | Reference annotation GTF matching the genome. Required for light/IsoQuant input; recommended for SCOTCH input. |

### Data type and platform

| Setting | Default | How to set it |
|---|---|---|
| `BULK` | `false` | For light/IsoQuant input: `true` for bulk RNA-seq without cell tags; `false` for single-cell/nucleus data. SCOTCH input uses its upstream cell mappings. |
| `CELL_TYPE_DF` | `""` | Optional CSV with `Cell` (barcode) and `CellType` (label) columns for cell-type analyses. Barcodes must match your input. |
| `PLATFORM` | `"ont-cdna"` | ONT cDNA: `"ont-cdna"`; ONT direct RNA: `"ont-drna"`; PacBio HiFi Iso-Seq: `"hifi-isoseq"`; PacBio MAS-Seq: `"hifi-masseq"`; other platforms: `"other"` (no classifier). |

### Where to run

| Setting | Default | How to set it |
|---|---|---|
| `RUNNER` | `"slurm"` | `"local"` to run on the current computer/server; `"slurm"` to submit to a SLURM cluster. For SLURM, set `PARTITION` and review the resource settings below. |
| `CORES` | `8` | Number of tasks allowed to run at once for local runs. Use a smaller number if memory is limited. Not used for SLURM runs. |
| `N_GENE_JOBS` | `50` | Total number of gene-processing jobs. Normally leave at `50`; this is not the CPU count. |

### Sample settings

| Setting | Default | How to set it |
|---|---|---|
| `N_SAMPLES` | `1` | Number of samples; keep `1` for light/IsoQuant input. For multiple SCOTCH samples, supply space-separated `BAM_PATH`, `SCOTCH_TARGET` and optional `CELL_TYPE_DF` lists in matching sample order. Individual paths must not contain spaces. |
| `SAME_INDIVIDUAL` | `false` | `false` to analyze samples independently. Set `true` to compare multiple bulk samples from the same individual, such as different tissues. Requires `INPUT="scotch"` and `N_SAMPLES` of at least `2`. |
| `PREFIX` | `""` | Optional output filename label, e.g. `"sample1"`. |
| `HIGH_ARTIFACT_MODE` | `false` | Set `true` to apply additional filtering for single-nucleus RNA artifacts. Keep `false` with IsoQuant input, where this option is unsupported. |

### Phasing and RNA editing

| Setting | Default | How to set it |
|---|---|---|
| `SEED` | `42` | Random seed. |
| `MAX_ITER` | `50` | Maximum phasing iterations per gene. |
| `TOL` | `1e-3` | Phasing convergence tolerance. |
| `GAP_TAU` | `1.0` | Additional SNV-filtering threshold; `1.0` disables it. |
| `RNA_EDITING_DB` | `""` | For human hg38, leave empty to use the bundled RNA-editing database. For another genome build or species, enter a compatible `.npz` database path or set `"none"` to disable the filter. |
| `SR_BAM` | `""` | Indexed short-read BAM aligned to the same reference; empty disables the filter and its `SR_*` thresholds. Supply one shared BAM or space-separated paths in `BAM_PATH` order, without spaces within paths. Evidence is pooled with `SAME_INDIVIDUAL=true`; the filter is skipped when step 3 uses `--snv_confidence_path`. |
| `SR_MIN_DEPTH` | `30` | Minimum short-read depth required to remove a candidate SNV. Lower-coverage sites are kept. |
| `SR_MAX_ALT` | `1` | Remove a candidate with at most this many non-reference short-read bases, provided depth reaches `SR_MIN_DEPTH`. |
| `SR_MIN_MAPQ` | `20` | Minimum mapping quality for short reads counted by this filter. |
| `SR_MIN_BASEQ` | `20` | Minimum base quality for short-read bases counted by this filter. |

### Exon and junction analyses

Regular mode only (`INPUT="scotch"` or `INPUT="isoquant"`).

| Setting | Default | How to set it |
|---|---|---|
| `EVENT_MODE` | `"all_events"` | `"all_events"`: all events; `"switching_events"`: isoform-switching boundaries; `"fdr_events"`: events passing FDR ≤ 0.05. Change the cutoff through `EXTRA_OPTS`, e.g. `"--fdr_events_value 0.01"`. |
| `SNV_EVENT_DISTANCE` | `50` | Maximum exonic distance in bases between an event and a linked SNV. |
| `EVENT_MIN_READS` | `10` | Minimum distinct reads per event: inclusion reads for the isoform test; include+skip reads for the alignment-based test. |
| `ASTU_SIG_ONLY` | `false` | Set `true` to restrict event analysis to ASTU-significant genes. |

### Additional command-line options

| Setting | Default | How to set it |
|---|---|---|
| `EXTRA_OPTS` | `""` | Additional flags from the [command-line reference](command_line.md). Combine all flags in one quoted string. |

Examples:

| Use | `EXTRA_OPTS` value |
|---|---|
| Analyze listed genes | `"--gene_subset_path /data/genes.txt"` |
| Use the selected platform without its classifier | `"--setting clf-free"` |
| Export combined count matrices as CSV | `"--csv"` |
| Export combined count matrices as Matrix Market | `"--mtx"` |
| Combine options | `"--setting clf-free --csv"` |

### SLURM resources

Used only when `RUNNER="slurm"`. Set memory, time and CPU requests within your cluster's limits. Memory values accept units such as `G`; times use `hours:minutes:seconds`.

| Setting | Default | How to set it |
|---|---|---|
| `PARTITION` | `"cpu"` | Your cluster's partition name. |
| `MEM_12` | `"16G"` | Memory per job for SNV calling and read preparation. |
| `TIME_12` | `"4:00:00"` | Time limit per job for SNV calling and read preparation. |
| `CPUS_12` | `2` | CPUs requested per job for SNV calling and read preparation. |
| `MEM_3` | `"32G"` | Memory per phasing job. |
| `TIME_3` | `"8:00:00"` | Time limit per phasing job. |
| `CPUS_3` | `4` | CPUs requested per phasing job. |
| `MEM_4` | `"64G"` | Memory for generating summaries and count matrices. |
| `TIME_4` | `"4:00:00"` | Time limit for generating summaries and count matrices. |
| `CPUS_4` | `8` | CPUs requested for generating summaries and count matrices. |
| `MEM_5` | `"64G"` | Memory per downstream analysis job. |
| `TIME_5` | `"6:00:00"` | Time limit per downstream analysis job. |
| `CPUS_5` | `8` | CPUs requested per downstream analysis job. |
