# LongAllele command-line reference

For the config-based workflow, see the [usage guide](pipeline_steps.md). Run the commands below from the repository root in a Bash shell.

## Full-mode commands

This example processes one sample. Set `UPSTREAM_DIR` to your SCOTCH output or converted IsoQuant directory (`isoquant_upstream`). See [IsoQuant input preparation](upstream_input_contract.md#isoquant) for required inputs.

```bash
UPSTREAM_DIR="/path/to/scotch_output"
SAMPLE_BAM="/path/to/sample.bam"
GENOME_FASTA="/path/to/genome.fa"
GENE_GTF="/path/to/genes.gtf"
RESULTS_DIR="/path/to/results"
SEQ_PLATFORM="ont-cdna"

common=(
    --scotch_target "$UPSTREAM_DIR"
    --bam_path "$SAMPLE_BAM"
    --ref_fasta_path "$GENOME_FASTA"
    --output_folder "$RESULTS_DIR"
    --platform "$SEQ_PLATFORM"
    --prefix sample1
    --n_jobs 1 --job_index 0
)
```

Use the same input paths, platform and prefix for every step.

### Step 1 — Call SNVs

```bash
python src/longallele.py --task step1 "${common[@]}"
```

### Step 2 — Prepare reads for phasing

```bash
python src/longallele.py --task step2 "${common[@]}"
```

### Step 3 — Phase SNVs and assign reads to haplotypes

```bash
python src/longallele.py --task step3 "${common[@]}" --n_workers 4 --gtf_path "$GENE_GTF"
```

For cell-type-specific results, add `--cell_type_df_path /path/to/cell_types.csv` to steps 3, 4 and 5. For a non-hg38 reference, add `--rna_editing_db /path/to/editing_sites.npz` or `--rna_editing_db none` to step 3.

### Step 4 — Write summaries and count matrices

```bash
python src/longallele.py --task step4 "${common[@]}" --n_workers 4 --csv
```

### Step 5 — Test transcript and exon/junction usage

```bash
python src/longallele.py --task step5 "${common[@]}" --n_workers 4
```

Results are described in the [output reference](output_schema.md).

`--csv` exports combined count matrices; use `--mtx` for Matrix Market. Without either flag, these combined matrices are not exported.

### Optional read-level exon/junction validation

Run these commands after step 1 and before step 5 to include the `obs_*` validation columns in `event_snv.csv`. The config-based workflow runs these tasks automatically in full mode.

```bash
python src/longallele.py --task step1_5 "${common[@]}"
python src/longallele.py --task step1_5_merge "${common[@]}"
```

## Light-mode commands

Set the BAM, FASTA, output and platform variables as in the full-mode example. Prepare the gene assignments:

```bash
GENE_GTF="/path/to/genes.gtf"

python src/longallele.py --task light_prep \
    --bam_path "$SAMPLE_BAM" --gtf_path "$GENE_GTF" \
    --output_folder "$RESULTS_DIR" --platform "$SEQ_PLATFORM" \
    --n_jobs 1 --job_index 0

python src/longallele.py --task light_merge \
    --gtf_path "$GENE_GTF" --output_folder "$RESULTS_DIR" --n_jobs 1
```

For bulk data, add `--bulk` to both commands. For single-cell data, the BAM must contain the cell and UMI tags used by `--barcode_cell` and `--barcode_umi` (defaults: `CB` and `UB`).

Set `UPSTREAM_DIR="$RESULTS_DIR/light_upstream"` and rebuild the `common` array above. Run steps 1 and 2, then:

```bash
python src/longallele.py --task step3 "${common[@]}" --n_workers 4 --skip_astu_test --gtf_path "$GENE_GTF"
python src/longallele.py --task step4 "${common[@]}" --n_workers 4 --csv
```

Light mode ends at step 4. For phasing and counts without ASE testing, replace `--skip_astu_test` with `--skip_tests` in step 3.

## Common options

When using a config file, use its named settings where available. Add other flags to `EXTRA_OPTS`, for example:

```bash
EXTRA_OPTS="--gene_subset_path /path/to/genes.txt"
```

| Option | Use |
|---|---|
| `--gene_subset_path` | Analyze only the gene IDs listed in a text file, one per line. |
| `--prefix` | Add a sample label to output filenames. Use the same value for all steps. |
| `--cell_type_df_path` | CSV with `Cell` and `CellType` columns; barcode values must match your input. |
| `--barcode_cell`, `--barcode_umi` | Change the BAM tag names used for light or IsoQuant input. Defaults: `CB`, `UB`. |
| `--setting clf-free` | Run without the SNV classifier while keeping the selected platform's other settings. |
| `--high_artifact_mode` | Apply additional single-nucleus RNA filters. Unsupported with IsoQuant input. |
| `--cover_existing` | Recompute existing per-gene results in step 3. For changes to inputs or analysis settings, use a new output directory for a complete run. |
| `--mtx`, `--csv` | Select Matrix Market or CSV count output in steps 3 and 4. |
| `--n_workers` | Control parallel processing in steps 2–5. |

## RNA-editing database

| Option | Use |
|---|---|
| `--rna_editing_db /path/to/editing_sites.npz` | Use an editing database for your genome build. |
| `--rna_editing_db none` | Disable the RNA-editing filter. |

Without this option, step 3 uses the bundled hg38 database. The config-file setting `RNA_EDITING_DB=""` also uses the bundled database.

Custom databases must use LongAllele's `.npz` format: sorted arrays of 0-based positions keyed by `AG__<chromosome>` or `TC__<chromosome>`. Chromosome names must match the BAM.

## Filtering and phasing options

The platform preset supplies the normal filtering settings. Explicit flags override the preset; use the same calling settings in steps 1–3.

| Option | Use |
|---|---|
| `--depth` | Minimum read coverage at a candidate SNV. |
| `--n_alt_count` | Candidate SNVs must have more alternate-allele reads than this value; `2` requires at least 3. |
| `--min_alt_frac` | Minimum alternate-allele fraction; `0` disables this cutoff. |
| `--min_mapq` | Minimum read mapping quality. |
| `--min_baseq` | Minimum base quality. |
| `--min_dist_to_end` | Exclude bases within this distance of either read end. |
| `--snv_confidence_path` | Supply known heterozygous SNVs in a TSV with `chrom`, `pos`, `ref` and `alt` columns; positions are 0-based. Provide one file per BAM. |
| `--snv_classifier` | Supply a compatible custom classifier (`.joblib`). |
| `--max_iter` | Maximum phasing iterations per gene; default `50`. |
| `--tol` | Phasing convergence tolerance; default `0.001`. |
| `--seed` | Random seed; default `42`. |
| `--gtf_path` | Reference annotation GTF; in step 3 it defines exon/intron for the SNV region labels (see `snv_region_label` in [docs/output_schema.md](output_schema.md)). |
| `--no_snv_region_rerun` | Keep the exonic/intronic SNV labels but do not re-run phasing on the exonic candidates when a gene's kept markers are all intronic (see `snv_region_label` in [docs/output_schema.md](output_schema.md)). |

## ASE and ASTU calls

These step 5 options control the final calls in `gene_snv.csv`. Both calls require a BH-adjusted p-value ≤ 0.05.

| Option | Use | Default |
|---|---|---|
| `--ase_call_margin` | ASE passes if its entire interval lies on one side of 0.5, or its interval midpoint differs from 0.5 by at least this amount. | 0.095 |
| `--conf_nonphasable_astu` | Minimum `conf_astu` for an ASTU call; 1 requires significance at the least favorable allocation. | 1.0 |

## Cell-type comparisons

Full mode automatically tests differences across cell types when `--cell_type_df_path` is provided, or across tissues with `--same_individual`. A gene needs at least two cell types or tissues with enough reads to be tested.

| Option | Use | Default |
|---|---|---|
| `--no-actv` | Disable cross-cell-type/tissue testing. | Testing is enabled when contexts are provided. |
| `--actv_permutations` | Number of permutations per gene. | 300 |
| `--actv_min_phasable_reads` | Minimum phasable reads per cell type or tissue. | 20 |
| `--actv_min_cells` | Minimum expressing cells per cell type; applies to single-cell comparisons. | 10 |
| `--actv_min_phasable_frac` | Set the threshold for the output flag `pass_phasable_frac`. | 0.6 |
| `--actv_min_actv` | Set the effect-size threshold for the output flag `pass_min_actv`. | 0.3 |
| `--actv_min_snvs` | Minimum sample-level called heterozygous SNVs per gene for `pass_min_snvs`; 0 disables this condition. | 2 |

The last three options set output flags; they do not remove rows from `actv_results.csv`. The `actv_gene_call` column combines these flags with permutation significance and the corresponding within-context significance requirement.

## Exon and junction results

| Option | Use | Default |
|---|---|---|
| `--event_mode` | Report `all_events`, `switching_events`, or `fdr_events`. | `all_events` |
| `--event_min_reads` | Minimum distinct reads for each test: inclusion reads for isoform evidence; include+skip reads for raw-read evidence. The gates are independent; the tests still use haplotype-probability weights. | 10 |
| `--snv_event_distance` | Maximum exonic distance in bases for linking an SNV to an event. | 50 |
| `--fdr_events_value` | FDR cutoff when using `fdr_events`. | 0.05 |
| `--astu_sig_only` | Restrict event analysis to ASTU-significant genes. | Off |

## Check completion

For the single-sample commands above:

```bash
python src/longallele.py --task check "${common[@]}"
```

If you ran steps with more than one job, pass the same `--n_jobs` value when checking completion.

For the complete list of available flags:

```bash
python src/longallele.py --help
```
