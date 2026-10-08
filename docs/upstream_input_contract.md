# Input preparation

LongAllele provides two modes: [Regular](#regular-mode) for haplotype inference and downstream allelic testing (ASE, ASTU, exon/junction tests and ACTV), and [Light](#light-mode) for haplotype inference and gene-level ASE.

Set the variables below in your copy of [config_template.sh](../config_template.sh); pass options beginning with `--` through `EXTRA_OPTS`, for example `EXTRA_OPTS="--barcode_cell CR"`.

## Regular mode

Regular mode supports different upstream tools, including [SCOTCH](#scotch), [IsoQuant](#isoquant), and [other tools](#other-upstream-tools).

### SCOTCH

LongAllele reads SCOTCH output directly, including its read-to-isoform assignments, transcript annotations and cell mappings.

**Required**

| Parameter | Value |
|---|---|
| `INPUT` | `scotch` |
| `SCOTCH_TARGET` | SCOTCH output directory |
| `BAM_PATH` | Genome-aligned, indexed BAM used by SCOTCH |
| `REF_FASTA` | Matching reference genome FASTA |

**Optional**

| Parameter | Default | Use |
|---|---|---|
| `GTF` | Empty | Reference gene annotation for exonic/intronic SNV labels |
| `CELL_TYPE_DF` | Empty | CSV with `Cell` and `CellType` columns for cell-type analyses |
| `--sample_name_parse` | Unset | Read assignments from SCOTCH's `samples/<name>/` subdirectory |
| `--ref_pickle_path` | Auto-detected | Specify the gene-structure pickle explicitly |

### IsoQuant

LongAllele automatically converts IsoQuant's read assignments, transcript model reads and extended annotation into its input format.

**Required**

| Parameter | Value |
|---|---|
| `INPUT` | `isoquant` |
| `ISOQUANT_DIR` | IsoQuant output directory |
| `GTF` | Reference gene annotation GTF used by IsoQuant |
| `BAM_PATH` | Genome-aligned, indexed BAM used by IsoQuant |
| `REF_FASTA` | Matching reference genome FASTA |
| `BULK` | `true` for bulk; `false` for single-cell/nucleus |

**Optional**

| Parameter | Default | Use |
|---|---|---|
| `ISOQUANT_PREFIX` | Empty | Run prefix when outputs are in a subdirectory with that name |
| `CELL_TYPE_DF` | Empty | CSV with `Cell` and `CellType` columns for cell-type analyses |
| `--isoquant_keep_policy` | `default` | Read-assignment selection policy: `default` or `strict` |
| `--novel_min_support_reads` | `2` | Minimum supporting reads for a novel transcript model |
| `--isoquant_no_model_construction` | Off | Use output without transcript model construction; excludes novel models |
| `--barcode_cell` | `CB` | BAM tag containing cell barcodes |
| `--barcode_umi` | `UB` | BAM tag containing UMIs |
| `--isoquant_force_version` | Off | Bypass the adapter's version check |

### Other upstream tools

Other tools require a custom converter that produces a read-assignment TSV, transcript annotation GTF and gene-structure pickle following the [adapter file specification](upstream_format.md).

**Required**

| Parameter | Value |
|---|---|
| `INPUT` | `scotch` — also used to read converted directories |
| `SCOTCH_TARGET` | Converted output directory |
| `BAM_PATH` | Corresponding genome-aligned, indexed BAM |
| `REF_FASTA` | Matching reference genome FASTA |

**Optional**

| Parameter | Default | Use |
|---|---|---|
| `GTF` | Empty | Reference gene annotation for exonic/intronic SNV labels |
| `CELL_TYPE_DF` | Empty | CSV with `Cell` and `CellType` columns for cell-type analyses |
| `--ref_pickle_path` | Auto-detected | Specify the gene-structure pickle explicitly |

## Light mode

LongAllele assigns reads to genes using the reference GTF; no upstream isoform assignments are needed.

**Required**

| Parameter | Value |
|---|---|
| `INPUT` | `light` |
| `BAM_PATH` | Genome-aligned, indexed BAM |
| `REF_FASTA` | Matching reference genome FASTA |
| `GTF` | Reference gene annotation file in GTF format |
| `BULK` | `true` for bulk; `false` for single-cell/nucleus |

**Optional**

| Parameter | Default | Use |
|---|---|---|
| `CELL_TYPE_DF` | Empty | CSV with `Cell` and `CellType` columns for cell-type analyses |
| `--barcode_cell` | `CB` | BAM tag containing cell barcodes |
| `--barcode_umi` | `UB` | BAM tag containing UMIs |
| `--light_assign_by` | `gene_range` | Assign reads by overlap with the whole gene span or, with `exonic`, its exons |
| `--light_min_exonic_bp` | `0` | Minimum exonic overlap in bases; keep 0 for `gene_range`, set at least 1 for `exonic` |
| `--light_ambiguity_ratio` | `1.0` | Minimum ratio of the best gene's overlap to the second-best gene's |
| `--target_units` | Automatic | Number of work units for preparing read assignments |
