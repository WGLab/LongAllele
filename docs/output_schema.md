# Output files and columns

SNV positions are **0-based**; event intervals are **0-based, half-open** (`start` included, `end` excluded). Empty values mean unavailable or not tested. Haplotype labels A/B are local to each gene, not parental labels.

## Output files

Paths are relative to `OUTPUT_DIR` for one sample, or `OUTPUT_DIR/{sample_name}` for multiple independent samples. With `PREFIX=sample1`, directory names gain `_sample1`; with the default empty prefix, use the names below. Light mode ends at step 4.

| File | Contents |
|---|---|
| `summary_statistics/summary_statistics.csv` | Gene × cell type: read counts, allelic balance, ASE and ASTU tests |
| `snv_hap/snv_hap_map.csv` | SNV haplotype assignments |
| `snv_hap/read_hap_map.csv` | Read haplotype probabilities |
| `count_matrix_hap/all_genes/` | Combined count matrices and isoform count tables |
| `downstream/gene_snv.csv` | SNV × gene × cell type: effects and final ASE/ASTU calls |
| `downstream/event_snv.csv` | Regular mode: exon/junction events and linked SNVs with `EVENT_MODE=all_events`; no significance filter. |
| `downstream/event_snv_switching.csv` | Event table with `EVENT_MODE=switching_events`. |
| `downstream/event_snv_fdr005.csv` | Event table with `EVENT_MODE=fdr_events` at cutoff 0.05; suffix becomes `fdr001` for 0.01. |
| `downstream/actv_results.csv` | Cross-cell-type/tissue tests; written when ACTV is enabled. See [comparison options](command_line.md#cell-type-comparisons). |

Summary and haplotype CSVs have an unnamed first column containing the row index; read them with `index_col=0` in pandas. Step 5 tables have no index column.

## Step 4 — `summary_statistics.csv`, `snv_hap_map.csv`, `read_hap_map.csv`

**`summary_statistics.csv` — per-gene haplotype + isoform statistics**

| Column | Description |
|---|---|
| `geneID`, `geneName` | Gene identifiers |
| `n_reads`, `n_reads_phasable`, `n_snvs` | Total reads, reads covering at least one retained phasing marker, and candidate SNVs in the reported fit. |
| `alpha_hat`, `alpha_hat_low`, `alpha_hat_high` | Minor-haplotype read fraction: EM estimate and bounds allowing non-phasable read allocation and phase-block orientation changes. These are allocation bounds, not confidence intervals. |
| `alpha_hat_block_low`, `alpha_hat_block_high` | Bounds from phase-block orientation alone; equal endpoints for a single block. |
| `n_phase_blocks` | Number of phase blocks; 1 means all retained markers are connected. |
| `major_hap` | Major haplotype label (`A` or `B`) |
| `ll_alt`, `ll_null` | Alternative and null (α = 0.5) log-likelihoods for the reported ASE test. Multi-block Bulk rows use the orientation with the largest p-value, fitting α separately for each orientation. |
| `lrt_stat`, `p_value` | ASE likelihood-ratio statistic and raw p-value. Multi-block Bulk rows use the largest p-value across orientations; more than 7 blocks gives missing test values. |
| `p_value_orient_min`, `n_orientations` | Smallest ASE p-value across block orientations, and number of orientations tested (Bulk rows). |
| `p_value_gene_adj` | ASE p-value adjusted by Benjamini–Hochberg (BH) across genes within each sample and cell type. |
| `chi2_isoform`, `df_isoform` | ASTU chi-square statistic and degrees of freedom; empty for multi-block Bulk rows. |
| `p_value_isoform`, `p_value_isoform_high`, `p_value_isoform_low` | ASTU p-value and upper/lower allocation bounds. Multi-block Bulk rows include block orientations; more than 7 blocks gives missing values. |
| `p_value_isoform_orient_min` | Smallest point ASTU p-value across block orientations (Bulk rows). |
| `p_value_isoform_adj`, `p_value_isoform_adj_high`, `p_value_isoform_adj_low` | FDR-adjusted ASTU p-values |
| `snv_region_label` | `exonic`: retained exonic marker; `intron_snv_only`: only intronic markers retained; `exonic_rerun`: result from an exonic-only rerun; `no_marker`: no confident marker; `unknown`: no usable reference annotation. |
| `n_candidates_exonic`, `n_candidates_intronic` | Candidate SNV counts by region before the optional exonic-only rerun. Regions use the supplied reference GTF. |
| `n_markers_exonic`, `n_markers_intronic`, `em_rerun_exonic` | Retained marker counts by region; `em_rerun_exonic=1` means the exonic-only rerun was used. |
| `CellType` | Cell type identifier (`Bulk` for bulk-level rows) |

**`snv_hap_map.csv` — per-SNV haplotype assignments**

| Column | Description |
|---|---|
| `chrom`, `pos` | SNV genomic coordinates (0-based) |
| `ref`, `alt` | Reference and alternate alleles |
| `depth`, `alt_count`, `alt_frac` | Total read depth, alternate-allele read count, and alternate-allele fraction at the SNV. |
| `ID` | SNV key in this table: `chr_pos_ref`, e.g. `chr22_20859091_C`. |
| `het_prob` | Posterior probability of heterozygosity under the configured genotype model. |
| `h_A` | Probability that the alt allele is on haplotype A |
| `h_m` | Marker probability — confidence that the SNV is a true heterozygous phasing marker |
| `phase_block` | Phase-block ID within the gene; 0 means not assigned to a retained-marker block. |
| `in_gene_span` | 1 if the SNV lies within the annotated gene interval; 0 if outside. |
| `hat_Z_binary` | Binary indicator (1 = SNV retained as a phasing marker after EM) |
| `snv_region` | `exonic`, `intronic`, or `unknown`, relative to the gene in the supplied reference GTF. |
| `geneName`, `geneID` | Gene identifiers |

**`read_hap_map.csv` — per-read haplotype posteriors**

| Column | Description |
|---|---|
| `Read` | BAM read name after normalization: names containing `/` have a trailing `_<digits>` suffix removed. |
| `hat_I` | Posterior probability that the read is on haplotype A |
| `hat_I_B` | Posterior probability that the read is on haplotype B (= 1 − `hat_I`) |
| `reads_phasable` | 1 if the read covers at least one retained phasing marker (so its `hat_I` comes from evidence), 0 if non-phasable (`hat_I` is the gene-level prior) |
| `read_block` | Which phase block the read's markers belong to (0 = none / orientation-invariant) |
| `geneName`, `geneID` | Gene identifiers |

## Step 5 — `gene_snv.csv`, `event_snv.csv`, `actv_results.csv`

**`gene_snv.csv` — one row per confident phased SNV per gene per cell type**

| Column | Description |
|---|---|
| `Sample`, `CellType` | Sample and cell type identifiers |
| `geneID`, `geneName`, `geneChr` | Gene identifiers |
| `n_reads`, `n_reads_phasable`, `gene_n_snvs`, `gene_n_snvs_called` | Total and phasable reads in this context; candidate SNVs in the reported gene fit; sample-level SNVs passing phasing confidence ≥ 0.5. |
| `gene_alpha_hat`, `gene_alpha_hat_low`, `gene_alpha_hat_high` | Minor-haplotype read fraction and allocation/orientation bounds, from the summary table. |
| `gene_alpha_hat_block_low`, `gene_alpha_hat_block_high` | Bounds from phase-block orientation alone. |
| `gene_alpha_hat_major`, `gene_alpha_hat_major_low`, `gene_alpha_hat_major_high` | Major haplotype allelic balance (1 − minor) |
| `gene_major_hap`, `gene_minor_hap` | Haplotype labels (A or B) |
| `gene_n_phase_blocks` | Number of phase blocks (1 = fully connected) |
| `gene_snv_region_label` | `exonic` / `intron_snv_only` / `exonic_rerun` / `no_marker` / `unknown`, copied from `summary_statistics.csv` (see there) |
| `gene_p_value`, `gene_p_value_adj` | Raw ASE p-value and BH-adjusted p-value within the sample and cell type. |
| `ASE_call` | `1`: adjusted p ≤ 0.05 and either the whole allocation interval lies on one side of 0.5 or `abs((low + high) / 2 − 0.5) ≥ --ase_call_margin` (default 0.095); `-1`: adjusted p > 0.05; `0`: neither condition, including missing test values. |
| `overall_dominant_isoform` | Most expressed isoform across both haplotypes in the extrapolated count table, excluding the pooled `Other` category. |
| `top_isoform_hap_major`, `top_isoform_hap_minor` | Top isoform or pooled `Other` category on each haplotype in the extrapolated count table. |
| `top_isoform_hap_major_frac`, `top_isoform_hap_minor_frac` | Fraction of each haplotype's reads assigned to its own top isoform; the two isoforms may differ. |
| `overall_dominant_frac_hap_major`, `overall_dominant_frac_hap_minor` | Unsmoothed usage fraction of the same overall dominant isoform on the major/minor haplotype. |
| `isoform_p_value`, `isoform_p_value_adj` | Raw ASTU p-value and BH-adjusted p-value within the sample and cell type. |
| `isoform_p_value_high`, `isoform_p_value_low`, `isoform_p_value_adj_high`, `isoform_p_value_adj_low` | Upper/lower ASTU p-value bounds and their BH-adjusted values. |
| `conf_astu` | Fraction of the allocation path from strongest to weakest ASTU evidence over which raw p ≤ 0.05; range 0–1. Multi-block Bulk rows use the minimum across orientations; missing if it cannot be evaluated. |
| `ASTU_call` | `1`: adjusted p ≤ 0.05 and `conf_astu ≥ --conf_nonphasable_astu` (default 1); `-1`: adjusted p > 0.05; `0`: neither condition, including missing test values. |
| `shrinkage_k` | Pseudocount used to regularize effect sizes. |
| `es_ase_point`, `es_ase_cons` | Regularized log2 major/minor read-count ratio. `point` uses the EM estimate; `cons` uses the midpoint of the allocation/orientation bounds. |
| `es_astu_point`, `es_astu_cons` | Absolute regularized log2 ratio of dominant-isoform usage between haplotypes. `point` uses proportional extrapolation of phasable counts; `cons` uses the midpoint of each haplotype's extreme usage fractions before taking the ratio. |
| `astu_source` | Source of isoform counts: `bulk`, `ct_specific`, or `bulk_fallback` when cell-type tables are unavailable. |
| `snvID` | Stable SNV key (`chr:pos:ref:alt`) |
| `snv_pos`, `snv_ref`, `snv_alt` | SNV coordinates and alleles |
| `snv_depth_bulk`, `snv_alt_count_bulk`, `snv_alt_frac_bulk` | Total depth, alternate-allele count and alternate-allele fraction from the pooled pileup. |
| `h_A`, `hat_Z_prob_revised` | Alt-allele probability on haplotype A, and phasing confidence `round(h_m × (1 − binary entropy(h_A)), 2)`. |
| `snv_hap` | Haplotype carrying the alt allele (A or B) |
| `snv_on_minor_hap` | Whether SNV alt allele is on the minor haplotype |
| `snv_expr_direction` | `+` if the alt allele is on the major-expression haplotype; `-` if on the minor haplotype. |
| `snv_es_ase_signed` | `es_ase_cons` with the sign given by `snv_expr_direction`. |
| `overall_dominant_pref_hap` | Haplotype with higher dominant isoform usage |
| `snv_astu_direction` | `+` if the alt-allele haplotype is the preferred haplotype for dominant-isoform usage; `-` otherwise. |
| `snv_es_astu_signed` | `es_astu_point` with the sign given by `snv_astu_direction`. |

**`event_snv.csv` — one row per event × cell type × linked SNV; events without a linked SNV keep one row with empty SNV fields**

Events pass at least one read-support threshold below, then the `EVENT_MODE` filter. The same columns apply to all event filenames.

| Column | Description |
|---|---|
| `Sample`, `CellType` | Sample and cell type identifiers |
| `geneID`, `geneName`, `geneChr` | Gene identifiers |
| `n_reads`, `n_reads_phasable`, `gene_n_snvs_called`, `gene_major_hap`, `shrinkage_k`, `es_ase_point`, `es_ase_cons`, `es_astu_point`, `es_astu_cons` | Gene context (duplicated from `gene_snv.csv` for self-containment) |
| `ASE_call`, `ASTU_call` | Final ASE / ASTU calls (3-category) for the gene, duplicated from `gene_snv.csv` — see that table for the rules. |
| `overall_dominant_isoform`, `top_isoform_hap_major`, `top_isoform_hap_minor` | Isoform context |
| `eventID` | Stable event key (`event_type:start-end`) |
| `event_type` | `exon` or `junction` |
| `event_start`, `event_end`, `event_length` | 0-based start, exclusive end, and length (`end − start`) in bases. |
| `hapA_present`, `hapA_absent`, `hapB_present`, `hapB_absent` | Haplotype-probability-weighted counts with/without the event, inferred from isoform assignments. Reads may count even if their alignments do not reach the event. |
| `event_n_include_reads` | Distinct reads assigned to an isoform containing the event; the isoform-based test requires ≥ `--event_min_reads` (default 10). |
| `obs_n_observed_reads` | Distinct reads whose alignments include or skip the event; the observed-event test requires ≥ `--event_min_reads` (default 10). |
| `obs_hapA_include`, `obs_hapA_skip`, `obs_hapA_unobserved` | Haplotype-A weighted counts classified by alignment: event included, event skipped, or no determinate observation (e.g. truncated read). |
| `obs_hapB_include`, `obs_hapB_skip`, `obs_hapB_unobserved` | The same alignment-based counts for haplotype B. Each read counts once; isoform-based counts can include multiple assignments. |
| `obs_chi2`, `obs_p_value`, `obs_p_value_adj` | Chi-square statistic, raw p-value and within-gene BH-adjusted p-value for haplotype × observed include/skip counts. Unobserved reads are excluded. All row/column totals must be non-zero. FDR adjustment is separate from the other tests. |
| `obs_test_type` | `chi2_hap_event`: tested; `insufficient_data`: too few observed reads or a zero row/column total; `no_bam`: alignment cache unavailable. Populate the cache with steps 1.5 and 1.5 merge. |
| `event_inclusion_frac_A`, `event_inclusion_frac_B` | Inclusion fraction per haplotype |
| `event_pref_hap` | Haplotype with higher event inclusion |
| `event_pref_major_minor` | `major` or `minor` relative to gene expression |
| `event_chi2`, `event_p_value`, `event_p_value_adj` | Chi-square statistic, raw p-value and within-gene BH-adjusted p-value for haplotype × isoform-inferred event membership. All row/column totals must be non-zero. FDR adjustment is separate from the other tests. |
| `has_linked_snv` | Whether a nearby confident SNV is linked |
| `linked_snv_count` | Number of nearby SNVs linked to this event |
| `is_nearest_snv_for_event` | Whether this is the closest linked SNV |
| `snvID`, `snv_pos`, `snv_ref`, `snv_alt` | Linked SNV identity (`NaN` if none) |
| `snv_hap`, `h_A`, `hat_Z_prob_revised` | SNV phasing info (`NaN` if none) |
| `exonic_distance` | Distance in bases along exonic sequence to the event; 0 for SNVs inside the event, otherwise distance to the nearest boundary. Introns contribute no distance. |
| `genomic_distance` | Genomic distance in bases to the nearest event boundary, including for SNVs inside the event. |
| `snv_expr_direction`, `snv_astu_direction` | SNV regulatory interpretation |
| `snv_event_direction` | `promotes_event` or `reduces_event` |
| `raw_validation_available` | Whether raw read validation was performed |
| `raw_ref_present`, `raw_ref_absent`, `raw_alt_present`, `raw_alt_absent` | Read counts by observed SNV allele (ref/alt) and isoform-inferred event membership (present/absent). |
| `raw_total_reads` | Total raw reads in contingency table |
| `raw_chi2`, `raw_p_value`, `raw_p_value_adj` | SNV-event test statistic, raw p-value and within-gene BH-adjusted p-value across event–SNV pairs. `raw_chi2` is empty for binomial tests. |
| `raw_test_type` | `chi2_cross_event`: allele × event chi-square test; `binomial_intra_event`: ref/alt binomial test against 0.5 for a SNV inside an exon event. |

**`actv_results.csv` — two rows per gene (`ase` and `astu`); failed calls remain in the table**

| Column | Description |
|---|---|
| `Sample`, `geneID`, `axis` | Sample, gene, and which effect this row tests (`ase` or `astu`) |
| `actv` | The statistic: range of the per-cell-type effect across participating cell types |
| `pval` | Permutation p-value, `(1 + #null ≥ actv) / (permutations + 1)` |
| `thr95` | 95th percentile of the permutation null distribution. |
| `call` | `pval ≤ 0.05` |
| `test_status` | `ok`, `missing_inputs`, `insufficient_contexts`, `undefined_observed_statistic`, or `permutation_limit_reached`. |
| `eligible_cells` | Whether the gene had at least two participating cell types |
| `qualifying_contexts` | The participating cell types (JSON list) |
| `delta_by_ct` | Per-context effect (JSON): ASE = `2 × hapA / (hapA + hapB) − 1`; ASTU = dominant-isoform usage in A minus usage in B. Uses phasable counts; undefined effects are `null`. |
| `n_cells_by_ct`, `n_expressing_cells_by_ct`, `n_phasable_reads_by_ct`, `n_reads_by_ct` | Per-context counts (JSON): units with phasable evidence, all expressing cells, phasable reads, and all reads. In read-unit mode, the first counts reads and expressing-cell counts are empty. |
| `n_permutations`, `n_valid_permutations`, `n_permutation_attempts`, `n_invalid_permutations` | Requested, valid, attempted and rejected permutations. |
| `ase_sig_ct`, `astu_sig_ct` | At least one participating context has raw ASE/ASTU p ≤ 0.05/m, where m is the number of participating contexts (Bonferroni correction). |
| `min_phasable_frac`, `pass_phasable_frac` | Phasable / total reads of the weakest participating cell type, and whether it reaches `--actv_min_phasable_frac` |
| `pass_min_actv` | Whether `actv` reaches `--actv_min_actv` |
| `gene_n_snvs_called`, `pass_min_snvs` | Sample-level called SNV count and whether it reaches `--actv_min_snvs` (default 2). Missing/inconsistent counts fail unless the threshold is 0. |
| `actv_gene_call` | Final ACTV call: `test_status=ok`, `call`, all three `pass_*` flags and the matching `ase_sig_ct`/`astu_sig_ct` flag must pass. |

## Count matrices and isoform tables

Combined matrices are exported only with `--csv` or `--mtx` (`EXTRA_OPTS="--csv"` or `"--mtx"` in a config). Files below are in `count_matrix_hap/all_genes/`, with the configured prefix added to the directory name.

| File | Rows and columns |
|---|---|
| `count_mat_gene.csv` | Rows: cells (one sample-level cell for bulk). Columns: `{geneName}_hapA` and `{geneName}_hapB`. Values: summed read-haplotype probabilities. |
| `count_mat_transcript.csv` | Same rows; columns: `{geneName}_{isoform}_hapA` and `_hapB`. |
| `count_mat_gene_phasable.csv`, `count_mat_transcript_phasable.csv` | Same layout, restricted to reads covering retained phasing markers. |
| `count_mat_*_ct_{cell_type}.csv` | The corresponding matrix restricted to cells of that type. |
| `count_mat_*.mtx`, `count_mat_*_meta.pkl` | Matrix Market equivalents. The pickle's `obs` and `var` lists give row and column labels in order. |

Regular-mode isoform tables are written regardless of `--csv`/`--mtx`. Each row is an isoform or pooled category.

| File | Counts |
|---|---|
| `isoform_agg.csv` | Phasable reads only. |
| `isoform_agg_balance.csv` | All reads using their EM haplotype probabilities. |
| `isoform_agg_extrap.csv` | All reads, allocating non-phasable reads in each isoform according to its phasable haplotype ratio; gene-level ratio used when unavailable. |
| `isoform_agg_unbalance.csv` | Allocation of non-phasable reads chosen to minimize the ASTU p-value. |
| `isoform_agg_pmax.csv` | Allocation chosen to maximize the ASTU p-value. |
| `ct_{cell_type}_isoform_agg*.csv` | Corresponding counts for one cell type. |

All isoform tables use these columns:

| Column | Description |
|---|---|
| First column (index) | Isoform label. Isoforms below `--chi_min_frac` (default 0.1) on both haplotypes are pooled as `{geneName}_Other`. |
| `hapA`, `hapB` | Weighted read counts on each haplotype, using the allocation specified above. |
| `geneID` | Gene identifier. |
