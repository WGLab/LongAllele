# Output schema — full column dictionary

This document lists every column produced by LongAllele's Step 5 downstream analysis (`gene_snv.csv` and `event_snv.csv`). For a quick orientation and the file structure of Step 3–4 outputs, see the main [README](../README.md#outputs).

**Coordinate system:** all genomic positions use **0-based** coordinates.

## `gene_snv.csv` — SNV-centric table

One row per confident phased SNV per gene per cell type.

| Column | Description |
|--------|-------------|
| `Sample`, `CellType` | Sample and cell type identifiers |
| `geneID`, `geneName`, `geneChr` | Gene identifiers |
| `n_reads`, `n_reads_phasable`, `gene_n_snvs`, `gene_n_snvs_called` | Gene-level read counts (`n_reads`, `n_reads_phasable`) and SNV counts — `gene_n_snvs` = total candidate SNVs, `gene_n_snvs_called` = SNVs actually called / used in phasing. |
| `gene_alpha_hat`, `gene_alpha_hat_low`, `gene_alpha_hat_high` | Minor haplotype allelic balance (EM estimate + CI) |
| `gene_alpha_hat_major`, `gene_alpha_hat_major_low`, `gene_alpha_hat_major_high` | Major haplotype allelic balance (1 − minor) |
| `gene_major_hap`, `gene_minor_hap` | Haplotype labels (A or B) |
| `gene_p_value`, `gene_p_value_adj` | Gene-level ASE significance test on phased reads (BH-adjusted). **Raw test significance, NOT the final call** — a small p alone over-calls ASE; the final call is `ASE_call` below, which additionally requires the allelic-balance CI to exclude 0.5. |
| `ASE_call` | **Final ASE call** (three-category): `1` significant (`gene_p_value_adj ≤ 0.05` **and** `gene_alpha_hat_high < 0.5`), `-1` not significant (`gene_p_value_adj > 0.05`), `0` inconclusive (p significant but the α CI overlaps 0.5 → insufficient power). |
| `dominant_isoform_overall` | Most expressed isoform across both haplotypes |
| `top_isoform_hap_major`, `top_isoform_hap_minor` | Top isoform on each haplotype |
| `top_isoform_hap_major_frac`, `top_isoform_hap_minor_frac` | Fraction of hap reads from top isoform |
| `isoform_p_value`, `isoform_p_value_adj` | Gene-level ASTU significance test on phased reads (point estimate, BH-adjusted). **Raw test significance, NOT the final call** — the final call is `ASTU_call` below, derived from the CI bounds. |
| `isoform_p_value_high`, `isoform_p_value_low`, `isoform_p_value_adj_high`, `isoform_p_value_adj_low` | ASTU significance evaluated at the high / low bounds of the isoform-balance CI (BH-adjusted). Used to derive `ASTU_call`. |
| `ASTU_call` | **Final ASTU call** (three-category): `1` significant (`isoform_p_value_adj_high ≤ 0.05`), `-1` not significant (`isoform_p_value_adj_low > 0.05`), `0` inconclusive (low bound significant but high bound not → insufficient power). |
| `shrinkage_k` | Shrinkage constant added to the major / minor haplotype read counts when computing `es_ase` / `es_astu` (effect-size regularization). Note: this regularizes the effect-size ratio, it does **not** shape the allelic-balance CI. |
| `es_ase` | ASE effect size: log2(major / minor hap reads) |
| `es_astu` | ASTU effect size: log2(dominant isoform major / minor fraction) |
| `astu_source` | `bulk`, `ct_specific`, or `bulk_fallback` |
| `snvID` | Stable SNV key (`chr:pos:ref:alt`) |
| `snv_pos`, `snv_ref`, `snv_alt` | SNV coordinates and alleles |
| `snv_depth_bulk`, `snv_alt_count_bulk`, `snv_alt_frac_bulk` | SNV read support from variant calling (bulk pileup) |
| `h_A`, `hat_Z_prob_revised` | Haplotype-A frequency and phasing confidence |
| `snv_hap` | Haplotype carrying the alt allele (A or B) |
| `snv_on_minor_hap` | Whether SNV alt allele is on the minor haplotype |
| `snv_expr_direction` | `higher_gene_expression` or `lower_gene_expression` |
| `snv_es_ase_signed` | Signed ASE effect from SNV alt allele perspective |
| `dominant_isoform_pref_hap` | Haplotype with higher dominant isoform usage |
| `snv_astu_direction` | `+` if dominant isoform increased on SNV hap, `−` otherwise |
| `snv_es_astu_signed` | Signed ASTU effect from SNV alt allele perspective |

## `event_snv.csv` — Event-centric table

One row per haplotype-associated event per cell type, duplicated per linked SNV. Events with no nearby SNV retain one row with SNV fields as `NaN`.

| Column | Description |
|--------|-------------|
| `Sample`, `CellType` | Sample and cell type identifiers |
| `geneID`, `geneName`, `geneChr` | Gene identifiers |
| `gene_major_hap`, `es_ase`, `es_astu` | Gene context (duplicated for self-containment) |
| `ASE_call`, `ASTU_call` | Final ASE / ASTU calls (3-category) for the gene, duplicated from `gene_snv.csv` — see that table for the rules. |
| `dominant_isoform_overall`, `top_isoform_hap_major`, `top_isoform_hap_minor` | Isoform context |
| `eventID` | Stable event key (`event_type:start-end`) |
| `event_type` | `exon` or `junction` |
| `event_start`, `event_end` | Event genomic coordinates |
| `w_A_present`, `w_A_absent`, `w_B_present`, `w_B_absent` | Weighted haplotype read counts (isoform-inferred event membership: a read assigned by SCOTCH to a full-length isoform contributes regardless of whether its alignment actually covers the event region). |
| `obs_hapA_include`, `obs_hapA_skip`, `obs_hapA_unobserved` | Haplotype-A weighted read counts from raw BAM CIGAR observation: read alignment includes the event (≥ `min(20bp, 30% × exon_length)` overlap for exons; CIGAR N op edges within ±20 bp of the junction for junctions), splices over it (read spans both flanks but skips this event), or fails to physically cover the event region (truncated read). |
| `obs_hapB_include`, `obs_hapB_skip`, `obs_hapB_unobserved` | Same three categories for haplotype-B. The three columns sum to per-read EM weight totals in the joined pool; the `w_A_*` / `w_B_*` columns above sum to the same total only when reads map to a single isoform each (SCOTCH multi-isoform assignments are double-counted there but not in `obs_*`). |
| `obs_chi2`, `obs_p_value`, `obs_p_value_adj` | Chi-square test on the 2×2 `[[hapA_include, hapA_skip], [hapB_include, hapB_skip]]` table — the `unobserved` column is dropped so truncated reads do not contribute to the test. The test runs whenever the table has non-zero row and column margins (a single zero cell is kept — complete inclusion/skip on one hap is the strongest allele-specific signal and is fully testable); only an all-zero row/column makes it `insufficient_data`. `obs_p_value_adj` is the within-gene BH FDR across all events whose `obs_test_type == 'chi2_hap_event'`; rows with NaN `obs_p_value` (no_bam / insufficient_data) get NaN adj. |
| `obs_test_type` | `chi2_hap_event` (test ran), `insufficient_data` (2×2 sum < `min_reads` or a whole row/column margin is 0), or `no_bam` (per-gene `read_blocks.pkl` not present; other `obs_*` fields are `None`). The pkl cache is written by step1.5 (`--task step1_5` per-sample SLURM array, one task per BAM, followed by `--task step1_5_merge` single-task union); decoupled from step1 to avoid the petagene-concurrency wall regression observed when block collection ran inline in step1's joint-variant pileup. |
| `event_inclusion_frac_A`, `event_inclusion_frac_B` | Inclusion fraction per haplotype |
| `event_pref_hap` | Haplotype with higher event inclusion |
| `event_pref_major_minor` | `major` or `minor` relative to gene expression |
| `event_chi2`, `event_p_value`, `event_p_value_adj` | Haplotype-event association test (within-gene FDR), SCOTCH isoform-inferred event membership. Compare against `obs_*` for a sensitivity check against read truncation. |
| `has_linked_snv` | Whether a nearby confident SNV is linked |
| `linked_snv_count` | Number of nearby SNVs linked to this event |
| `is_nearest_snv_for_event` | Whether this is the closest linked SNV |
| `snvID`, `snv_pos`, `snv_ref`, `snv_alt` | Linked SNV identity (`NaN` if none) |
| `snv_hap`, `h_A`, `hat_Z_prob_revised` | SNV phasing info (`NaN` if none) |
| `exonic_distance`, `genomic_distance` | Distance from SNV to event boundary |
| `snv_expr_direction`, `snv_astu_direction` | SNV regulatory interpretation |
| `snv_event_direction` | `promotes_event` or `reduces_event` |
| `raw_validation_available` | Whether raw read validation was performed |
| `raw_ref_present`, `raw_ref_absent`, `raw_alt_present`, `raw_alt_absent` | Raw BAM allele × event counts |
| `raw_total_reads` | Total raw reads in contingency table |
| `raw_chi2`, `raw_p_value`, `raw_p_value_adj` | Raw-read validation statistics. `raw_chi2` is populated only when `raw_test_type == 'chi2_cross_event'`; for intra-event SNVs, `raw_p_value` carries a binomial test result and `raw_chi2` is `None`. `raw_p_value_adj` is the within-gene BH FDR across all `(event, SNV)` raw tests in the gene; rows with NaN `raw_p_value` get NaN adj. |
| `raw_test_type` | `chi2_cross_event` (default 2×2 chi-square against event-absent reads) or `binomial_intra_event` (SNV inside the exon event — fallback binomial test on ref vs alt counts under p=0.5). `None` when raw validation is unavailable. |

## Interpretation — two complementary event tests

`event_snv.csv` carries two complementary statistical tests for linking variants to splicing / isoform events. They answer different questions and have different power regimes; the table below summarizes both.

| | Haplotype-event test (isoform-inferred) | Haplotype-event test (CIGAR-observed) | SNV-event test (raw) |
|---|---|---|---|
| **Column prefix** | `event_chi2`, `event_p_value`, `event_p_value_adj` | `obs_chi2`, `obs_p_value`, `obs_p_value_adj` | `raw_chi2`, `raw_p_value`, `raw_p_value_adj` |
| **Question** | Does this event differ between haplotype A and B? | Same question, restricted to reads that physically observed the event | Does a specific SNV allele associate with this event? |
| **Haplotype source** | EM posterior (`hat_I`), integrates ALL phasing SNVs in the gene | EM posterior (`hat_I`), same pool | Hard allele call at ONE specific SNV position |
| **Event membership** | SCOTCH isoform assignment — full-length isoform inferred from read mapping, applies to all reads | Raw BAM CIGAR — read alignment blocks and N gaps directly tested against event coordinates; truncated reads marked `unobserved` and excluded from the chi-square | SCOTCH isoform assignment, conditioned on SNV-covering reads |
| **Strength** | Detects haplotype effects even when the causal variant is distant from the event; uses all reads that SCOTCH could assign | Robust to read-truncation bias in SCOTCH's "ideal isoform completion"; reflects only what alignment actually shows | Directly tests a specific variant, unaffected by EM phasing uncertainty |
| **Weakness / caveat** | A truncated read assigned by SCOTCH to a full-length isoform is counted whether or not it physically reached the event region | Throws away reads that did not span the event, so power drops on short fragments or 3'-degraded RNA | Requires the SNV to lie within or near the event; same isoform-inference assumption as Task 4a |

Diagnostic pattern when comparing `event_*` vs `obs_*`: large `obs_*_unobserved` weight (relative to the `event_*` totals) flags events where SCOTCH's full-length isoform inference is doing heavy lifting and the raw alignment does not directly confirm the membership.
