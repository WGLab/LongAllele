# LongAllele demo — PBMC (10 genes, chr22)

A small, self-contained example that runs the full LongAllele pipeline
(step1 → step5) end to end in a few minutes. It subsets 10 genes on chr22 from a
single public PBMC ONT long-read sample, letting you clone the repo and reproduce
a complete run — variant calling, EM haplotyping, and downstream ASE/ASTU testing —
without any external data.

Data source: PBMC ONT long-read RNA-seq from
[Xu et al., *Nature Communications* 2026](https://www.nature.com/articles/s41467-026-72665-5).

## Run

```bash
cd examples/pbmc_demo
bash run_demo.sh          # writes ./demo_output/
```

Compare `demo_output/` against the shipped reference in `expected_output/`.
The same run through the config-driven driver: `RUNNER=local` in a config
pointing at these inputs, then `bash ../../longallele.sh my_demo.sh`.

`expected_output/` was produced with the pinned versions in `requirements.txt`.
Compare it numerically rather than byte for byte: step 3 runs the EM across all
available cores, so the order of the parallel reductions shifts the last
iteration slightly and a few values move by ~1e-4 (p-values by ~3e-3) between
runs on the same machine. The ASE/ASTU calls in the table below are unaffected —
those are what the demo is meant to reproduce. Older library versions stay
within the same calls but move individual numbers considerably more.

## Inputs

| Path | What it is |
|------|-----------|
| `demo.bam` (+ `.bai`) | Aligned long reads for the 10 demo genes |
| `scotch_target/reference/geneStructureInformationupdated.pkl` | SCOTCH gene/isoform structure (the pickle LongAllele loads) |
| `scotch_target/auxiliary/all_read_isoform_exon_mapping.tsv` | Read → gene/isoform map from SCOTCH (the legacy `auxillary` spelling from older SCOTCH versions is also accepted) |
| `ref/chr22.fa.gz` (+ `.fai` `.gzi`) | Reference sequence, chr22 only |
| `sample7_celltype.csv` | Cell barcode → cell type (for per-cell-type results) |

## Outputs (`demo_output/`)

| File | What it is |
|------|-----------|
| `variant_align1/variants_by_gene/*_snvs.csv` | step1: per-gene SNV candidates from pileup |
| `em_input/*` | step2: per-gene read × SNV matrices fed to the EM |
| `snv_hap_demo/snv_hap_map.csv` | step3: each SNV's haplotype assignment |
| `snv_hap_demo/read_hap_map.csv` | step3: each read's haplotype assignment |
| `summary_statistics_demo/summary_statistics.csv` | step4: per-gene, per-cell-type haplotype summary + p-values |
| `count_matrix_hap_demo/*` | step4: haplotype-resolved isoform count matrices |
| `downstream_demo/gene_snv.csv` | step5: per-SNV ASE / ASTU effect sizes |
| `downstream_demo/event_snv.csv` | step5: haplotype ↔ exon/junction event associations |
| `downstream_demo/actv_results.csv` | step5: allelic cell-type variability (ACTV) per gene — on because a cell-type file is given |

`expected_output/` ships the five headline CSVs
(`summary_statistics.csv`, `snv_hap_map.csv`, `gene_snv.csv`, `event_snv.csv`,
`actv_results.csv`) as a reference to diff against. They were regenerated on
2026-10-02 with `--platform ont-cdna` (the shipped preset: depth 10, 2 alt
reads, the ONT cDNA classifier).

## Expected result

Every output carries a `CellType` column (Bulk + B / Monocyte / NK / T cells).
At the Bulk level the calls are (✓ = called, ~ = test significant but the
interval does not support a call, – = not significant, n/a = no p-value):

| Gene | ASE | ASTU |    | Gene | ASE | ASTU |
|------|-----|------|----|------|-----|------|
| SELENOM  | ✓ | ✓ | | NDUFA6 | – | – |
| APOBEC3G | ✓ | ✓ | | SAMM50 | – | – |
| NCAPH2   | ~ | ~ | | SFI1   | n/a | n/a |
| SNAP29   | ~ | – | | TCF20  | n/a | n/a |
| UBE2L3   | n/a | n/a | | TPST2  | n/a | n/a |

Four genes get no p-value: under the preset they call 24–57 SNVs that fall into
8–17 phase blocks, and LongAllele reports NA for genes with more than 7 blocks
rather than a p-value that was not checked under every block orientation (see
`n_phase_blocks` in `summary_statistics.csv`). ACTV (`actv_results.csv`) flags
NDUFA6 and APOBEC3G as having an ASE effect that differs between cell types.

(Significance is FDR-adjusted within this 10-gene set, so p-values differ from a
genome-wide run.)
