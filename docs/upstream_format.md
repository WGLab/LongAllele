# Upstream format for adapter developers

Use this specification when writing a converter from another tool's read assignments and transcript annotations. The converted directory must contain a read-assignment TSV, transcript annotation GTF and gene-structure pickle.

Supply one directory per sample, using these filenames:

```text
upstream/
  auxiliary/
    all_read_isoform_exon_mapping.tsv
  reference/
    geneStructureInformationupdated.pkl
    SCOTCH_updated_annotation_filtered.gtf
```

The directory spelling `auxillary` is also accepted. Provide transcript structures matching the read assignments; use the GTF to preserve true exon/junction boundaries.

### Read mapping TSV

Tab-separated, with a header. Extra columns are allowed.

| Column | Meaning |
|---|---|
| `Read` | BAM read name. For names containing `/`, a trailing `_<digits>` suffix is stripped before matching. |
| `geneID` | Gene ID matching the structure pickle and GTF. |
| `geneName` | Gene symbol; use the gene ID if no symbol is available. |
| `geneChr` | Chromosome name matching the BAM. |
| `Cell` | Cell barcode, or a sample-level placeholder for bulk. Must be non-empty. |
| `Umi` | UMI; may be empty. LongAllele does not deduplicate by this column. |
| `Isoform` | Transcript ID matching the structure pickle and GTF, or `{geneName}_uncategorized` for gene-only assignments. |
| `Keep` | Numeric `1` to include the assignment, `0` to exclude it. |

For custom input, supply at most one kept assignment per canonical read and do not assign a kept read to multiple genes. All rows for a gene must agree on its name and chromosome. Transcript IDs must identify the same transcript across all three files and be unique across genes. Isoform labels containing `novel` are treated as novel; labels containing `uncategorized` are excluded from isoform aggregation.

An optional `all_read_isoform_exon_mapping.tsv.geneidx.tsv` accelerates loading. It has no header and four tab-separated fields: `geneID`, byte offset, byte length and number of kept rows. Each gene must occupy one contiguous, non-overlapping block in the mapping TSV; regenerate the index whenever the TSV changes.

### Gene structure pickle

Python dictionary: `geneID → (gene_info, exon_positions, exon_isoform_dict)`.

| Field | Meaning |
|---|---|
| `gene_info` | Dictionary containing `geneID`, `geneName`, `geneChr`, `geneStart`, `geneEnd`. |
| `geneStart`, `geneEnd` | **0-based inclusive** gene bounds: subtract 1 from both GTF bounds. |
| `exon_positions` | Sorted atomic sub-exon intervals, **0-based half-open**: subtract 1 from GTF starts, keep GTF ends. Split the exon union at every transcript exon boundary. |
| `exon_isoform_dict` | Transcript ID → sorted list of indices into `exon_positions`; include each atom fully contained in that transcript's exons. |

### Transcript annotation GTF

Use standard **1-based inclusive** GTF coordinates. Each exon row needs `gene_id` and `transcript_id` matching the TSV and pickle. Include the known and novel transcripts used for assignment. Keep a single `SCOTCH_updated_annotation_filtered*.gtf` in the reference directory.

[Back to other upstream tools](upstream_input_contract.md#other-upstream-tools)
