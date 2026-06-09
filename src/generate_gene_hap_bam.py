import argparse
import glob
import os
import re
import subprocess
from collections import defaultdict

import pandas as pd
import pysam


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate haplotype-split BAM files per cell type for a single gene."
    )
    parser.add_argument("--gene_name", required=True, help="Gene symbol, e.g. KMT2C")
    parser.add_argument(
        "--gene_id",
        default=None,
        help="Optional gene ID, e.g. ENSG00000055609",
    )
    parser.add_argument("--scotch_target", required=True, help="SCOTCH output directory")
    parser.add_argument(
        "--bam_path",
        required=True,
        help="Input BAM path or directory of chromosome-split BAMs",
    )
    parser.add_argument(
        "--longallele_path",
        required=True,
        help="Longallele output directory",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix used for LongAllele output folder naming (e.g. 'snvfilter', 'LongAllele'). "
             "Determines paths like snv_hap_{prefix}/ and summary_statistics_{prefix}/; "
             "the script also checks haplotype_summary/ for flattened summary outputs. "
             "If omitted, searches the unprefixed directories.",
    )
    parser.add_argument(
        "--celltype_csv",
        required=True,
        help="CSV with Cell, CellType, and optionally Sample columns",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--sample_id",
        required=True,
        help="Sample ID for SCOTCH layout and optional celltype filtering",
    )
    return parser.parse_args()


def _prefixed_dir_names(base, prefix):
    """Return candidate directory names: prefixed first, then bare."""
    names = []
    if prefix is not None:
        names.append(f"{base}_{prefix}")
    names.append(base)
    return names


def _snv_hap_dir_names(prefix):
    return _prefixed_dir_names("snv_hap", prefix)


def _summary_dir_names(prefix):
    return ["haplotype_summary"] + _prefixed_dir_names("summary_statistics", prefix)


def _read_hap_summary_files(root, prefix):
    for dirname in _snv_hap_dir_names(prefix) + ["haplotype_summary"]:
        yield os.path.join(root, dirname, "read_hap_map.csv")


def _snv_hap_map_files(root, prefix):
    for dirname in _snv_hap_dir_names(prefix) + ["haplotype_summary"]:
        yield os.path.join(root, dirname, "snv_hap_map.csv")


def _has_longallele_outputs(path, prefix):
    expected_dirs = (
        _snv_hap_dir_names(prefix)
        + _summary_dir_names(prefix)
        + ["count_mat_hap", "count_matrix_hap"]
    )
    return any(os.path.isdir(os.path.join(path, dirname)) for dirname in expected_dirs)


def iter_longallele_roots(longallele_path, prefix):
    seen = set()
    candidates = []

    if os.path.isdir(longallele_path):
        candidates.append(longallele_path)
        for entry in sorted(os.listdir(longallele_path)):
            child = os.path.join(longallele_path, entry)
            if os.path.isdir(child):
                candidates.append(child)

    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_longallele_outputs(candidate, prefix):
            yield candidate

    if longallele_path not in seen:
        yield longallele_path


def resolve_read_hap_path(longallele_path, gene_name, gene_id=None, prefix=None):
    search_dirs = []
    seen = set()

    for root in iter_longallele_roots(longallele_path, prefix):
        for snv_dir_name in _snv_hap_dir_names(prefix):
            for subdir_name in ["all_genes_separate", "all_genes_seperate"]:
                search_dir = os.path.join(root, snv_dir_name, subdir_name)
                if os.path.isdir(search_dir) and search_dir not in seen:
                    search_dirs.append(search_dir)
                    seen.add(search_dir)

    for search_dir in search_dirs:
        if gene_id is not None:
            direct = os.path.join(search_dir, f"{gene_name}_{gene_id}_read_hap.csv")
            if os.path.isfile(direct):
                return direct

        matches = sorted(glob.glob(os.path.join(search_dir, f"{gene_name}_*_read_hap.csv")))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(
                f"Warning: multiple read_hap files found for {gene_name}; using {matches[0]}"
            )
            return matches[0]

    for root in iter_longallele_roots(longallele_path, prefix):
        for candidate in _read_hap_summary_files(root, prefix):
            if os.path.isfile(candidate):
                return candidate

    best_root = next(iter_longallele_roots(longallele_path, prefix), longallele_path)
    snv_dir = _snv_hap_dir_names(prefix)[0]
    return os.path.join(
        best_root,
        snv_dir,
        "all_genes_separate",
        f"{gene_name}_{gene_id or 'UNKNOWN'}_read_hap.csv",
    )


def resolve_scotch_tsv(scotch_target, sample_id):
    candidates = [
        os.path.join(
            scotch_target,
            f"samples/{sample_id}/auxillary/all_read_isoform_exon_mapping.tsv",
        ),
        os.path.join(scotch_target, "auxillary/all_read_isoform_exon_mapping.tsv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find SCOTCH read-isoform mapping TSV. Tried:\n"
        + "\n".join(candidates)
    )


def resolve_bam_path(bam_path, chrom):
    if os.path.isfile(bam_path):
        return bam_path
    if not os.path.isdir(bam_path):
        raise FileNotFoundError(f"BAM path does not exist: {bam_path}")

    candidates = []
    for fname in os.listdir(bam_path):
        if not fname.endswith(".bam"):
            continue
        full_path = os.path.join(bam_path, fname)
        if f".{chrom}." in fname or fname.endswith(f".{chrom}.bam"):
            candidates.append(full_path)

    if not candidates:
        for fname in os.listdir(bam_path):
            if not fname.endswith(".bam"):
                continue
            full_path = os.path.join(bam_path, fname)
            stem_parts = fname.split(".")
            if len(stem_parts) >= 3 and stem_parts[-2] == chrom:
                candidates.append(full_path)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find chromosome-specific BAM for {chrom} in directory: {bam_path}"
        )

    return sorted(candidates)[0]


def sanitize_label(label):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(label))


def load_read_hap_assignments(read_hap_path, gene_name, gene_id=None):
    if not os.path.exists(read_hap_path):
        raise FileNotFoundError(f"Read haplotype CSV not found: {read_hap_path}")

    df = pd.read_csv(read_hap_path)
    required_columns = {"Read", "reads_phasable", "hat_I"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Read haplotype CSV missing required columns {sorted(missing)}: {read_hap_path}"
        )

    basename = os.path.basename(read_hap_path)
    if basename == "read_hap_map.csv":
        if gene_id is not None and "geneID" in df.columns:
            df = df[df["geneID"].astype(str) == str(gene_id)].copy()
        elif "geneName" in df.columns:
            df = df[df["geneName"].astype(str) == str(gene_name)].copy()
        elif gene_id is not None:
            raise ValueError(
                f"Read haplotype map does not contain geneID column needed for {gene_id}: {read_hap_path}"
            )
        else:
            raise ValueError(
                f"Read haplotype map must contain geneName or geneID columns: {read_hap_path}"
            )

    df = df[df["reads_phasable"] == 1].copy()
    return {
        str(row["Read"]): ("hapA" if float(row["hat_I"]) > 0.5 else "hapB")
        for _, row in df.iterrows()
    }


def load_cell_types(celltype_csv, sample_id):
    df = pd.read_csv(celltype_csv)
    required_columns = {"Cell", "CellType"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Cell type CSV missing required columns {sorted(missing)}: {celltype_csv}"
        )

    if "Sample" in df.columns:
        df = df[df["Sample"].astype(str) == str(sample_id)].copy()

    df = df.dropna(subset=["Cell", "CellType"]).copy()
    if df.empty:
        raise ValueError(f"No usable cell type rows found in {celltype_csv}")

    df["Cell"] = df["Cell"].astype(str)
    df["CellType"] = df["CellType"].astype(str)
    barcode_to_ct = dict(zip(df["Cell"], df["CellType"]))
    cell_types = sorted(df["CellType"].unique().tolist())
    return barcode_to_ct, cell_types


def load_scotch_read_barcodes(scotch_tsv_path, gene_name, gene_id=None):
    read_to_barcode = {}
    total_rows = 0
    matched_rows = 0

    for chunk in pd.read_csv(scotch_tsv_path, sep="\t", chunksize=100000):
        total_rows += len(chunk)

        if "Read" not in chunk.columns or "Cell" not in chunk.columns:
            raise ValueError(
                f"SCOTCH TSV must contain Read and Cell columns: {scotch_tsv_path}"
            )

        if "Keep" in chunk.columns:
            chunk = chunk[chunk["Keep"] == 1]

        if gene_id is not None and "geneID" in chunk.columns:
            chunk = chunk[chunk["geneID"].astype(str) == str(gene_id)]
        elif "geneName" in chunk.columns:
            chunk = chunk[chunk["geneName"].astype(str) == str(gene_name)]
        elif gene_id is not None and "geneID" not in chunk.columns:
            raise ValueError(
                f"SCOTCH TSV does not contain geneID column needed for {gene_id}: {scotch_tsv_path}"
            )
        else:
            raise ValueError(
                f"SCOTCH TSV must contain geneName or geneID columns: {scotch_tsv_path}"
            )

        matched_rows += len(chunk)
        if chunk.empty:
            continue

        chunk = chunk.dropna(subset=["Read", "Cell"])
        read_to_barcode.update(
            dict(zip(chunk["Read"].astype(str), chunk["Cell"].astype(str)))
        )

    print(
        f"Loaded {len(read_to_barcode)} SCOTCH read->cell mappings "
        f"from {matched_rows} matched rows out of {total_rows}"
    )
    return read_to_barcode


def get_gene_region(
    longallele_path, gene_name, gene_id=None, prefix=None, padding=10000
):
    snv_map_path = None
    for root in iter_longallele_roots(longallele_path, prefix):
        for candidate in _snv_hap_map_files(root, prefix):
            if os.path.exists(candidate):
                snv_map_path = candidate
                break
        if snv_map_path is not None:
            break

    if snv_map_path is None:
        best_root = next(iter_longallele_roots(longallele_path, prefix), longallele_path)
        expected_dir = _snv_hap_dir_names(prefix)[0]
        raise FileNotFoundError(
            "SNV haplotype map not found. Tried roots under "
            f"{longallele_path}, expected e.g. "
            f"{os.path.join(best_root, expected_dir, 'snv_hap_map.csv')}"
        )

    snv_map = pd.read_csv(snv_map_path)
    required_columns = {"chrom", "pos"}
    missing = required_columns.difference(snv_map.columns)
    if missing:
        raise ValueError(
            f"SNV haplotype map missing required columns {sorted(missing)}: {snv_map_path}"
        )

    if gene_id is not None and "geneID" in snv_map.columns:
        gene_snvs = snv_map[snv_map["geneID"].astype(str) == str(gene_id)].copy()
    elif "geneName" in snv_map.columns:
        gene_snvs = snv_map[snv_map["geneName"].astype(str) == str(gene_name)].copy()
    elif gene_id is not None:
        raise ValueError(
            f"SNV haplotype map does not contain geneID column needed for {gene_id}: {snv_map_path}"
        )
    else:
        raise ValueError(
            f"SNV haplotype map must contain geneName or geneID columns: {snv_map_path}"
        )

    gene_snvs = gene_snvs.dropna(subset=["chrom", "pos"])
    if gene_snvs.empty:
        gene_desc = f"{gene_name} ({gene_id})" if gene_id else gene_name
        raise ValueError(f"No SNVs found for gene {gene_desc} in {snv_map_path}")

    gene_chr = str(gene_snvs["chrom"].iloc[0])
    gene_start = max(0, int(gene_snvs["pos"].min()) - padding)
    gene_end = int(gene_snvs["pos"].max()) + padding
    return gene_chr, gene_start, gene_end


def create_writers(bam_in, output_dir, gene_name, sample_id, cell_types):
    writers = {}
    out_paths = {}

    for ct in cell_types + ["Bulk"]:
        ct_label = sanitize_label(ct)
        for hap in ("hapA", "hapB"):
            key = (ct, hap)
            path = os.path.join(
                output_dir, f"{gene_name}_{sample_id}_{ct_label}_{hap}.bam"
            )
            writers[key] = pysam.AlignmentFile(path, "wb", template=bam_in)
            out_paths[key] = path

    return writers, out_paths


def sort_index_and_cleanup(out_paths, counts):
    for key, path in out_paths.items():
        if counts[key] == 0:
            if os.path.exists(path):
                os.remove(path)
            continue

        sorted_path = path.replace(".bam", "_sorted.bam")
        subprocess.run(["samtools", "sort", "-o", sorted_path, path], check=True)
        os.replace(sorted_path, path)
        subprocess.run(["samtools", "index", path], check=True)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    read_hap_path = resolve_read_hap_path(
        args.longallele_path, args.gene_name, args.gene_id, args.prefix
    )
    scotch_tsv_path = resolve_scotch_tsv(args.scotch_target, args.sample_id)

    print(f"Loading read haplotypes: {read_hap_path}")
    read_to_hap = load_read_hap_assignments(
        read_hap_path, args.gene_name, args.gene_id
    )
    print(f"Loaded {len(read_to_hap)} phasable read haplotype assignments")

    print(f"Loading SCOTCH read->cell mappings: {scotch_tsv_path}")
    read_to_barcode = load_scotch_read_barcodes(
        scotch_tsv_path, args.gene_name, args.gene_id
    )

    print(f"Loading cell types: {args.celltype_csv}")
    barcode_to_ct, cell_types = load_cell_types(args.celltype_csv, args.sample_id)
    print(f"Loaded {len(cell_types)} cell types")

    gene_chr, gene_start, gene_end = get_gene_region(
        args.longallele_path, args.gene_name, args.gene_id, args.prefix
    )
    print(f"Gene region: {gene_chr}:{gene_start}-{gene_end}")

    bam_file = resolve_bam_path(args.bam_path, gene_chr)
    print(f"Opening BAM: {bam_file}")
    bam_in = pysam.AlignmentFile(bam_file, "rb")

    writers, out_paths = create_writers(
        bam_in, args.output_dir, args.gene_name, args.sample_id, cell_types
    )
    counts = defaultdict(int)
    processed_reads = 0

    try:
        for read in bam_in.fetch(gene_chr, gene_start, gene_end):
            processed_reads += 1
            query_name = read.query_name
            hap = read_to_hap.get(query_name)
            if hap is None:
                continue

            writers[("Bulk", hap)].write(read)
            counts[("Bulk", hap)] += 1

            barcode = read_to_barcode.get(query_name)
            if barcode is None:
                continue

            cell_type = barcode_to_ct.get(barcode)
            if cell_type is None:
                continue

            key = (cell_type, hap)
            if key in writers:
                writers[key].write(read)
                counts[key] += 1
    finally:
        bam_in.close()
        for writer in writers.values():
            writer.close()

    sort_index_and_cleanup(out_paths, counts)

    print(f"Reads processed in region: {processed_reads}")
    print("Reads written:")
    for key in sorted(out_paths):
        print(f"  {key[0]} {key[1]}: {counts[key]}")
    print(f"Done. Output written to {args.output_dir}")


if __name__ == "__main__":
    main()
