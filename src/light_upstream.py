
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import pysam

LIGHT_DIR_NAME = 'light_upstream'
MAPPING_BASENAME = 'all_read_isoform_exon_mapping.tsv'
MAPPING_COLUMNS = ['Read', 'geneName', 'geneID', 'geneChr', 'Isoform',
                   'Cell', 'Umi', 'Keep']
_GTF_ATTR_RE = re.compile(r'(\S+) "([^"]*)"')


def _parse_gtf_attributes(field):
    return dict(_GTF_ATTR_RE.findall(field))


def build_gene_structures(gtf_path, logger=None):
    spans = {}
    exons = defaultdict(list)
    with open(gtf_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 9 or f[2] not in ('gene', 'exon'):
                continue
            attrs = _parse_gtf_attributes(f[8])
            gene_id = attrs.get('gene_id')
            if not gene_id:
                raise ValueError(f'GTF row without gene_id: {line[:120]!r}')
            start1, end1 = int(f[3]), int(f[4])
            if start1 > end1:
                raise ValueError(f'GTF start > end for {gene_id}: {line[:120]!r}')
            name = attrs.get('gene_name', gene_id)
            if gene_id in spans and (spans[gene_id][0] != f[0]
                                     or spans[gene_id][1] != f[6]):


                raise ValueError(
                    f'gene_id {gene_id} reused on a different contig/strand '
                    f'({spans[gene_id][0]}{spans[gene_id][1]} vs {f[0]}{f[6]})')
            if f[2] == 'gene':
                if gene_id in spans:
                    spans[gene_id][2] = min(spans[gene_id][2], start1 - 1)
                    spans[gene_id][3] = max(spans[gene_id][3], end1 - 1)
                    spans[gene_id][4] = name
                else:
                    spans[gene_id] = [f[0], f[6], start1 - 1, end1 - 1, name]
            else:
                exons[gene_id].append((start1 - 1, end1))
                if gene_id not in spans:
                    spans[gene_id] = [f[0], f[6], start1 - 1, end1 - 1, name]
                else:
                    spans[gene_id][2] = min(spans[gene_id][2], start1 - 1)
                    spans[gene_id][3] = max(spans[gene_id][3], end1 - 1)
    gsi = {}
    n_exonless = 0
    for gene_id, (chrom, strand, s0, e0, name) in spans.items():
        ivs = sorted(exons.get(gene_id, []))
        if not ivs:
            n_exonless += 1
            ivs = [(s0, e0 + 1)]
        merged = [list(ivs[0])]
        for a, b in ivs[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        gene_info = {'geneID': gene_id, 'geneName': name, 'geneChr': chrom,
                     'geneStrand': strand, 'geneStart': s0, 'geneEnd': e0,
                     'numofExons': len(merged)}
        gsi[gene_id] = (gene_info, [tuple(iv) for iv in merged], {})
    if not gsi:
        raise ValueError(f'no gene/exon rows parsed from {gtf_path}')
    if logger:
        logger.info(f'light_upstream: parsed {len(gsi)} genes from GTF '
                    f'({n_exonless} without exon rows -> whole-span exonInfo)')
    return gsi


class GeneOverlapIndex:

    def __init__(self, gsi):
        per_chrom = defaultdict(list)
        self.exons = {}
        self.meta = {}
        for gene_id, (info, exon_info, _iso) in gsi.items():
            per_chrom[info['geneChr']].append(
                (info['geneStart'], info['geneEnd'] + 1, gene_id))
            starts = np.array([a for a, _ in exon_info], dtype=np.int64)
            ends = np.array([b for _, b in exon_info], dtype=np.int64)
            self.exons[gene_id] = (starts, ends)
            self.meta[gene_id] = info
        self.index = {}
        for chrom, items in per_chrom.items():
            items.sort()
            starts = np.array([s for s, _, _ in items], dtype=np.int64)
            ends = np.array([e for _, e, _ in items], dtype=np.int64)
            ids = np.array([g for _, _, g in items], dtype=object)

            maxend = np.maximum.accumulate(ends)
            self.index[chrom] = (starts, ends, ids, maxend)

    def candidates(self, chrom, start, end):
        entry = self.index.get(chrom)
        if entry is None:
            return []
        starts, ends, ids, maxend = entry
        hi = int(np.searchsorted(starts, end, side='left'))
        lo = int(np.searchsorted(maxend[:hi], start, side='right'))
        return [ids[i] for i in range(lo, hi)
                if starts[i] < end and ends[i] > start]

    def genic_overlap_bp(self, gene_id, blocks):
        info = self.meta[gene_id]
        g_start, g_end = int(info['geneStart']), int(info['geneEnd']) + 1
        total = 0
        for b_start, b_end in blocks:
            lo, hi = max(b_start, g_start), min(b_end, g_end)
            if hi > lo:
                total += hi - lo
        return total

    def exonic_overlap_bp(self, gene_id, blocks):
        starts, ends = self.exons[gene_id]
        total = 0
        for b_start, b_end in blocks:
            lo = int(np.searchsorted(ends, b_start, side='right'))
            hi = int(np.searchsorted(starts, b_end, side='left'))
            for i in range(lo, hi):
                total += min(ends[i], b_end) - max(starts[i], b_start)
        return total


def adjudicate(index, chrom, blocks, min_exonic_bp, ambiguity_ratio,
               assign_by='exonic', stats=None):
    if not blocks:
        return None, 'no_candidate'
    span_start, span_end = blocks[0][0], blocks[-1][1]
    cands = index.candidates(chrom, span_start, span_end)
    if not cands:
        return None, 'no_candidate'


    if stats is not None and len(cands) >= 2:
        stats['reads_with_2plus_candidates'] += 1


    score = (index.genic_overlap_bp if assign_by == 'gene_range'
             else index.exonic_overlap_bp)
    scored = [(score(g, blocks), g) for g in cands]
    scored.sort(key=lambda t: (-t[0], t[1]))
    best_bp, best_gene = scored[0]
    if best_bp < min_exonic_bp:
        return None, 'below_min_exonic'
    second_bp = scored[1][0] if len(scored) > 1 else 0


    if second_bp > 0 and best_bp < ambiguity_ratio * second_bp:
        return None, 'ambiguous'
    if stats is not None and second_bp > 0:
        _record_margin(stats, index, best_gene, best_bp, second_bp, blocks,
                       assign_by)
    return best_gene, 'assigned'


MARGIN_BINS = ((0.5, 'margin_lt50'), (0.8, 'margin_50_80'),
               (0.95, 'margin_80_95'), (1.0, 'margin_95_100'))


def _record_margin(stats, index, best_gene, best_bp, second_bp, blocks,
                   assign_by):
    if best_bp <= 0:
        return
    r = second_bp / best_bp
    key = 'margin_tie' if r >= 1.0 else next(
        k for edge, k in MARGIN_BINS if r < edge)
    stats[key] += 1
    if assign_by != 'gene_range':
        return


    if index.exonic_overlap_bp(best_gene, blocks) == 0:
        stats['assigned_zero_exonic'] += 1
        stats['zeroexon_' + key] += 1


def build_work_units(bam_path, gene_contigs, target_units=None, n_jobs=1,
                     min_window_bp=50_000, logger=None):
    import math
    if target_units is None:
        target_units = max(64, n_jobs * 8)
    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        counts = {s.contig: s.mapped for s in bam.get_index_statistics()}
        lengths = {c: bam.get_reference_length(c) for c in gene_contigs
                   if c in set(bam.references)}
        total = sum(counts.get(c, 0) for c in lengths)
        budget = max(1, total // target_units)
        units = []
        n_counted = n_sliced = 0
        for c in sorted(lengths):
            n = counts.get(c, 0)
            k = max(1, math.ceil(n / budget))
            clen = lengths[c]
            step = math.ceil(clen / k)
            queue = []
            for w in range(k):
                s, e = w * step, min((w + 1) * step, clen)
                if s >= e:
                    continue
                if k == 1:
                    units.append((c, s, e, 1, 0, float(n)))
                    continue
                queue.append((s, e, None))
            while queue:
                s, e, n_true = queue.pop()
                if n_true is None:
                    n_true = bam.count(c, s, e)
                    n_counted += 1
                if n_true <= budget or e - s <= 1:
                    if n_true > budget:
                        m = math.ceil(n_true / budget)
                        n_sliced += 1
                        for kk in range(m):
                            units.append((c, s, e, m, kk, n_true / m))
                    else:
                        units.append((c, s, e, 1, 0, float(n_true)))
                elif e - s > min_window_bp:
                    mid = (s + e) // 2
                    queue.append((s, mid, None))
                    queue.append((mid, e, None))
                else:
                    m = math.ceil(n_true / budget)
                    n_sliced += 1
                    for kk in range(m):
                        units.append((c, s, e, m, kk, n_true / m))
    if logger:
        logger.info(f'build_work_units: {len(units)} units '
                    f'({n_counted} true-counted windows, '
                    f'{n_sliced} point-dense loci hash-sliced)')
    return units


def assign_units(units, n_jobs):
    order = sorted(units, key=lambda u: (-u[5], u[0], u[1], u[3], u[4]))
    loads = [0.0] * n_jobs
    jobs = [[] for _ in range(n_jobs)]
    for c, s, e, m, kk, est in order:
        j = loads.index(min(loads))
        jobs[j].append([c, s, e, m, kk])
        loads[j] += est
    return [sorted(job) for job in jobs]


def collect_assignments(bam_path, index, units, cell_tag='CB', umi_tag='UB',
                        min_mapq=20, min_exonic_bp=30, ambiguity_ratio=2.0,
                        assign_by='exonic',
                        row_sink=None, bulk=False):
    stats = {k: 0 for k in ('alignments_seen', 'skipped_secondary_or_unmapped',
                            'skipped_low_mapq', 'skipped_missing_tags',
                            'assigned', 'no_candidate', 'below_min_exonic',
                            'ambiguous', 'window_dedup_skips',
                            'windows_fetched',


                            'reads_with_2plus_candidates',


                            'assigned_zero_exonic',
                            'margin_tie', 'margin_lt50', 'margin_50_80',
                            'margin_80_95', 'margin_95_100',
                            'zeroexon_margin_tie', 'zeroexon_margin_lt50',
                            'zeroexon_margin_50_80', 'zeroexon_margin_80_95',
                            'zeroexon_margin_95_100')}
    n_rows = 0
    batch = []
    import zlib


    grouped = {}
    for unit in units:
        c, s, e = unit[0], int(unit[1]), int(unit[2])
        m = int(unit[3]) if len(unit) > 3 else 1
        k = int(unit[4]) if len(unit) > 4 else 0
        grouped.setdefault((c, s, e, m), set()).add(k)
    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        bam_contigs = set(bam.references)
        for (contig, w_start, w_end, mod_m), my_ks in sorted(
                grouped.items()):
            if contig not in bam_contigs:
                continue
            stats['windows_fetched'] += 1
            for aln in bam.fetch(contig, w_start, w_end):
                if not (w_start <= aln.reference_start < w_end):
                    stats['window_dedup_skips'] += 1
                    continue
                if mod_m > 1 and \
                        zlib.crc32(aln.query_name.encode()) % mod_m \
                        not in my_ks:
                    stats['window_dedup_skips'] += 1
                    continue
                stats['alignments_seen'] += 1
                if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                    stats['skipped_secondary_or_unmapped'] += 1
                    continue
                if aln.mapping_quality < min_mapq:
                    stats['skipped_low_mapq'] += 1
                    continue
                if bulk:
                    cell, umi = 'bulk', aln.query_name
                else:
                    try:
                        cell = aln.get_tag(cell_tag)
                        umi = aln.get_tag(umi_tag)
                    except KeyError:
                        stats['skipped_missing_tags'] += 1
                        continue
                blocks = aln.get_blocks()
                gene_id, reason = adjudicate(index, contig, blocks,
                                             min_exonic_bp, ambiguity_ratio,
                                             assign_by=assign_by, stats=stats)
                stats[reason] += 1
                if gene_id is None:
                    continue
                info = index.meta[gene_id]
                batch.append([aln.query_name, info['geneName'], gene_id,
                              info['geneChr'],
                              f"{info['geneName']}_lightweight",
                              cell, umi, 1])
                n_rows += 1
                if len(batch) >= 100000 and row_sink is not None:
                    row_sink(batch)
                    batch = []
    if batch and row_sink is not None:
        row_sink(batch)
    return n_rows, stats


def _light_dir(output_folder):
    return os.path.join(output_folder, LIGHT_DIR_NAME)


def _part_path(output_folder, job_index):
    return os.path.join(_light_dir(output_folder), 'auxillary',
                        f'mapping_part{job_index}.tsv')


def _validate_light_params(n_jobs, job_index, min_mapq, min_exonic_bp, assign_by,
                           ambiguity_ratio):
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f'n_jobs must be a positive integer, got {n_jobs}')
    if not 0 <= job_index < n_jobs:
        raise ValueError(f'job_index {job_index} outside [0, {n_jobs})')
    if min_mapq < 0 or min_exonic_bp < 0:
        raise ValueError('min_mapq / min_exonic_bp must be non-negative')
    if assign_by not in ('exonic', 'gene_range'):
        raise ValueError(f"--light_assign_by must be 'exonic' or 'gene_range', "
                         f"got {assign_by!r}")
    if assign_by == 'gene_range' and min_exonic_bp:


        raise ValueError(
            '--light_assign_by gene_range scores overlap with the whole gene '
            f'span, so --light_min_exonic_bp {min_exonic_bp} would discard the '
            'purely intronic reads this mode exists to keep. Pass '
            '--light_min_exonic_bp 0 with gene_range.')
    if assign_by == 'exonic' and min_exonic_bp < 1:


        raise ValueError(
            "--light_assign_by exonic needs an explicit --light_min_exonic_bp "
            ">= 1: at floor 0 a read with ZERO exonic overlap everywhere is "
            "assigned by gene-id tiebreak instead of being discarded. Use 30 "
            "to reproduce FINAL-era runs, 1 for thedecided value.")
    if ambiguity_ratio < 1.0:
        raise ValueError(f'ambiguity_ratio must be >= 1, got {ambiguity_ratio}')


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _run_fingerprint(bam_path, gtf_path, n_jobs, params):
    st = os.stat(bam_path)
    return {'gtf_sha256': _sha256_file(gtf_path),
            'bam': {'path': os.path.abspath(bam_path),
                    'size': st.st_size, 'mtime': int(st.st_mtime)},
            'n_jobs': n_jobs, 'params': params}


def run_light_prep(bam_path, gtf_path, output_folder, n_jobs=1, job_index=0,
                   cell_tag='CB', umi_tag='UB', min_mapq=20, min_exonic_bp=30,
                   assign_by='exonic',
                   ambiguity_ratio=2.0, bulk=False, target_units=None,
                   logger=None):
    _validate_light_params(n_jobs, job_index, min_mapq, min_exonic_bp,
                           assign_by, ambiguity_ratio)
    if target_units is not None and (not isinstance(target_units, int)
                                     or target_units < 1):
        raise ValueError(f'target_units must be a positive integer or None, '
                         f'got {target_units}')
    params = {'assign_by': assign_by,
              'cell_tag': cell_tag, 'umi_tag': umi_tag, 'min_mapq': min_mapq,
              'min_exonic_bp': min_exonic_bp,
              'ambiguity_ratio': ambiguity_ratio, 'bulk': bool(bulk),
              'target_units': target_units}
    fingerprint = _run_fingerprint(bam_path, gtf_path, n_jobs, params)
    gsi = build_gene_structures(gtf_path, logger)
    index = GeneOverlapIndex(gsi)
    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        gene_contigs = sorted({info['geneChr'] for info in index.meta.values()})
        contigs = [c for c in gene_contigs if c in set(bam.references)]
        absent = sorted(set(gene_contigs) - set(contigs))
    if absent and logger:
        logger.info(f'light_prep: {len(absent)} GTF contigs absent from BAM '
                    f'(skipped): {absent[:5]}...')
    all_units = build_work_units(bam_path, contigs, target_units=target_units,
                                 n_jobs=n_jobs, logger=logger)
    my_units = assign_units(all_units, n_jobs)[job_index]
    if logger:
        est = {tuple(u[:5]): u[5] for u in all_units}
        logger.info(f'light_prep job {job_index}: {len(my_units)} work units, '
                    f'~{sum(est.get(tuple(u), 0) for u in my_units):,.0f} '
                    f'reads (est-aware LPT over {len(all_units)} units)')
    part = _part_path(output_folder, job_index)
    os.makedirs(os.path.dirname(part), exist_ok=True)
    tmp = part + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write('\t'.join(MAPPING_COLUMNS) + '\n')

        def sink(rows):
            fh.write('\n'.join('\t'.join(map(str, r)) for r in rows) + '\n')

        n_rows, stats = collect_assignments(
            bam_path, index, my_units, cell_tag=cell_tag, umi_tag=umi_tag,
            min_mapq=min_mapq, min_exonic_bp=min_exonic_bp,
            assign_by=assign_by,
            ambiguity_ratio=ambiguity_ratio, row_sink=sink, bulk=bulk)
    stats_path = part.replace('.tsv', '_stats.json')


    if os.path.isfile(stats_path):
        os.remove(stats_path)
    os.replace(tmp, part)
    stats_tmp = stats_path + '.tmp'
    with open(stats_tmp, 'w') as fh:
        json.dump({'job_index': job_index,
                   'all_units': [list(u) for u in all_units],
                   'units': my_units, 'rows': n_rows,
                   'tsv_sha256': _sha256_file(part),
                   'fingerprint': fingerprint, 'counters': stats}, fh, indent=1)
    os.replace(stats_tmp, stats_path)
    if logger:
        logger.info(f'light_prep job {job_index}/{n_jobs}: {n_rows} assigned '
                    f'rows over {len(my_units)} units '
                    f'({len({u[0] for u in my_units})} contigs); stats: {stats}')
    return part, stats


def run_light_merge(gtf_path, output_folder, n_jobs=1, logger=None):
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f'n_jobs must be a positive integer, got {n_jobs}')
    light = _light_dir(output_folder)
    aux = os.path.join(light, 'auxillary')
    parts = [_part_path(output_folder, j) for j in range(n_jobs)]
    stats_paths = [p.replace('.tsv', '_stats.json') for p in parts]
    missing = [p for pair in zip(parts, stats_paths) for p in pair
               if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f'light_merge: {len(missing)} part/stats files missing '
            f'(first: {missing[0]}) — rerun light_prep for those job indexes')


    part_meta = []
    for j, sp in enumerate(stats_paths):
        with open(sp) as fh:
            meta = json.load(fh)
        for key in ('job_index', 'all_units', 'units', 'fingerprint',
                    'counters', 'rows', 'tsv_sha256'):
            if key not in meta:
                raise ValueError(f'{sp}: missing {key!r} — part written by an '
                                 'older light_prep; rerun light_prep')
        if meta['job_index'] != j:
            raise ValueError(f'{sp}: job_index {meta["job_index"]} != {j}')
        part_meta.append(meta)
    fp0 = part_meta[0]['fingerprint']
    all_units = part_meta[0]['all_units']
    expected_assign = assign_units([tuple(u) for u in all_units], n_jobs)
    for j, meta in enumerate(part_meta):
        if meta['fingerprint'] != fp0:
            raise ValueError(
                f'light_merge: part {j} was produced by a DIFFERENT run '
                f'configuration (BAM/GTF/params/n_jobs mismatch) — rerun '
                f'light_prep for all {n_jobs} jobs of one configuration')
        if meta['all_units'] != all_units:
            raise ValueError(f'light_merge: part {j} disagrees on the work-'
                             'unit universe — mixed-run parts')
        if meta['units'] != expected_assign[j]:
            raise ValueError(f'light_merge: part {j} covers {meta["units"]} '
                             f'but the {n_jobs}-way LPT expects '
                             f'{expected_assign[j]}')
    if fp0['n_jobs'] != n_jobs:
        raise ValueError(f'light_merge: parts were produced with n_jobs='
                         f'{fp0["n_jobs"]}, merge called with {n_jobs}')
    if _sha256_file(gtf_path) != fp0['gtf_sha256']:
        raise ValueError('light_merge: the GTF on disk differs from the one '
                         'the parts were assigned against')


    final_tsv = os.path.join(aux, MAPPING_BASENAME)
    tmp_body = final_tsv + '.body.tmp'
    tmp_tsv = final_tsv + '.tmp'
    n_rows = 0
    import hashlib
    with open(tmp_body, 'wb') as out:
        for p, meta in zip(parts, part_meta):


            h = hashlib.sha256()
            with open(p, 'rb') as fh:
                first = fh.readline()
                h.update(first)
                if first.decode().rstrip('\n') != '\t'.join(MAPPING_COLUMNS):
                    raise ValueError(f'{p}: unexpected header {first!r}')
                for chunk in iter(lambda: fh.read(1 << 20), b''):
                    h.update(chunk)
                    out.write(chunk)
                    n_rows += chunk.count(b'\n')
            if h.hexdigest() != meta['tsv_sha256']:
                raise ValueError(
                    f'light_merge: {p} does not hash to its stats record — '
                    'part TSV and stats are from different generations; rerun '
                    f'light_prep job {meta["job_index"]}')


    import subprocess
    with open(tmp_tsv, 'wb') as out:
        out.write(('\t'.join(MAPPING_COLUMNS) + '\n').encode())
        out.flush()
        gene_col = MAPPING_COLUMNS.index('geneID') + 1
        env = dict(os.environ, LC_ALL='C')
        proc = subprocess.run(
            ['sort', '-t', '\t', f'-k{gene_col},{gene_col}', '-T', aux,
             tmp_body], stdout=out, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f'light_merge: sort by geneID failed '
                           f'(exit {proc.returncode})')
    os.remove(tmp_body)

    tmp_idx = final_tsv + '.geneidx.tmp'
    with open(tmp_tsv, 'rb') as fh, open(tmp_idx, 'w') as idx_out:
        offset = len(fh.readline())
        cur_gene, block_start, block_reads = None, offset, 0
        for line in fh:
            gid = line.split(b'\t')[gene_col - 1].decode()
            if gid != cur_gene:
                if cur_gene is not None:
                    idx_out.write(f'{cur_gene}\t{block_start}\t'
                                  f'{offset - block_start}\t{block_reads}\n')
                cur_gene, block_start, block_reads = gid, offset, 0
            block_reads += 1
            offset += len(line)
        if cur_gene is not None:
            idx_out.write(f'{cur_gene}\t{block_start}\t'
                          f'{offset - block_start}\t{block_reads}\n')
    ref_dir = os.path.join(light, 'reference')
    os.makedirs(ref_dir, exist_ok=True)
    gsi = build_gene_structures(gtf_path, logger)
    pkl_path = os.path.join(ref_dir, 'geneStructureInformation.pkl')
    tmp_pkl = pkl_path + '.tmp'
    with open(tmp_pkl, 'wb') as fh:
        pickle.dump(gsi, fh)

    merged_stats = defaultdict(int)
    for meta in part_meta:
        merged_stats['rows'] += int(meta['rows'])
        for k, v in meta['counters'].items():
            merged_stats[k] += int(v)
    provenance = {
        'upstream': 'gtf_direct',
        'mode': 'lightweight (variant call + phasing; no isoform analysis)',
        'gtf': os.path.abspath(gtf_path),
        'n_genes': len(gsi),
        'mapping_rows_keep1': n_rows,
        'geneidx_sidecar': True,
        'assignment_stats': dict(merged_stats),
        'fingerprint': fp0,
        'note': ('step5 / ASTU unsupported on this directory; attach a real '
                 'SCOTCH or IsoQuant run (QNAME join) for isoform analyses'),
    }
    prov_path = os.path.join(light, 'light_provenance.json')
    tmp_prov = prov_path + '.tmp'
    with open(tmp_prov, 'w') as fh:
        json.dump(provenance, fh, indent=1)


    if os.path.isfile(prov_path):
        os.remove(prov_path)
    os.replace(tmp_pkl, pkl_path)
    os.replace(tmp_tsv, final_tsv)
    os.replace(tmp_idx, final_tsv + '.geneidx.tsv')
    os.replace(tmp_prov, prov_path)
    if logger:
        logger.info(f'light_merge: {n_rows} Keep=1 rows -> {final_tsv}; '
                    f'{len(gsi)} genes -> {pkl_path}')
        logger.info(f'light_merge: pass --scotch_target {light} to steps 1-4')
    return light
