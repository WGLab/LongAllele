
import argparse
import dataclasses
import gzip
import json
import os
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from enum import Enum


SUPPORTED_ISOQUANT_VERSIONS = ('3.5.0', '3.13.1')


ASSIGNMENT_TYPES_UNIQUE = frozenset({'unique', 'unique_minor_difference'})
ASSIGNMENT_TYPES_AMBIGUOUS = frozenset({'ambiguous'})


ASSIGNMENT_TYPES_INCONSISTENT = frozenset({'inconsistent', 'inconsistent_ambiguous',
                                           'inconsistent_non_intronic'})
ASSIGNMENT_TYPES_DROP = frozenset({'noninformative', 'intergenic', 'suspended'})
KNOWN_ASSIGNMENT_TYPES = (ASSIGNMENT_TYPES_UNIQUE | ASSIGNMENT_TYPES_AMBIGUOUS
                          | ASSIGNMENT_TYPES_INCONSISTENT | ASSIGNMENT_TYPES_DROP)


DEFAULT_DISALLOWED_EVENTS = frozenset({
    'intron_shift', 'exon_misalignment', 'fake_micro_intron_retention',
    'fake_terminal_exon_5', 'fake_terminal_exon_3',
    'terminal_exon_misalignment_5', 'terminal_exon_misalignment_3',
})

REQUIRED_ASSIGNMENT_COLUMNS = ('read_id', 'chr', 'strand', 'isoform_id', 'gene_id',
                               'assignment_type', 'assignment_events',
                               'additional_info')
REQUIRED_MODEL_READS_COLUMNS = ('read_id', 'transcript_id')

UNCATEGORIZED_LABEL = 'uncategorized'
NOVEL_ID_TEMPLATE = 'novelIsoform_{model_id}'


NOVEL_GENE_PREFIX = 'novel_gene_'


EXCLUDED_GENE = '__excluded_gene__'
MAPPING_COLUMNS = ['Read', 'geneName', 'geneID', 'geneChr', 'Isoform',
                   'Cell', 'Umi', 'Keep']
MAPPING_BASENAME = 'all_read_isoform_exon_mapping.tsv'
PICKLE_BASENAME = 'geneStructureInformationupdated.pkl'
GTF_BASENAME = 'SCOTCH_updated_annotation_filtered.gtf'
PROVENANCE_BASENAME = 'isoquant_provenance.json'
_GTF_ATTR_RE = re.compile(r'(\S+) "([^"]*)"')
_VERSION_RE = re.compile(r'IsoQuant version:\s*([0-9][0-9A-Za-z.]*)')


class KeepPolicy(str, Enum):
    STRICT = 'strict'
    DEFAULT = 'default'


class Verdict(str, Enum):
    KEEP_ISOFORM = 'keep_isoform'
    KEEP_UNCATEGORIZED = 'keep_uncat'
    DROP = 'drop'


@dataclasses.dataclass(frozen=True)
class AdapterConfig:
    read_assignments: str
    extended_gtf: str
    reference_gtf: str
    bam: str
    out_dir: str
    transcript_model_reads: str = None
    model_construction_enabled: bool = True
    keep_policy: KeepPolicy = KeepPolicy.DEFAULT
    cell_tag: str = 'CB'
    umi_tag: str = 'UB'
    strip_barcode_suffix: bool = False
    barcode_whitelist: str = None
    bulk_mode: bool = False
    sample_name: str = 'sample'
    emit_dropped: bool = False
    novel_min_support_reads: int = 2
    disallowed_events: frozenset = DEFAULT_DISALLOWED_EVENTS
    force_version: bool = False


def _validated_gtf_bounds(start_1based, end_1based):
    s, e = int(start_1based), int(end_1based)
    if s < 1 or e < s:
        raise ValueError(f'malformed GTF interval ({start_1based}, {end_1based})')
    return s, e


def gtf_exon_to_half_open(start_1based, end_1based):
    s, e = _validated_gtf_bounds(start_1based, end_1based)
    return s - 1, e


def gtf_gene_bounds_to_inclusive(start_1based, end_1based):
    s, e = _validated_gtf_bounds(start_1based, end_1based)
    return s - 1, e - 1


def build_sub_exon_partition(transcript_exons):
    for tid, exons in transcript_exons.items():
        if not exons:
            raise ValueError(f'transcript {tid!r} has no exons (.2)')
        for iv in exons:
            if (len(iv) != 2 or not all(isinstance(x, int) for x in iv)
                    or iv[0] < 0 or iv[1] <= iv[0]):
                raise ValueError(f'transcript {tid!r}: invalid exon interval {iv!r}')

    boundaries = set()
    all_exons = sorted({tuple(e) for exons in transcript_exons.values() for e in exons})
    for s, e in all_exons:
        boundaries.add(s)
        boundaries.add(e)
    merged = []
    for s, e in all_exons:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    cuts = sorted(boundaries)
    atoms = []
    for us, ue in merged:
        inner = [c for c in cuts if us < c < ue]
        edges = [us] + inner + [ue]
        for a, b in zip(edges[:-1], edges[1:]):
            if b > a:
                atoms.append((a, b))
    atoms = sorted(set(atoms))

    exon_isoform_dict = {}
    for tid, exons in transcript_exons.items():
        exons_sorted = sorted(tuple(e) for e in exons)
        indices = [i for i, (a, b) in enumerate(atoms)
                   if any(es <= a and b <= ee for es, ee in exons_sorted)]
        exon_isoform_dict[tid] = indices
    return atoms, exon_isoform_dict


def strip_ensembl_version(feature_id):
    if not isinstance(feature_id, str) or not feature_id:
        raise ValueError(f'invalid feature id: {feature_id!r}')
    if feature_id.startswith(('ENSG', 'ENST')) and '.' in feature_id:
        return feature_id.split('.', 1)[0]
    return feature_id


def _detect_value_collisions(mapping, what):
    values = list(mapping.values())
    if len(set(values)) != len(values):
        counts = Counter(values)
        dupes = sorted(v for v, n in counts.items() if n > 1)
        raise ValueError(f'{what} id collision after normalization: {dupes[:10]} '
                         '(two distinct features collapsing is a hard error,.4)')


def build_gene_normalization_map(native_gene_ids):
    norm = {gid: strip_ensembl_version(gid) for gid in native_gene_ids}
    _detect_value_collisions(norm, 'gene')
    return norm


def build_transcript_normalization_map(reference_transcript_ids,
                                       eligible_novel_model_ids):
    ref = set(reference_transcript_ids)
    nov = set(eligible_novel_model_ids)
    overlap = ref & nov
    if overlap:
        raise ValueError(f'ids in both reference and novel sets: {sorted(overlap)[:10]}')
    norm = {tid: strip_ensembl_version(tid) for tid in ref}
    for mid in nov:
        if not isinstance(mid, str) or not mid:
            raise ValueError(f'invalid novel model id: {mid!r}')
        norm[mid] = NOVEL_ID_TEMPLATE.format(model_id=mid)
    _detect_value_collisions(norm, 'transcript')
    return norm


@dataclasses.dataclass(frozen=True)
class ReadEvidence:
    assignment_type: str = None
    candidate_genes: frozenset = frozenset()
    candidate_isoforms: frozenset = frozenset()
    assignment_events: frozenset = frozenset()
    eligible_models: frozenset = frozenset()
    ineligible_model_genes: frozenset = frozenset()
    has_cell_barcode: bool = False


def _require_single(values, what):
    if len(values) != 1:
        raise ValueError(f'malformed evidence: expected exactly one {what}, '
                         f'got {sorted(values)!r} (.2 — never select arbitrarily)')
    return next(iter(values))


def decide_keep(ev, policy, disallowed_events, single_cell):
    if single_cell and not ev.has_cell_barcode:
        return Verdict.DROP, None, 'no_cell_barcode'

    at = ev.assignment_type
    if at is not None and at not in KNOWN_ASSIGNMENT_TYPES:
        raise ValueError(f'unknown assignment_type {at!r} — IsoQuant vocabulary drift (.0)')
    if at in (ASSIGNMENT_TYPES_UNIQUE | ASSIGNMENT_TYPES_AMBIGUOUS) and not ev.candidate_genes:
        raise ValueError(f'{at} assignment without a gene — malformed evidence')


    models = ev.eligible_models
    if models and len(ev.candidate_genes) == 1:
        (gene,) = ev.candidate_genes
        models = frozenset(m for m in models if m[1] == gene)
    if models:
        model_genes = frozenset(g for _, g in models)
        if len(model_genes) != 1:
            return Verdict.DROP, None, 'novel_models_cross_gene'
        if ev.candidate_genes and ev.candidate_genes != model_genes:


            return Verdict.DROP, None, 'novel_model_gene_disagreement'
        if len(models) > 1:
            return Verdict.KEEP_UNCATEGORIZED, None, 'multi_model'
        (model_id, _), = models
        return Verdict.KEEP_ISOFORM, model_id, 'novel_model'

    if ev.ineligible_model_genes:
        if policy is KeepPolicy.STRICT:
            return Verdict.DROP, None, 'ineligible_model_strict'
        genes = ev.ineligible_model_genes | (ev.candidate_genes if at else frozenset())
        if len(genes) == 1:
            return Verdict.KEEP_UNCATEGORIZED, None, 'ineligible_model'
        return Verdict.DROP, None, 'ineligible_model_gene_conflict'


    if at is None or at in ASSIGNMENT_TYPES_DROP:
        return Verdict.DROP, None, at or 'unassigned'

    if at == 'unique':
        _require_single(ev.candidate_genes, 'gene for unique assignment')
        iso = _require_single(ev.candidate_isoforms, 'isoform for unique assignment')
        return Verdict.KEEP_ISOFORM, iso, 'unique'

    if at == 'unique_minor_difference':
        _require_single(ev.candidate_genes, 'gene for unique_minor_difference')
        iso = _require_single(ev.candidate_isoforms, 'isoform for unique_minor_difference')
        if policy is KeepPolicy.STRICT:
            return Verdict.KEEP_UNCATEGORIZED, None, 'minor_difference_strict'
        if disallowed_events is None:
            return Verdict.KEEP_UNCATEGORIZED, None, 'minor_difference_denylist_unpinned'
        if ev.assignment_events & disallowed_events:
            return Verdict.KEEP_UNCATEGORIZED, None, 'minor_difference_demoted'
        return Verdict.KEEP_ISOFORM, iso, 'unique_minor'

    if at in ASSIGNMENT_TYPES_INCONSISTENT:

        if not ev.candidate_genes:
            return Verdict.DROP, None, 'inconsistent_no_gene'
        if len(ev.candidate_genes) == 1:
            if policy is KeepPolicy.STRICT:
                return Verdict.DROP, None, 'inconsistent_strict'
            return Verdict.KEEP_UNCATEGORIZED, None, 'inconsistent_one_gene'
        return Verdict.DROP, None, 'inconsistent_cross_gene'


    if len(ev.candidate_genes) == 1:
        if policy is KeepPolicy.STRICT:
            return Verdict.DROP, None, 'ambiguous_strict'
        return Verdict.KEEP_UNCATEGORIZED, None, 'ambiguous_one_gene'
    return Verdict.DROP, None, 'ambiguous_cross_gene'


def _open_text(path):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r')


def _canonical_read_name(name):
    if '/' in name:
        return re.sub(r'_\d+$', '', name)
    return name


def _parse_gtf_attributes(field):
    return dict(_GTF_ATTR_RE.findall(field))


def parse_gtf(path):
    genes, transcripts = {}, {}
    with _open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 9:
                raise ValueError(f'{path}: GTF row with {len(f)} fields (9 required): {line[:120]!r}')
            if f[2] not in ('gene', 'transcript', 'exon'):
                continue
            attrs = _parse_gtf_attributes(f[8])
            gid = (attrs.get('gene_id') or '').strip()
            if not gid:
                raise ValueError(f'GTF row without gene_id: {line[:120]!r}')
            s1, e1 = _validated_gtf_bounds(f[3], f[4])
            chrom, strand = f[0], f[6]
            g = genes.get(gid)
            if g is None:
                g = genes[gid] = {'chrom': chrom, 'strand': strand, 'start1': s1,
                                  'end1': e1, 'name': attrs.get('gene_name', gid),
                                  'has_gene_row': False}
            elif g['chrom'] != chrom or g['strand'] != strand:
                raise ValueError(f'gene_id {gid} reused on a different contig/strand '
                                 f'({g["chrom"]}{g["strand"]} vs {chrom}{strand})')
            if f[2] == 'gene':
                g['start1'], g['end1'] = s1, e1
                g['has_gene_row'] = True
                if 'gene_name' in attrs:
                    g['name'] = attrs['gene_name']
                continue
            if s1 < g['start1'] or e1 > g['end1']:


                if g['has_gene_row']:
                    g['bounds_expanded'] = True
                g['start1'], g['end1'] = min(g['start1'], s1), max(g['end1'], e1)
            if g['name'] == gid and 'gene_name' in attrs:
                g['name'] = attrs['gene_name']
            tid = (attrs.get('transcript_id') or '').strip()
            if not tid:
                raise ValueError(f'{f[2]} row without transcript_id: {line[:120]!r}')
            t = transcripts.get(tid)
            if t is None:
                t = transcripts[tid] = {'gene': gid, 'chrom': chrom, 'strand': strand,
                                        'exons': [], 'source': f[1]}
            elif t['gene'] != gid:
                raise ValueError(f'transcript {tid} listed under two genes '
                                 f'({t["gene"]}, {gid}) —.5')
            if f[2] == 'exon':
                t['exons'].append((s1, e1))
    exonless = sorted(t for t, rec in transcripts.items() if not rec['exons'])
    if exonless:
        raise ValueError(f'{len(exonless)} transcript(s) without exon rows '
                         f'(first: {exonless[0]}) —.2')
    return genes, transcripts


def detect_isoquant_version(path):
    with _open_text(path) as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            m = _VERSION_RE.search(line)
            if m:
                return m.group(1)
    return None


def _check_version(version, force):
    if version in SUPPORTED_ISOQUANT_VERSIONS:
        return
    what = f'IsoQuant version {version!r}' if version else 'an IsoQuant table with no version line'
    if force:
        sys.stderr.write(f'[isoquant2longallele] WARNING: running on {what}, which no '
                         f'fixture has been tested against (tested: '
                         f'{", ".join(SUPPORTED_ISOQUANT_VERSIONS)}); --force_version given\n')
        return
    raise SystemExit(f'[isoquant2longallele] {what} is not in the tested set '
                     f'{SUPPORTED_ISOQUANT_VERSIONS} (INPUT_CONTRACT.0). Pass '
                     f'--force_version to run anyway; the header is still checked.')


def _is_header_line(line):
    return line.lstrip('#').split('\t', 1)[0].strip() == 'read_id'


def _header_index(header, required, what):
    cols = header.rstrip('\n').lstrip('#').split('\t')
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f'{what}: header lacks {missing}; found columns {cols} '
                         f'(.0 — columns are resolved by name, never position)')
    return {c: i for i, c in enumerate(cols)}


def _parse_additional_info(text):
    out = {}
    for tok in text.replace(';', ' ').split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            out[k] = v
    return out


def parse_read_assignments(path, config, gene_map, transcript_map):
    counters = Counter()
    reads = {}
    with _open_text(path) as fh:
        header = None
        for line in fh:
            if line.startswith('#') and not _is_header_line(line):
                continue
            if header is None:
                header = _header_index(line, REQUIRED_ASSIGNMENT_COLUMNS, 'read_assignments')
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < len(header):
                raise ValueError(f'read_assignments: short row {line[:120]!r}')
            counters['rows'] += 1
            raw_read = f[header['read_id']].strip()
            read = _canonical_read_name(raw_read)
            at = f[header['assignment_type']].strip()
            if at not in KNOWN_ASSIGNMENT_TYPES:
                raise ValueError(f'unknown assignment_type {at!r} in {path} — IsoQuant '
                                 f'vocabulary drift (.0); known: {sorted(KNOWN_ASSIGNMENT_TYPES)}')
            rec = reads.get(read)
            if rec is None:
                rec = reads[read] = {'assignment_type': at, 'genes': set(), 'isoforms': set(),
                                     'events': set(), 'gene_assignment': None}
            elif rec['assignment_type'] != at:
                raise ValueError(f'read {read!r} carries two assignment_types '
                                 f'({rec["assignment_type"]}, {at}) — malformed table')
            gid = f[header['gene_id']].strip()
            tid = f[header['isoform_id']].strip()
            if gid and gid != '.':
                if gid.startswith(NOVEL_GENE_PREFIX):
                    counters['rows_in_novel_genes'] += 1
                    rec['genes'].add(EXCLUDED_GENE)
                    continue
                if gid not in gene_map:
                    counters['rows_gene_not_in_gtf'] += 1
                    rec['genes'].add(EXCLUDED_GENE)
                    continue
                rec['genes'].add(gene_map[gid])
            if tid and tid != '.':
                if tid not in transcript_map:


                    counters['rows_isoform_not_in_maps'] += 1
                    continue
                rec['isoforms'].add(transcript_map[tid])
            ev = f[header['assignment_events']].strip()
            if ev and ev != '.':
                rec['events'].update(tok.split(':', 1)[0] for tok in ev.split(','))
            info = _parse_additional_info(f[header['additional_info']])
            if 'gene_assignment' in info:
                rec['gene_assignment'] = info['gene_assignment']
    if header is None:
        raise ValueError(f'{path}: no header row found')
    counters['reads'] = len(reads)
    return reads, counters


def parse_transcript_model_reads(path, novel_model_genes, reference_norm_ids,
                                 novel_gene_model_ids=frozenset()):
    counters = Counter()
    read_models = defaultdict(set)
    support = Counter()
    unknown_models = set()
    with _open_text(path) as fh:
        header = None
        for line in fh:
            if line.startswith('#') and not _is_header_line(line):
                continue
            if header is None:
                header = _header_index(line, REQUIRED_MODEL_READS_COLUMNS, 'transcript_model_reads')
                continue
            f = line.rstrip('\n').split('\t')
            counters['rows'] += 1
            mid = f[header['transcript_id']].strip()
            if not mid or mid == '*':
                counters['rows_unassigned'] += 1
                continue
            if mid in novel_model_genes:
                read = _canonical_read_name(f[header['read_id']].strip())
                if mid not in read_models[read]:
                    read_models[read].add(mid)
                    support[mid] += 1
                continue
            if mid in reference_norm_ids or strip_ensembl_version(mid) in reference_norm_ids:
                counters['rows_reference_model'] += 1
                continue
            if mid in novel_gene_model_ids:


                counters['rows_model_in_novel_gene'] += 1
                unknown_models.add(mid)
                continue
            raise ValueError(f'transcript_model_reads names model {mid!r} which the '
                             f'extended GTF does not define (.2 rule 3)')
    if header is None:
        raise ValueError(f'{path}: no header row found')
    counters['reads_with_novel_model'] = len(read_models)
    counters['models_in_novel_genes'] = len(unknown_models)
    return dict(read_models), support, counters


def extract_cell_umi(bam_path, config, wanted=None):
    import pysam
    whitelist = None
    if config.barcode_whitelist:
        with open(config.barcode_whitelist) as fh:
            whitelist = {l.strip() for l in fh if l.strip()}
    out, counters, seen = {}, Counter(), set()
    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        for aln in bam.fetch(until_eof=True):
            if aln.is_secondary or aln.is_supplementary or aln.is_unmapped:
                continue
            read = _canonical_read_name(aln.query_name)
            if wanted is not None and read not in wanted:
                continue
            if read in seen:
                continue
            seen.add(read)
            counters['primary_reads'] += 1
            if config.bulk_mode:
                out[read] = ('bulk', aln.query_name)
                continue


            cell = str(aln.get_tag(config.cell_tag)) if aln.has_tag(config.cell_tag) else ''
            if config.strip_barcode_suffix and '-' in cell:
                cell = cell.rsplit('-', 1)[0]
            if not cell:
                counters['reads_without_cell_barcode'] += 1
                continue
            if whitelist is not None and cell not in whitelist:
                counters['reads_barcode_not_in_whitelist'] += 1
                continue
            umi = str(aln.get_tag(config.umi_tag)) if aln.has_tag(config.umi_tag) else ''
            if not umi:
                counters['reads_without_umi'] += 1
            out[read] = (cell, umi)
    return out, counters


def emit_mapping_tsv(rows, out_dir, gene_col='geneID'):
    aux = os.path.join(out_dir, 'auxiliary')
    os.makedirs(aux, exist_ok=True)
    final = os.path.join(aux, MAPPING_BASENAME)
    seen = set()
    for r in rows:
        if r[7] == 1:
            if r[0] in seen:
                raise ValueError(f'read {r[0]!r} would be kept twice (.2)')
            seen.add(r[0])
    body = final + '.body.tmp'
    with open(body, 'w') as out:
        for r in rows:
            out.write('\t'.join(str(x) for x in r) + '\n')
    tmp = final + '.tmp'
    gene_idx = MAPPING_COLUMNS.index(gene_col) + 1
    with open(tmp, 'wb') as out:
        out.write(('\t'.join(MAPPING_COLUMNS) + '\n').encode())
        out.flush()
        proc = subprocess.run(['sort', '-t', '\t', f'-k{gene_idx},{gene_idx}', '-s',
                               '-T', aux, body], stdout=out,
                              env=dict(os.environ, LC_ALL='C'))
    if proc.returncode != 0:
        raise RuntimeError(f'sort by geneID failed (exit {proc.returncode})')
    os.remove(body)
    idx_tmp = final + '.geneidx.tmp'
    keep_idx = MAPPING_COLUMNS.index('Keep')
    with open(tmp, 'rb') as fh, open(idx_tmp, 'w') as idx:
        offset = len(fh.readline())
        cur, start, n = None, offset, 0
        for line in fh:
            f = line.rstrip(b'\n').split(b'\t')
            gid = f[gene_idx - 1].decode()
            if gid != cur:
                if cur is not None:
                    idx.write(f'{cur}\t{start}\t{offset - start}\t{n}\n')
                cur, start, n = gid, offset, 0
            n += f[keep_idx] == b'1'
            offset += len(line)
        if cur is not None:
            idx.write(f'{cur}\t{start}\t{offset - start}\t{n}\n')
    os.replace(tmp, final)
    os.replace(idx_tmp, final + '.geneidx.tsv')
    return final


def emit_gene_structure_pickle(gene_records, out_dir):
    ref = os.path.join(out_dir, 'reference')
    os.makedirs(ref, exist_ok=True)
    path = os.path.join(ref, PICKLE_BASENAME)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(gene_records, fh)
    os.replace(tmp, path)
    return path


def emit_rewritten_gtf(transcripts, genes, gene_map, transcript_map, out_dir):
    ref = os.path.join(out_dir, 'reference')
    os.makedirs(ref, exist_ok=True)
    path = os.path.join(ref, GTF_BASENAME)
    tmp = path + '.tmp'
    by_gene = defaultdict(list)
    for tid, t in transcripts.items():
        if tid in transcript_map and t['gene'] in gene_map:
            by_gene[t['gene']].append(tid)
    with open(tmp, 'w') as out:
        out.write('# rewritten by isoquant2longallele (INPUT_CONTRACT v0.3.5): '
                  'ids normalized, novel genes and unsupported models omitted\n')
        for gid in sorted(by_gene, key=lambda g: (genes[g]['chrom'], genes[g]['start1'], g)):
            g = genes[gid]
            ngid = gene_map[gid]
            out.write(f'{g["chrom"]}\tisoquant2longallele\tgene\t{g["start1"]}\t{g["end1"]}'
                      f'\t.\t{g["strand"]}\t.\tgene_id "{ngid}"; gene_name "{g["name"]}";\n')
            for tid in sorted(by_gene[gid], key=lambda t: transcript_map[t]):
                t = transcripts[tid]
                ntid = transcript_map[tid]
                s1 = min(e[0] for e in t['exons']); e1 = max(e[1] for e in t['exons'])
                out.write(f'{t["chrom"]}\t{t["source"]}\ttranscript\t{s1}\t{e1}\t.\t'
                          f'{t["strand"]}\t.\tgene_id "{ngid}"; transcript_id "{ntid}"; '
                          f'gene_name "{g["name"]}";\n')
                for es, ee in sorted(t['exons']):
                    out.write(f'{t["chrom"]}\t{t["source"]}\texon\t{es}\t{ee}\t.\t'
                              f'{t["strand"]}\t.\tgene_id "{ngid}"; transcript_id "{ntid}"; '
                              f'gene_name "{g["name"]}";\n')
    os.replace(tmp, path)
    return path


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def emit_provenance(config, counters, out_dir, isoquant_version=None,
                    normalization_map_paths=None, junction_source='true-gtf'):
    payload = {
        'upstream': 'isoquant',
        'adapter': 'isoquant2longallele',
        'isoquant_version': isoquant_version,
        'tested_isoquant_versions': list(SUPPORTED_ISOQUANT_VERSIONS),
        'config': _jsonable(dataclasses.asdict(config)),
        'counters': _jsonable(counters),
        'normalization_maps': _jsonable(normalization_map_paths or {}),
        'junction_source': junction_source,
        'high_artifact_mode_supported': False,
        'isoform_labels': {'uncategorized': UNCATEGORIZED_LABEL,
                           'novel': NOVEL_ID_TEMPLATE},
    }
    path = os.path.join(out_dir, PROVENANCE_BASENAME)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def _log(logger, msg):
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def run_adapter(config, logger=None):
    out_dir = config.out_dir
    os.makedirs(out_dir, exist_ok=True)
    prov_path = os.path.join(out_dir, PROVENANCE_BASENAME)
    if os.path.isfile(prov_path):
        os.remove(prov_path)
    counters = Counter()

    version = detect_isoquant_version(config.read_assignments)
    _check_version(version, config.force_version)
    _log(logger, f'[isoquant2longallele] IsoQuant version: {version or "unknown"}; '
                 f'policy={config.keep_policy.value}; novel_min_support_reads='
                 f'{config.novel_min_support_reads}; bulk={config.bulk_mode}')


    ref_genes, ref_transcripts = parse_gtf(config.reference_gtf)
    ext_genes, ext_transcripts = parse_gtf(config.extended_gtf)
    ref_tids_stripped = {strip_ensembl_version(t): t for t in ref_transcripts}


    reference_native, novel_native, novel_gene_models = [], [], set()
    for tid, t in ext_transcripts.items():
        if t['gene'].startswith(NOVEL_GENE_PREFIX):
            counters['transcripts_in_novel_genes'] += 1
            novel_gene_models.add(tid)
            continue
        if tid in ref_transcripts or strip_ensembl_version(tid) in ref_tids_stripped:
            reference_native.append(tid)
        else:
            novel_native.append(tid)
    counters['reference_transcripts_in_extended_gtf'] = len(reference_native)
    counters['novel_models_in_extended_gtf'] = len(novel_native)
    novel_model_genes = {tid: ext_transcripts[tid]['gene'] for tid in novel_native}


    read_models, support, mcount = {}, Counter(), Counter()
    if config.model_construction_enabled:
        if not config.transcript_model_reads:
            raise SystemExit('model construction is enabled but no transcript_model_reads '
                             'file was given (.2 rule 4); IsoQuant writes it with '
                             '--large_output read2transcripts, or pass --no_model_construction')
        read_models, support, mcount = parse_transcript_model_reads(
            config.transcript_model_reads, novel_model_genes,
            set(ref_tids_stripped) | set(ref_transcripts), novel_gene_models)
        counters.update({f'model_reads_{k}': v for k, v in mcount.items()})
    eligible = {m for m in novel_native if support[m] >= config.novel_min_support_reads}
    ineligible = set(novel_native) - eligible
    counters['novel_models_eligible'] = len(eligible)
    counters['novel_models_ineligible'] = len(ineligible)


    gene_ids = [g for g in ext_genes if not g.startswith(NOVEL_GENE_PREFIX)]
    gene_map = build_gene_normalization_map(gene_ids)
    transcript_map = build_transcript_normalization_map(reference_native, eligible)
    gene_names = {gene_map[g]: ext_genes[g]['name'] for g in gene_ids}
    for g in gene_ids:
        if g in ref_genes:
            gene_names[gene_map[g]] = ref_genes[g]['name']
    gene_chrom = {gene_map[g]: ext_genes[g]['chrom'] for g in gene_ids}


    reads, acount = parse_read_assignments(config.read_assignments, config,
                                           gene_map, transcript_map)
    counters.update({f'assignments_{k}': v for k, v in acount.items()})


    wanted = set(reads) | set(read_models)
    tags, bcount = extract_cell_umi(config.bam, config, wanted=wanted)
    counters.update({f'bam_{k}': v for k, v in bcount.items()})


    model_gene_norm = {m: gene_map[g] for m, g in novel_model_genes.items() if g in gene_map}
    rows, reasons = [], Counter()
    single_cell = not config.bulk_mode
    for read in sorted(wanted):
        rec = reads.get(read)
        models = read_models.get(read, set())
        elig = frozenset((m, model_gene_norm[m]) for m in models
                         if m in eligible and m in model_gene_norm)
        inelig_genes = frozenset(model_gene_norm[m] for m in models
                                 if m in ineligible and m in model_gene_norm)
        ev = ReadEvidence(
            assignment_type=rec['assignment_type'] if rec else None,
            candidate_genes=frozenset(rec['genes']) if rec else frozenset(),
            candidate_isoforms=frozenset(rec['isoforms']) if rec else frozenset(),
            assignment_events=frozenset(rec['events']) if rec else frozenset(),
            eligible_models=elig, ineligible_model_genes=inelig_genes,
            has_cell_barcode=read in tags)
        if ev.candidate_genes == frozenset({EXCLUDED_GENE}):


            verdict, iso_native, reason = Verdict.DROP, None, 'gene_excluded'
        else:
            verdict, iso_native, reason = decide_keep(ev, config.keep_policy,
                                                      config.disallowed_events, single_cell)
        reasons[reason] += 1
        kept_genes = (ev.candidate_genes - {EXCLUDED_GENE}) or {g for _, g in elig} or inelig_genes
        if verdict is Verdict.DROP:
            if config.emit_dropped and kept_genes:
                gene = sorted(kept_genes)[0]
                rows.append((read, gene_names[gene], gene, gene_chrom[gene], '',
                             *(tags.get(read, ('', ''))), 0))
            continue
        if len(kept_genes) != 1:
            raise RuntimeError(f'read {read!r} kept with {sorted(kept_genes)} genes — '
                               f'decide_keep invariant broken (reason {reason})')
        (gene,) = kept_genes
        if verdict is Verdict.KEEP_ISOFORM:
            iso = transcript_map[iso_native] if iso_native in transcript_map else iso_native
        else:
            iso = UNCATEGORIZED_LABEL
        cell, umi = tags[read]
        rows.append((read, gene_names[gene], gene, gene_chrom[gene], iso, cell, umi, 1))
    counters['keep_reasons'] = dict(reasons)
    counters['rows_keep1'] = sum(1 for r in rows if r[7] == 1)
    counters['rows_keep0_emitted'] = sum(1 for r in rows if r[7] == 0)


    mapping_path = emit_mapping_tsv(rows, out_dir)
    per_gene = defaultdict(dict)
    for tid in list(reference_native) + sorted(eligible):
        t = ext_transcripts[tid]
        per_gene[t['gene']][transcript_map[tid]] = [gtf_exon_to_half_open(s, e)
                                                     for s, e in t['exons']]
    gene_records = {}
    for g in gene_ids:
        ng = gene_map[g]
        info = ext_genes[g]
        s0, e0 = gtf_gene_bounds_to_inclusive(info['start1'], info['end1'])
        if per_gene.get(g):
            atoms, iso = build_sub_exon_partition(per_gene[g])
        else:
            atoms, iso = [(s0, e0 + 1)], {}
        gene_records[ng] = ({'geneID': ng, 'geneName': gene_names[ng], 'geneChr': info['chrom'],
                             'geneStrand': info['strand'], 'geneStart': s0, 'geneEnd': e0,
                             'numofExons': len(atoms)}, atoms, iso)
    counters['genes_in_pickle'] = len(gene_records)
    counters['genes_bounds_expanded_to_cover_exons'] = sum(
        1 for g in gene_ids if ext_genes[g].get('bounds_expanded'))
    pkl_path = emit_gene_structure_pickle(gene_records, out_dir)
    gtf_path = emit_rewritten_gtf(ext_transcripts, ext_genes, gene_map, transcript_map, out_dir)
    maps = {}
    for name, m in (('gene_id_map.tsv', gene_map), ('transcript_id_map.tsv', transcript_map)):
        p = os.path.join(out_dir, name)
        with open(p, 'w') as fh:
            fh.write('normalized_id\tnative_id\n')
            for native, norm in sorted(m.items(), key=lambda kv: kv[1]):
                fh.write(f'{norm}\t{native}\n')
        maps[name] = p
    prov = emit_provenance(config, counters, out_dir, isoquant_version=version,
                           normalization_map_paths=maps)
    _log(logger, f'[isoquant2longallele] {counters["rows_keep1"]} Keep=1 rows -> {mapping_path}')
    _log(logger, f'[isoquant2longallele] {len(gene_records)} genes -> {pkl_path}; GTF -> {gtf_path}')
    _log(logger, f'[isoquant2longallele] keep reasons: {dict(reasons)}')
    _log(logger, f'[isoquant2longallele] provenance -> {prov}; pass --scotch_target {out_dir}')
    return out_dir


def find_isoquant_outputs(isoquant_dir, prefix=None):
    cands = [isoquant_dir]
    if prefix:
        cands.insert(0, os.path.join(isoquant_dir, prefix))
    found = {}
    for d in cands:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if name.endswith(('.read_assignments.tsv', '.read_assignments.tsv.gz')):
                found.setdefault('read_assignments', p)
            elif name.endswith(('.transcript_model_reads.tsv', '.transcript_model_reads.tsv.gz')):
                found.setdefault('transcript_model_reads', p)
            elif name.endswith('.extended_annotation.gtf'):
                found.setdefault('extended_gtf', p)
        if 'read_assignments' in found:
            break
    return found


def build_parser():
    p = argparse.ArgumentParser(
        description='Convert IsoQuant output to the LongAllele upstream input contract '
                    '(INPUT_CONTRACT.md v0.3). Emits the SCOTCH filenames.')
    p.add_argument('--isoquant_dir', default=None,
                   help='IsoQuant output directory (the three tables are found by suffix; '
                        'explicit --read_assignments / --extended_gtf / '
                        '--transcript_model_reads override)')
    p.add_argument('--isoquant_prefix', default=None, help='IsoQuant --prefix (subdirectory)')
    p.add_argument('--read_assignments', default=None)
    p.add_argument('--extended_gtf', default=None)
    p.add_argument('--transcript_model_reads', default=None)
    p.add_argument('--reference_gtf', required=True)
    p.add_argument('--bam', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--no_model_construction', action='store_true')
    p.add_argument('--keep_policy', choices=[k.value for k in KeepPolicy],
                   default=KeepPolicy.DEFAULT.value)
    p.add_argument('--cell_tag', default='CB')
    p.add_argument('--umi_tag', default='UB')
    p.add_argument('--strip_barcode_suffix', action='store_true')
    p.add_argument('--barcode_whitelist', default=None)
    p.add_argument('--bulk_mode', action='store_true')
    p.add_argument('--sample_name', default='sample')
    p.add_argument('--emit_dropped', action='store_true')
    p.add_argument('--novel_min_support_reads', type=int, default=2)
    p.add_argument('--force_version', action='store_true',
                   help='run on an IsoQuant version outside the tested set (warns)')
    return p


def config_from_args(args):
    found = {}
    if args.isoquant_dir:
        found = find_isoquant_outputs(args.isoquant_dir, args.isoquant_prefix)
    ra = args.read_assignments or found.get('read_assignments')
    eg = args.extended_gtf or found.get('extended_gtf')
    mr = args.transcript_model_reads or found.get('transcript_model_reads')
    if not ra or not eg:
        raise SystemExit('need --read_assignments and --extended_gtf (or an --isoquant_dir '
                         f'containing them); found: {found}')
    return AdapterConfig(
        read_assignments=ra, extended_gtf=eg, reference_gtf=args.reference_gtf,
        bam=args.bam, out_dir=args.out_dir, transcript_model_reads=mr,
        model_construction_enabled=not args.no_model_construction,
        keep_policy=KeepPolicy(args.keep_policy), cell_tag=args.cell_tag,
        umi_tag=args.umi_tag, strip_barcode_suffix=args.strip_barcode_suffix,
        barcode_whitelist=args.barcode_whitelist, bulk_mode=args.bulk_mode,
        sample_name=args.sample_name, emit_dropped=args.emit_dropped,
        novel_min_support_reads=args.novel_min_support_reads,
        force_version=args.force_version)


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_adapter(config_from_args(args))


if __name__ == '__main__':
    main()
