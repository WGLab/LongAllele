import os
import pandas as pd
import re
import pysam
import subprocess

def sub_gtf(gene_name, gtf_path, out_path = None):
    gtf = pd.read_csv(gtf_path, sep='\t', comment='#', header=None)
    gtf.columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    filtered_gtf = gtf[gtf['attribute'].str.contains(f'gene_name "{gene_name}"')]
    transcript_ids = filtered_gtf['attribute'].apply(
        lambda x: re.search(r'transcript_id "([^"]+)"', x).group(1) if re.search(r'transcript_id "([^"]+)"',
                                                                                 x) else None
    ).dropna().unique()
    gene_chr = filtered_gtf['seqname'].iloc[0]
    gene_start = filtered_gtf['start'].min()
    gene_end = filtered_gtf['end'].max()
    gene_strand = filtered_gtf['strand'].iloc[0]
    if out_path is not None:
        filtered_gtf.to_csv(out_path, sep='\t', index=False, header=False, quoting=3)
    return filtered_gtf, transcript_ids, gene_chr, gene_start, gene_end, gene_strand


def generate_bam_by_hap(geneName, bamFile, scotch_target, longallele_path, selected_isoform = None):
    out_folder = os.path.join(longallele_path, "bam_by_hap")
    os.makedirs(out_folder, exist_ok=True)
    read_hap_path = os.path.join(longallele_path, 'snv_hap_snvfilter/all_genes_seperate')
    read_hap_path_gene = [os.path.join(read_hap_path, f) for f in os.listdir(read_hap_path) if f.startswith(geneName)][
        0]
    read_hap_df = pd.read_csv(read_hap_path_gene)
    read_hap_df = read_hap_df[read_hap_df.reads_phasable==1]
    gtf_path = os.path.join(scotch_target, 'reference/SCOTCH_updated_annotation_filtered.gtf')
    filtered_gtf, transcript_ids, gene_chr, gene_start, gene_end, gene_strand = sub_gtf(geneName, gtf_path, out_path=None)
    transcript_ids = filtered_gtf['attribute'].apply(
        lambda x: re.search(r'transcript_id "([^"]+)"', x).group(1) if re.search(r'transcript_id "([^"]+)"',
                                                                                 x) else None)
    transcript_ids_known = [t for t in transcript_ids.unique().tolist()[1:] if t.startswith('ENST')]
    transcript_ids_novel = [t for t in transcript_ids.unique().tolist()[1:] if t.startswith('novel') and t in selected_isoform]
    selected_isoform = transcript_ids_known + transcript_ids_novel
    filtered_gtf = pd.concat([filtered_gtf.iloc[[0]], filtered_gtf[transcript_ids.isin(selected_isoform)]], ignore_index=True)
    # write bam file
    if os.path.isfile(bamFile)==False: #bamFile is a folder
        bamFile_name = [f for f in os.listdir(bamFile) if f.endswith('.bam') and '.'+gene_chr+'.' in f]
        bamFile = os.path.join(bamFile,bamFile_name[0])
    bamFilePysam = pysam.Samfile(bamFile, "rb")
    #-------------------write bam files------------------#
    bam_path_hapA=os.path.join(out_folder, f'{geneName}_hapA.bam')
    bam_path_hapB = os.path.join(out_folder,f'{geneName}_hapB.bam')
    outA = pysam.AlignmentFile(bam_path_hapA, "wb", template=bamFilePysam)
    outB = pysam.AlignmentFile(bam_path_hapB, "wb", template=bamFilePysam)
    reads_hapA = set(read_hap_df[read_hap_df.hat_I > 0.5].Read.tolist())
    reads_hapB = set(read_hap_df[read_hap_df.hat_I <= 0.5].Read.tolist())
    reads = bamFilePysam.fetch(gene_chr, gene_start, gene_end)
    for read in reads:
        qn = read.query_name
        inA, inB = qn in reads_hapA, qn in reads_hapB
        if inA:
            outA.write(read)
        if inB:
            outB.write(read)
    outA.close()
    outB.close()
    bamFilePysam.close()
    for p in (bam_path_hapA, bam_path_hapB):
        sorted_p = p.replace(".bam", "_sorted.bam")
        subprocess.run(["samtools", "sort", "-o", sorted_p, p], check=True)
        os.replace(sorted_p, p)
        subprocess.run(["samtools", "index", p], check=True)
    filtered_gtf.to_csv(os.path.join(out_folder, f'{geneName}_SCOTCH_filtered.gtf'), sep='\t', index=False, header=False,quoting=3)




