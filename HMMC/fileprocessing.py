import glob
import pandas as pd
import re
import numpy as np
import cooler
import bioframe


def get_files(filedir, ext = 'mcool'):
    files = glob.glob(filedir + '/**/*'+ext, recursive=True)
    return files


def cools_df(cools, resolution = 100000):
    df = pd.DataFrame([np.array(re.split("[\_\./\-]+", cool))[[-8, -5]].tolist() + [cool]
                       for cool in cools],
                      columns=['cell_line', 'assembly', 'path'])
    c_list = list()
    for i in range(len(df)):
        try:
            cooler_obj = cooler.Cooler(df.iloc[i]['path'] + '::/resolutions/' + str(resolution))
            c_list.append(cooler_obj)
        except:
            c_list.append(float("NaN"))

    df['cooler'] = pd.Series(c_list)

    return df



def beds_df(beds):
    
    df = pd.DataFrame([ bed.split('/')[5:] + [bed]  for bed in beds],
                        columns=[
                            'cell_line','assay', 'file_format', 'output_type',
                            'assembly', 'file_status', 'target', 'biosample_treatment',
                            'lab', 'replicate', 'file_name', 'file_location'])
    
    df['cell_line'] = df['cell_line'].apply(lambda a: a.replace('-', ''))
    return df


def get_genecov(df):
    genecov_dict = dict()
    for assembly in df.assembly.unique():
        bins = df[df.assembly == assembly]['cooler'].iloc[0].bins()[:]
        genecov = bioframe.tools.frac_gene_coverage(bins, assembly)
        genecov_dict[assembly] = genecov['gene_coverage']

    return genecov_dict

def get_gc(df, fasta_path):
    gc_dict = dict()
    fasta_records = bioframe.load_fasta(fasta_path)
    for assembly in df.assembly.unique():
        bins = df[df.assembly == assembly]['cooler'].iloc[0].bins()[:]
        gc = bioframe.genomeops.frac_gc(bins, fasta_records)
        gc_dict[assembly] = gc

    return gc_dict
