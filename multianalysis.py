import cooltools.eigdecomp as eigdecomp
import cooler
import fileprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



'''
    All the tools in this package with the purpose of analysing multiple datasets go here
    Tools for multianalysis of eigenvectors, HMMs and Bed files
'''

def get_eigs(df, genecov_dict=None, n= 3):
    '''
    
        df: dataframe of multiple coolers. More info in preprocessing.py
        
        genecove_dict: dictionary of genenomic coverage for each genomic assembly,
                        it's recommended to pass this as a parameter since this it takes a lot of time and memory to call the function.
                        
        n: amount of eigenvectors to be passed.
        
        
        Gets the eigenvectors of multiple celllines into two lists vals and tracks, 
        vals contains a list of dataframes of the eigenvalues,
        tracks contains a list of dataframes containing eigenvectors from cooltools
        
        
        By default, the eigenvectors will be taken from a balanced matrix,
        ignoring the first 4 diagonals and the percent being clipped will be after the 99th percentile.
        Change this manually if needed.
    '''
    
    if genecov_dict == None:
        genecov_dict = fileprocessing.get_genecov(df)
        
    vals, tracks = [], []
    for i in range(len(df)):
        c = df.cooler.iloc[i]
        bins = c.bins()[:]
        bins['gene_count'] = genecov_dict[df.assembly.iloc[i]]
        regions = [(chrom, 0, c.chromsizes[chrom]) for chrom in c.chromnames]
        cis_vals, cis_eigs = eigdecomp.cooler_cis_eig(
                c, 
                bins, 
                regions = regions,
                n_eigs=n, 
                balance=True,
                phasing_track_col='gene_count',
                ignore_diags = 4, clip_percentile=99)
        tracks.append(cis_eigs)
        vals.append(cis_vals)
    return vals, tracks


def multi_analyze(df, tracks, Es = None):
    
    if Es == None:
        Es = tracks[0].columns.tolist()[5:]
    results = dict()
    for E in Es:
        compared_df = tracks[0].iloc[:, :5]
        for i in range(len(df)):
            cell_line = df.iloc[i].cell_line
            compared_df[cell_line] = tracks[i][E]
        results[E] =compared_df
    return results


def get_correlations(resdict, w = 10, l = 10):
    
    dict_size = len(resdict.keys())
    plt.figure(1, figsize  =(w, l))
    #plt.suptitle(title)
    
    
    
    
    i = 1
    for k, v in resdict.items():
        plt.subplot(dict_size//5 + 1, np.max(  [dict_size%5, 5] ),i)
        corr = np.abs(v.iloc[:,5:].corr())
        #print(corr)
        plt.imshow(corr,cmap='YlOrRd', vmin=0, vmax=1)
        plt.colorbar();
        ticks = np.arange(0,len(corr.columns),1)
        plt.xticks(ticks, labels =corr.columns,  rotation=90)
        plt.yticks(ticks, labels = corr.columns)
        for z in range(len(corr)):
            for j in range(len(corr)):
                text = plt.text(j, z, np.round(corr.iloc[z, j], 2),
                       ha="center", va="center", color="w")
        plt.title(k)
        i+=1
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                    wspace=0.35)
    
    
def get_chrom_correlations(resdict, chroms):
    dict_size = len(resdict.keys())
    
    for ch in chroms:
        plt.figure(figsize  =(25, 4))
        plt.suptitle(ch)
        i = 1
        for k, v in resdict.items():
            plt.subplot(1, 5,i)
            region = v[v.chrom == ch]
            corr = np.abs(region.iloc[:,5:].corr())
            #print(corr)
            plt.imshow(corr,cmap='YlOrRd', vmin=0, vmax=1)
            plt.colorbar();
            ticks = np.arange(0,len(corr.columns),1)
            plt.xticks(ticks, labels =corr.columns,  rotation=90)
            plt.yticks(ticks, labels = corr.columns)
            for z in range(len(corr)):
                for j in range(len(corr)):
                    text = plt.text(j, z, np.round(corr.iloc[z, j], 2),
                           ha="center", va="center", color="w")
            plt.title(k)
            i+=1
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                    wspace=0.35)