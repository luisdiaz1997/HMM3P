import matplotlib.pyplot as plt
import numpy as np


def plot_chroms(track, *signal):
    '''
        Used to plot different biological signals in different chromosomes
        track: it takes in a pandas dataframe of Hi-C data
        signal: takes in a tuple or multiple parameters of the columns to be plotted, it will do so per chromosome
        
        
        By default the signals plotted will only be at points with no NaNs in the E1 column.
    '''
    plt.figure(figsize = (30, 20))
    i = 1
    for ch in track.chrom.unique()[:-3]: #avoids the last three chroms, X, Y and M
        plt.subplot(11, 2, i)
        plt.title(ch)
        P = track[track.chrom == ch]
        mask = ~P['E1'].isna()
        for s in signal:
            E = P[mask][s].values
            if s != 'E1':
                E/= np.nanmax(E) - np.nanmin(E)
                E = 2*E -1
            plt.plot(E)
            plt.margins(0)
        i+=1
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                        wspace=0.35)
    
    
def compare_chroms(tracks, chrom, start, end, *signal):
    '''
        Used to plot different biological signals in the same chromosome in different cell_lines
        trackd: it takes in a list of pandas dataframes of Hi-C data
        signal: takes in a tuple or multiple parameters of the columns to be plotted, it will do so per chromosome
        
        
        By default the signals plotted will only be at points with no NaNs in the E1 column.
    '''
    plt.figure(figsize = (30, 20))
    i = 1
    for track in tracks:
        plt.subplot(7, 2, i)
        P = track[track.chrom == chrom]
        #mask = ~P['E1'].isna()
        for s in signal:
            E = P[s].values
            '''
            if (s != 'E1') or (s!= 'binary'):
                E/= np.nanmax(E) - np.nanmin(E)
                E = 2*E -1
            '''
            E = E[start:end]
            x = np.linspace(0, len(E) -1, len(E))
            plt.fill_between(x, 0, E, where = E>0)
            plt.fill_between(x, 0, E, where = E<0)
            plt.margins(0)
            plt.subplot(7, 2, i+1)
            plt.plot(E)
            plt.margins(0)
        i+=2
            
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.1)