import numpy as np

import bioframe as bf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import rc
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd


# defining colors for plotting segmentation maps

names = {'HMM3': ['B', 'M', 'A', 'N','N', 'N'], 
         'HMM3P':['B', 'Mbb', 'M', 'Maa', 'A', 'N'],
         'binary':['B', 'A', 'N','N','N', 'N'],
         'HMM5':['B', 'Mbb', 'M', 'Maa', 'A', 'N'],
        }

nan_color = 5
pallete={ 'HMM3' :   np.array([0, 2, 4, nan_color, nan_color, nan_color]), 
          'HMM3P' :  np.array([0, 1, 2, 3, 4, nan_color]),
         'HMM5' :  np.array([0, 1, 2, 3, 4, nan_color]),
          'binary' : np.array([0, 4, nan_color, nan_color, nan_color, nan_color])
         }

colordict={'B':'#74add1',
           'Mbb':'#e0f3f8',
           'M':'#ffffbf',
           'Maa':'#fdae61',
           'A':'#f46d43',
           'N':'#f8f8f8'}

def hex_to_rgb(hex_val):
    hex_val = hex_val.lstrip('#')
    hlen = len(hex_val)
    return tuple(int(hex_val[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))
   
cmap = np.array(list([ np.array(hex_to_rgb(colordict[key]))/255 for key in colordict.keys()]))


def track_to_bed(hmm_track, annotation_type):
    
    annotation_track = hmm_track.copy()[annotation_type].values.reshape(-1)    
    annotation_track[np.isnan(annotation_track)] = nan_color
    annotation_track = annotation_track.astype(int)
    names_track = np.array(names[annotation_type])[annotation_track]
    annotation_track = pallete[annotation_type][annotation_track]
    mat_rgb = (cmap[annotation_track] * 255).astype(int).astype(str)
    str_rgb = [_ for _ in map(','.join, zip(mat_rgb[:, 0], mat_rgb[:, 1], mat_rgb[:, 2]))]

    bed_df = hmm_track.copy().iloc[:, 0:3]
    bed_df['name'] = names_track
    bed_df['score'] = 0
    bed_df['strand'] = '-'
    bed_df['thickStart'] = bed_df.start
    bed_df['thickEnd'] = bed_df.end
    bed_df['rgb'] = str_rgb
    return bed_df


# utilities for plotting segmentations as colored heatmaps

def track_to_mat(hmm_track, region, annotation_type, pallete, heatmap_width = 5, horizontal=True):
    y_sig = bf.select(hmm_track, region)[annotation_type].values
    mat = np.tile(y_sig.reshape(1,-1), (heatmap_width, 1))
    mat[np.isnan(mat)] = nan_color
    mat = mat.astype(int)
    mat_c = pallete[annotation_type][mat]
    if not horizontal:
        mat_c = mat_c.T
    return mat_c


def cmap_from_bed(bed_df, rgb = True):
    unique_colors = bed_df.rgb.unique().reshape(-1)
    if rgb:
        unique_colors= pd.Series(unique_colors, dtype=str)
        cmap = np.array([color for color in unique_colors.str.split(',').values], dtype='float')/255
    else:
        cmap = np.array(list([ np.array(hex_to_rgb(color))/255 for color in unique_colors]))
    
    return cmap
    

def bed_to_mat(bed_df, region, heatmap_width = 5, horizontal=True):
    
    unique_states = bed_df.name.unique().reshape(-1)
    enc = OrdinalEncoder(categories=[unique_states])
    enc.fit(unique_states.reshape(-1, 1))
 
    y_sig = bf.select(bed_df, region).name.values.reshape(-1,1)
    
    
    mat = np.tile(enc.transform(y_sig), (1, heatmap_width))
    mat = mat.astype(int)
    if horizontal:
        mat = mat.T
        
    return mat

def plot_track(hmm_track, region, annotation_type, pallete):
    plt.figure(figsize=(figure_width, figure_width/10))
    mat_c = track_to_mat(hmm_track, region, annotation_type, pallete, horizontal=True)
    im = plt.imshow(mat_c,cmap=matplotlib.colors.ListedColormap(cmap), vmin=0, vmax=5, aspect='auto')
    
    # handling the legend
    values = pallete[annotation_type]
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label=names[annotation_type][i] ) 
               for i in range(len(np.unique(names[annotation_type])) )]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axis('off');
    
    
def plot_bedtrack(bed_track, region, figure_width=10, rgb=True):
    plt.figure(figsize=(figure_width, figure_width/10))
    cmap = cmap_from_bed(bed_track, rgb) 
    mat = bed_to_mat(bed_track, region)
    im = plt.imshow(mat,cmap=matplotlib.colors.ListedColormap(cmap), vmin=0, vmax=len(cmap), aspect='auto')
    
    # handling the legend
    names = bed_track.name.unique().reshape(-1)
    order = np.argsort(names)
    names = list(names[order])
    colors = [im.cmap(im.norm(i)) for i in order]
    patches = [ mpatches.Patch(color=color, label=name) for name, color in zip(names, colors)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axis('off');

def names_to_nums(name_sig):
    mask = name_sig=='N'
    name_sig = OrdinalEncoder().fit_transform(name_sig.reshape(-1, 1)).reshape(-1)
    name_sig[mask]=np.nan
    return name_sig

    
def plotmap(hic_cooler, bed_track, region1, region2, rgb=True, figure_width=5, plotColorbar=True):
    
    chrm1, start1, end1 = bf.parse_region(region1)
    chrm2, start2, end2 = bf.parse_region(region2)
    width = (end1-start1)
    height = (end2-start2)
    fig= plt.figure(figsize = (figure_width, figure_width * height/width))
    gs = gridspec.GridSpec(20, int(20 * height/width), figure=fig, wspace=2, hspace=2)

    # plot vertical segmentation map
    ax0= plt.subplot(gs[2:, 0:2])
    
    
    mat_c = bed_to_mat(bed_track, region2, horizontal=False)
    cmap = cmap_from_bed(bed_track, rgb)
    ax0.matshow(mat_c,cmap=matplotlib.colors.ListedColormap(cmap), aspect='auto', vmin=0, vmax=len(cmap))
    ax0.axis('off')
    ax0.margins(0)
    ax0.set_ylim([(mat_c.shape[0]-1), 0])
    ax0.xaxis.tick_top()
    ax0.yaxis.tick_right()
    plt.yticks(rotation=90)

    # plot horizontal segmentation map
    ax1= plt.subplot(gs[0:2, 2:])
    mat_c = bed_to_mat(bed_track, region1, horizontal=True)
    ax1.matshow(mat_c,cmap=matplotlib.colors.ListedColormap(cmap), aspect='auto', vmin=0, vmax=len(cmap))
    ax1.axis('off')
    ax1.margins(0)
    ax1.set_xlabel('Position along'+ chrm1 +'(50Kb)')
    ax1.set_xlim([0, (mat_c.shape[1] -1)])

    # plot Hi-C map with overlaid lines
    ax2 = plt.subplot(gs[2:, 2:])
    
    mat = hic_cooler.matrix(balance=True).fetch(region2, region1)
    res = hic_cooler.binsize
    im = ax2.matshow(np.log10(mat + 5e-6),cmap = 'YlOrRd', aspect='auto', interpolation ='none')
    
    ymax, xmax = mat.shape
    y_sig1 = bf.select(bed_track, region1).name.values
    y_sig1 = names_to_nums(y_sig1)
    
    y_sig2 = bf.select(bed_track, region2).name.values
    y_sig2 = names_to_nums(y_sig2)
    
    vlines = (np.where(np.abs(np.diff(y_sig1))>0)[0]+0.5)
    hlines = (np.where(np.abs(np.diff(y_sig2))>0)[0]+0.5)
    
    vlines = vlines/ (len(y_sig1) / mat.shape[1])
    hlines = hlines/ (len(y_sig2) / mat.shape[0])
    
    ax2.vlines(vlines, ymin=0, ymax=(ymax),color=[0,0,0,.9995])
    ax2.hlines(hlines, xmin=0, xmax=(xmax), 
               colors=rc('lines', linewidth=0.4, color='black'))
    ax2.set_xlim([0, (xmax-1)])
    ax2.set_ylim([(ymax-1), 0])
        
    ax2.yaxis.set_ticks_position('right')
    chrm2, start2, end2 = bf.parse_region(region2)
    yticklabels = ((ax2.yaxis.get_ticklocs()*res/1e3)+(start2)/1e3).astype(int)
    ax2.yaxis.set_ticklabels(yticklabels)
    
    ax2.xaxis.set_ticks_position('bottom')
    chrm1, start1, end1 = bf.parse_region(region1)
    xticklabels = ((ax2.xaxis.get_ticklocs()*res/1e3)+(start1)/1e3).astype(int)
    ax2.xaxis.set_ticklabels(xticklabels)
    
    if plotColorbar:
        cbar_ax = fig.add_axes([1.05, 0.2, 0.025, 0.3])
        cbar = fig.colorbar(im, cax=cbar_ax, shrink = .7, label='log10 contact freq')
        
    plt.close(fig)
    
    return fig
