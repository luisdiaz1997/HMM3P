import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cooler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def plot_chroms(track, *signal):
    '''
        Used to plot different biological signals in different chromosomes
        track: it takes in a pandas dataframe of Hi-C data
        signal: takes in a tuple or multiple parameters of the columns to be plotted, it will do so per chromosome
        
        
        By default the signals plotted will only be at points with no NaNs in the E1 column.
    '''
    fig = plt.figure(figsize = (30, 20))
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
    plt.close(fig)
    return fig
    
    
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
    
    
def plot_map(cooler_obj, track, chrom, signal = "E1"):
    
    arr = cooler_obj.matrix().fetch(chrom)
    fig= plt.figure(figsize = (10, 10))
    gs = gridspec.GridSpec(10, 10, wspace=0.01, hspace=0.01)
    
    ax = plt.subplot(gs[1:, 1:])
    ax.matshow(np.log(arr + 1e-5), cmap = 'YlOrRd')
#     ax1.axis('off')

#     ax2 = plt.subplot(gs[0:1, 1:])
#     ax2.plot(track.signal
    plt.close(fig)
    return fig


def plot_map_plotly(cooler_obj, track, chrom, signal = "HMM3"):
    loc_eig = track[track.chrom==chrom].copy().reset_index()
    masked_signal = loc_eig[loc_eig['E1'] == loc_eig['E1']]
    indexes_t = np.where(np.abs(np.diff(masked_signal[signal])) > 0)[0] + 1
    line_loc = masked_signal.iloc[indexes_t].index.values #indexes of lines
    
    
    
    arr = cooler_obj.matrix(balance = True).fetch(chrom)
    x = np.linspace(0, len(arr) -1, len(arr), dtype = int)
    y = loc_eig["E1"].values
    y_sig = loc_eig[signal].values
    y_sig = y_sig*2/np.nanmax(y_sig)  -1
    
       
    trace1 = go.Heatmap(z= np.log(arr+1e-5), colorscale = "ylorrd", showscale = False)#, color_continuous_scale='YlOrRd', aspect ='equal')
    
    #trace1.update_xaxes(side="top")
    trace2 = go.Scatter(x = x, y = y, xaxis = "x", yaxis = "y2", mode = 'lines', marker = dict(color = 'orange'), showlegend=False)
    
    trace2_2 = go.Scatter(x = x, y = y_sig, xaxis = "x", yaxis = "y2", mode = 'lines', marker = dict(color = 'blue'), showlegend=False)
    
    trace3 = go.Scatter(x = y, y = x, xaxis = "x2", yaxis = "y", mode = 'lines', marker = dict(color = 'orange'),showlegend=False)
    
    trace3_2 = go.Scatter(x = y_sig, y = x, xaxis = "x2", yaxis = "y", mode = 'lines', marker = dict(color = 'blue'),showlegend=False)

    
    layout = go.Layout(
        xaxis=dict(
            autorange=False
            ,domain=[0.12, 1]
            ,side="top"
#             ,range=[0, len(arr) -1]
            ,range=[80, 200]
            ,tickfont=dict(family='Rockwell', color='black', size=6)
            ,scaleanchor="y"
            ,showgrid=False
#             ,nticks = 6
#             ,tick0 = 0
            
        ),
        yaxis=dict(
             
            domain=[0, 0.88]
            ,autorange= False
#             ,range=[len(arr) -1, 0]
            ,range=[120, 0]
            ,tickangle=-90
            ,tickfont=dict(family='Rockwell', color='black', size=6)
            ,showgrid=False
#             ,nticks = 6
#             ,tick0 = 0
            
            
        ),
        yaxis2=dict(
            domain= [0.9, 1]
            ,tickfont=dict(family='Rockwell', color='black', size=6)
            ,range=[min(-1, np.nanmin(y)), max(np.nanmax(y), 1)]
            ,tick0=-1
            ,dtick=1
            ,showgrid=False
            ,fixedrange=True
            
        ),
        xaxis2=dict(
            domain= [0, 0.1]
            ,autorange= False
            ,range=[max(np.nanmax(y), 1), min(-1, np.nanmin(y))]
            ,side="top"
            ,tickangle=0
            ,tickfont=dict(family='Rockwell', color='black', size=6)
            ,tick0=-1
            , dtick=1
            ,showgrid=False
            ,fixedrange=True
            
        ),
        hovermode="closest",
        
    
    )
    
      
    fig = go.Figure(data=[trace1, trace2,trace2_2, trace3, trace3_2], layout=layout)
    
    '''
    Draw Lines to the map
    '''
    for loc in line_loc:
        fig.add_shape(
            # Line reference to the axes
                type="line",
                xref="x",
                yref="y",
                x0=0,
                y0=loc,
                x1=len(y)-1,
                y1=loc,
                line=dict(
                    color="black",
                    width=1,
                ),
            )
        fig.add_shape(
            # Line reference to the axes
                type="line",
                xref="x",
                yref="y",
                x0=loc,
                y0=0,
                x1=loc,
                y1=len(y)-1,
                line=dict(
                    color="black",
                    width=1,
                ),
            )
    
    fig.update_layout(title=dict(text=chrom), width = 650, height = 650, autosize=False, margin=dict(t=30, b=30, l=30, r=30))
    
    
    return fig
    
    
