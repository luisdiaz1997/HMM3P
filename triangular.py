import sklearn.preprocessing as preprocessing
from scipy.signal import correlate
import numpy as np
import matplotlib.pyplot as plt




def find_pattern(signal, pattern, depth):
    '''
        Finds the index of a pattern in a given discrete signal
        middle state is defined as elements in pattern[1:-1]
        
        Parameters:
            pattern (np.array):  pattern to be found
            signal (np.array): signal discrete signal
            depth (int): number containing the maximum repetition of the middle state on the pattern
        
        Returns:
            index_dict (dict): each element contains a list of np.arrays of indexes.
                                each list is on the following format
                                [indexes(pattern), indexes(reversed(pattern))]
        
        
    '''
    encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
    encoder.fit(signal[:, None])
    index_dict = dict()
    for current_depth in range(1, depth + 1):
        
        h = np.append( pattern[0], np.append(np.repeat(pattern[1:-1], current_depth), pattern[-1]))
        #middle state is expanded every iteration
        
        
        x_ = encoder.transform(signal[:, None]).toarray()
        h_ = encoder.transform(h[:, None]).toarray()
        
        
                
        y_ = correlate(x_.T, h_.T, mode = 'valid')[0]
        
        y_r = correlate(x_.T, h_[::-1].T, mode = 'valid')[0]
        
        index_dict[current_depth] = [np.where(y_ == len(h_))[0], np.where(y_r == len(h_))[0]]
    
    return index_dict


def get_window(indexes, signal, win_size = 5):
    '''
    Will take a signal, from a given index will take the [index-win_size:index+win_size+1] elements
    
        Parameters:
            indexes (np.array- 1D or list) : list of indexes
            signal (np.array):  signal to be windowed
            win_size (int): size of window
        Returns:
            window (2D-array)
        
    '''
    indexes = indexes[np.where( (indexes>win_size) & (indexes < (len(signal) -win_size) ) )[0]]
    if len(indexes) == 0:
        return np.array([]).reshape(0, 2*win_size + 1)    
    
    window = np.array([signal[index-win_size:index+win_size+1] for index in indexes])
    return window




    

def stack_windows(index_dict, target_signal, win_size = 5):
    '''
        Gets all the windows of size 2*win_size +1
        For a given index
        Parameters:
            index_dict (dict): dictionary that has the depth as an index.
            target_signal (np.array): array containing signal to be processed
            win_size (int): size of windows
        Returns:
            windows_dict (dict): each value contains a list
                                 each list is of the format
                                 [windows(indexes(patterns)),  windows(indexes(reversed(pattern)))]
    '''
    windows_dict = dict()
    
    for key, val in index_dict.items():
                    
       
        windows_dict[key] = [get_window(val[0] + (key + 1)//2, target_signal, win_size = win_size), 
                get_window(val[1] + int(np.ceil((key + 1)/2)), target_signal, win_size = win_size)]
        
        
    return windows_dict


def create_triangle(windows_dict, average = False):
    
    triangle = list()
    for i in range(1, len(windows_dict) + 1):
        
        combined = np.concatenate( ( windows_dict[i][0], windows_dict[i][1][:, ::-1]))
        if average and combined.size:
            combined = combined.mean(axis = 0, keepdims = True)
        
        triangle.append(combined)
        
    return np.vstack(triangle)
        
    
def plot_triangle(index_dict, signal, figsize = (10, 8), win_size = 10, average = True, fold = False, title = None):

    windows_dict = stack_windows(index_dict,signal, win_size)
    triangle = create_triangle(windows_dict, average)
    fig = plt.figure( figsize=figsize)
    if fold:
        
        mask = triangle > np.percentile(triangle, 25)
        median = np.median(triangle[mask])
        enrich = np.clip(triangle/median, 0, 2)
        plt.imshow(enrich, cmap = 'bwr', vmin = 0, vmax = 2, extent  = [ -(triangle.shape[1]-1)*100/2, (triangle.shape[1]-1)*100/2, triangle.shape[0], 0 ], aspect = 'auto' )
    else:
        
        plt.imshow(triangle, cmap = 'bwr', vmin = -1, vmax = 1, extent  = [ -(triangle.shape[1]-1)*100/2, (triangle.shape[1]-1)*100/2, triangle.shape[0], 0 ], aspect = 'auto' )
    
    if title:
        plt.title(title)
    plt.colorbar()
    plt.xlabel("Kb");
    plt.close(fig)
    return fig
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        