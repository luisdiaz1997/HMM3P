# from bioframe.util import tsv
# from bioframe.util import bedtools

from bioframe.tools import bedtools
from bioframe.io.process import tsv

import numpy as np

'''
    All the tools in this package that deal with bedtools and bioframe
'''

def get_chips(beds_df, cell_line, assembly, output_type=None, replicate = None):
    '''
        Filter chipseq dataset// in progress
    '''
    loc = beds_df[( beds_df.cell_line == cell_line)& 
                  (beds_df.assembly == assembly)                 ]

    loc = loc.sort_values(by='target')
    loc = loc.reset_index(drop = True)
    return loc


def bedtools_intersect(left, right, chromhmm=False, **kwargs):
    '''
        left: takes dataframe
        right: file location of bed file

    '''

    with tsv(left) as a:
        out = bedtools.intersect(a=a.name, b=right, wa=False, wb=False, loj=True, **kwargs)
        
        if chromhmm:
            columns = left.columns.tolist() + ['chrom_p', 'start_p', 'end_p', 'name_p']
            out = out.iloc[:, :len(columns)]
            out.columns = columns
            out['name_p'] = out['name_p'].astype('str')
        else:
            columns = left.columns.tolist() + ['chrom_p', 'start_p', 'end_p', 'name_p', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak']
            out.columns = columns
            out['signalValue'] = out['signalValue'].apply(lambda a: a.replace('.', '0'))
            out['signalValue'] = out['signalValue'].astype(float)

    return out

def bedtools_intersect_basePairs(a, b, hmm_state, rsuffix='_', **kwargs):
    """
    >>> ixn = bedtools_intersect(bed1, bed2, wao=True), needed to delete other keys b/c they clashed
   """
    
    out = bedtools.intersect(a=a,b=b,wao=True)
    out = out.drop(list(np.arange(4,10)) + list(np.arange(13, 18)), axis =1)
    out.columns = ['chrom', 'start', 'end', hmm_state, 'start_p', 'end_p', 'name_p', 'bp']
    #out.columns = list(left.columns) + [c+rsuffix if c in left.columns else c for c in right.columns] + ['bpOverlap']
    return out


def bp_over_total(inter_df, hmm_state='HMM3'):
    
    n_states = inter_df[hmm_state].unique()
    n_states = n_states[n_states == n_states]
    n_states.sort()
    
    mask = (inter_df['start_p']!=-1)
    
    total_nucleotides = (inter_df[mask]['end_p'] - inter_df[mask]['start_p'])
    
    return np.array([total_nucleotides[(inter_df[hmm_state]==state) & mask].sum() for state in n_states])/total_nucleotides.sum()

    
    
def bp_over_state(inter_df, hmm_track, hmm_state='HMM3', chrom_state = False, normalize = True):
    
    n_states = inter_df[hmm_state].unique()
    n_states = n_states[n_states == n_states]
    n_states.sort()
    
    mask = (inter_df['start_p']!=-1)
    if chrom_state:        
        mask = mask & (inter_df['name_p']==chrom_state)
  
    total_nucleotides = (inter_df['end_p'] - inter_df['start_p'])
    
    if normalize:
        return np.array([total_nucleotides[(inter_df[hmm_state]==state) & mask].sum()/(len(hmm_track[hmm_track[hmm_state]==state])) for state in n_states])/ (total_nucleotides[mask].sum()/len(hmm_track))
    
    else:
        return np.array([total_nucleotides[(inter_df[hmm_state]==state) & mask].sum()/(100000*len(hmm_track[hmm_track[hmm_state]==state])) for state in n_states])


def drop_between(df, identifier, start, end, signal):
    '''
        Drops peaks in between two bins to, the bin that contains most basepairs takes all the peak value

        df: intersected dataframe
        indentifier: the column name of the unique identifier of each peak
        start: column name of the peaks basepair start
        end: column name of the peaks basepair end
        signal: the signal to be analyzed
    '''
    same_id = df[ (df[identifier] != '.') & df.duplicated([identifier], keep=False)]
    if len(same_id) == 0:
        return df

    l1 = same_id[(same_id[start] > same_id['start']) & ((same_id['end'] - same_id[start]) <=  (same_id[end] - same_id['end']))]
    l2 = same_id[(same_id[start] < same_id['start']) & ((same_id['start'] - same_id[start]) >  (same_id[end] - same_id['start']))]
    test = df.copy()
    test.loc[l1.index, signal] = 0
    test.loc[l2.index, signal] = 0
    return test


def add_peaks(df, signal):
    '''
        This adds all signals in the same Hi-C bin, assuming
        df: intersected hi-c and bed dataframe
        signal: column name of the signal to be added
    '''
    same_bin = df[ df.duplicated(['start', 'chrom'] , keep=False)] #get peaks in the same bin
    sum_signals = same_bin.groupby(['chrom', 'start']).agg('sum')[signal].values #sum all the signals in same bin
    uniq_bins = same_bin[same_bin.duplicated([ 'chrom', 'start'] , keep='first') == False].copy() #per bin, get the first peak
    test = df[df.duplicated([ 'chrom', 'start'] , keep='first') == False].copy()
    test.loc[uniq_bins.index, signal] = sum_signals
    return test



def chip_intersect(track, bed_dir):
    '''
        Does interesections and applies the drop_between and add_peaks
        track: hi-c dataframe
        bed_dir: directory of bed file
    '''

    inter = bedtools_intersect(track, bed_dir)
    inter = drop_between(inter, identifier = 'name_p', start = 'start_p', end = 'end_p', signal = 'signalValue')
    inter = add_peaks(inter, signal = 'signalValue')
    return inter


def fold_score(inter_track, model):
    '''
        This gets the fold score of the chipseq with respect to the model
        inter_track: intersected dataframe
        model: name of the column containing the model to be analyzed
    '''
    hmm_sig = inter_track[(inter_track.E1 == inter_track.E1)][model].values.astype(int)
    chip_sig = inter_track[(inter_track.E1 == inter_track.E1)].signalValue.values
    m = np.median(chip_sig[chip_sig != 0])
    d = np.array([np.median(chip_sig[(hmm_sig == i)& (chip_sig !=0)]) for i in range(np.max(hmm_sig) +1)])
    return d/m


def multi_fold(hic_info, result_track, beds_df, model, *targets):

    vals = list()
    beds = beds_df[( beds_df.cell_line == hic_info.cell_line)& (beds_df.target == target) & (beds_df.assembly == hic_info.assembly)]

    for target in targets:

        val = fold_score(result_track, model)
        vals.append(val)
    return np.array(vals)




def get_window(df, chrom, signal, model, coverage):
    curr = df[df['chrom'] == chrom]
    z= curr[curr[model]==curr[model]][model].values
    sig = curr[curr[model]==curr[model]][signal].values
    z_t = transition_indices(z)

    bound = np.where((z_t == True))[0] ##double check boundaries
    #indexes = bound[np.where(np.diff(bound)>coverage)[0]] ##improve this later
    #print('Bound shape:', bound.shape)
    indexes1 = bound[0::2] #get half of the signals
    indexes2 = bound[1::2] #get the other half
    if z[indexes2[0]] > z[indexes1[0]]:  #Make sure is high to low signal
        test = indexes1
        indexes1 = indexes2
        indexes2 = test
    #print('I1 shape', indexes1.shape)
    #print('I2 shape', indexes2.shape)
    scores1 = np.array([sig[index- coverage: index+coverage+1]  for index in indexes1 if len(sig[index- coverage: index+coverage+1]) > coverage*2])
    scores2 = np.array([np.flip(sig[index- coverage+1: index+coverage+2])  for index in indexes2 if len(sig[index- coverage+1: index+coverage+2]) > coverage*2])

    #print(scores1.shape)
    #print(scores2.shape)
    scores = np.concatenate((scores1, scores2), axis = 0)
    return scores
