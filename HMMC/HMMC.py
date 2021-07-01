import numpy as np
from hmmlearn import hmm
import pandas as pd
import bioframe as bf
# import warnings
# warnings.filterwarnings("error")

"""
    Tools for Hidden Markov Models for segmenting 
    compartment vectors obtained from Hi-C data.
"""


def _make_transmat(n_states, constrained=False):
    transmat = np.ones((n_states, n_states)) / 100#n_states
    np.fill_diagonal(transmat, (1 - 1 / (n_states+1)))

    if n_states is 4 and constrained is True:
        small = 0
        # dis-allow transitions between B and B-in-A
        transmat[0, 2] = small
        # dis-allow transitions between A-in-B and A or B-in-A
        transmat[1, 2:] = small
        # dis-allow transitions between B-in-A and B or A-in-B
        transmat[2, :2] = small
        # dis-allow transitions between A and A-in-B
        transmat[3, 1] = small

    if n_states is 5 and constrained is True:
        small = 0
        transmat[0, 3:] = small
        transmat[1, 2:] = small
        transmat[2, 1] = small
        transmat[2, 3] = small
        transmat[3, :3] = small
        transmat[4, :2] = small

    if n_states is 6 and constrained is True:
        small = 0
        transmat[0, 3:5] = small
        transmat[1, 2:] = small
        transmat[2, 0:2] = small
        transmat[2, 3:5] = small
        transmat[3, 1:3] = small
        transmat[3, 4:] = small
        transmat[4, 0:4] = small
        transmat[5, 1:3] = small

    transmat /= transmat.sum(axis=1, keepdims=True)
    return transmat

def _init_percentiles(n):
    if n==1:
        return np.array([50])
    
    return np.append(
#         np.array([18]),
        np.array([5]),
        np.append(50 if n == 3 else np.linspace(15, 85, n - 2), np.array([95])),
#         np.append(50 if n == 3 else np.linspace(20, 55, n - 2), np.array([82])),
#         np.append(50 if n == 3 else np.array([50]* (n - 2)), np.array([82])),

    )

def _build_model(data, n_components, constrain_transmat=False, n_iter=1000, ):
    transmat = _make_transmat(n_components, constrained=constrain_transmat)
    means = np.percentile(data, _init_percentiles(n_components)).reshape(-1, 1)
    covariance = np.std(data.reshape(-1))
#     print(covariance)
    model = hmm.GaussianHMM(
        n_components,
        algorithm="viterbi",
#         transmat_prior=1 + transmat*(10 ** (n_components - 2)) ,
#         means_weight=10 if n_components == 2 else 10e5,
#         means_weight=1e5,
#         means_prior=means,
        covariance_type="spherical",
        params="smct",
        init_params="s",
#         covars_prior = covariance,
        tol = 1e-10,
        n_iter=n_iter,
    )

    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = np.array([covariance]*n_components)[:, None]
    model.fit(data)
#     print("means", model.means_)
#     print("transmat", model.transmat_)
#     print("covariance", model.covars_)
    return model

def _bic_hmm(model, data):
    ## compute BIC for a trained HMM given a set of observations
    ## based on implementation from https://github.com/fmorenopino/HeterogeneousHMM/blob/master/src/model_order_selection_utils.py
    train_params = model.params
    n_fit_scalars_per_param = model._get_n_fit_scalars_per_param()
    dof = 0
    for par in n_fit_scalars_per_param:
        if par in train_params:
            dof += n_fit_scalars_per_param[par]
    log_likelihood = model.score(data)
    n_samples = len(data.flatten())
    bic = -2 * log_likelihood + dof * np.log(n_samples)
    return bic

def _E1_data(seg_df, region):
    data = bf.select(seg_df, region).E1
    data = data.iloc[~data.isna().values]
    inds = data.index
    data_reshaped = data.values.reshape(-1, 1)
    return inds, data_reshaped


def _run_model(seg_df, n_components, regions, constrain_transmat):
    seg = np.zeros((len(seg_df),) )*np.nan
    bic = 0
    for region in regions:
#         print(region)
        is_chrom = type(region) is str
    
        inds, data_reshaped = _E1_data(seg_df, region)
        
        if len(data_reshaped)==0:
            continue
        
        if len(data_reshaped) < 2*(n_components**2):
            if is_chrom:
                print("skipped chrom", region)
            else:
                print("not enough data in region", region, "fitting entire chromosome instead")
                inds, data_reshaped = _E1_data(seg_df, region[0])
            
        model = _build_model(data_reshaped, 
            n_components,constrain_transmat=constrain_transmat)
        
        predicted = model.predict(data_reshaped)
        bic_region = _bic_hmm(model, data_reshaped)
        
        
        try:
            _check_model(model, region)
            
        except ValueError as inst:
            print(inst)
            print("models means", model.means_.reshape(-1))
            means_order = np.argsort(np.argsort(model.means_.reshape(-1)))
            print("predicted before", predicted)
            predicted = means_order[predicted.reshape(-1)].reshape(predicted.shape)
            print("predicted after", predicted)
            
        
        

        bic += bic_region
        seg[inds] = predicted 
        
    return seg, bic

def _get_state_num(state):
    return int("".join(filter(str.isdigit, state)))

def get_segmentation(eig_df, 
                    state_list= ["binary", "HMM3P"], 
                    regions = None, 
                    constrain_transmat=False,
                    return_BIC_dict = False,
                    verbose = False):
    """Obtain segmentations with HMMs for specified set of states. 

        Parameters
        ----------
        eig_df : pandas.DataFrame 
            Genome-wide binned dataframe with eigenvector calculated at desired resolution.

        state_list : list
            List of HMM states for obtaining segmentations, provided as 
            'HMM2', 'HMM3', etc.
            'binary' calculates the naive binary segmentation.
            'HMM3P' calculates the postprocessed 3-state HMM segmentation.  
        
        regions : pandas.Dataframe
            Genomic intervals stored as a DataFrame, used to limit calculation of the segmentation.
            If not provided, infers regions from unique chromosomes in eig_df.

        constrain_transmat : bool
            If true, initialized HMMs with 3 to 6 states with constrained transition matrices.

        return_BIC_dict : bool
            if True, return dictionary of BICs for HMMs in state_list 

        Returns
        -------
        seg_df : pandas.DataFrame
            Genome-wide binned dataframe with additional segmentations as additional columns
    """
    seg_df = eig_df.copy()
    mask = ~seg_df.E1.isna()
    if return_BIC_dict:
        BIC_dict = {}
    if regions is None:
        regions = seg_df.chrom.unique()

    for state in state_list:
        if verbose is True: 
            print('generating segmentation for '+state)
        try: 
            if state is 'binary':
                seg = (seg_df.E1.copy())
                seg[mask] = np.where(seg[mask] > 0, 1, 0)      
            elif '3P' in state:
                seg = _postprocess_seg(seg_df, state, 
                                      regions, constrain_transmat)
            elif 'HMM' in state:
                seg, bic = _run_model(seg_df, _get_state_num(state),
                                     regions, constrain_transmat)
                if return_BIC_dict:
                    BIC_dict[state] = bic
            seg_df[state] = seg
        except:
            raise ValueError("unknown state", state)

    if return_BIC_dict:
        return seg_df, BIC_dict
    else:
        return seg_df

def _postprocess_seg(seg_df, state, regions, constrain_transmat):
    if '3' not in state:
        raise ValueError("initial segmentation must have 3 states")
    if 'HMM3' in seg_df.keys():
        seg = seg_df['HMM3'].values
    else: 
        seg, bic = _run_model(seg_df, _get_state_num(state),
                              regions, constrain_transmat)    
    mask = np.isnan(seg)
    hmm3p = np.zeros(np.shape(seg)) *np.nan
    sig = seg[~mask] - 1  # sig is -1, 0, 1
    sig_d = np.append(0, np.diff(sig))  # signal of changes
    changes = np.where(sig_d != 0)[0]  # where do changes happen
    changes_d = np.append(np.diff(sig_d[changes]), 0)  # changes of changes,
    interest = np.where((np.abs(changes_d) == 2))
    correction = np.zeros(sig.shape)
    for start, end in zip(changes[interest[0]], changes[interest[0] + 1]):
        if sig[start] == 0:
            correction[start:end] = sig[end]
    hmm3p[~mask] =  sig * 2 + correction + 2
    return hmm3p

def _check_model(model, region):
    l = model.means_
    if not np.array_equal(l, np.sort(l, axis=0)):
        raise ValueError(len(l), '-state model means shifted during model training at region:', region,
                        "means values:", l)
