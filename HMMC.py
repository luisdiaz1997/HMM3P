import numpy as np
from hmmlearn import hmm
import pandas as pd

'''
    Tools for Hidden Markov Models with Hi-C data
'''


def transmat_2():
    transmat = np.array([[0.5, 0.5], [0.5, 0.5]])
    transmat /= transmat.sum(axis = 1, keepdims= True)
    return transmat

def transmat_3():
    transmat = np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]])
    transmat /= transmat.sum(axis = 1, keepdims= True)
    return transmat

def transmat_4():
    n_states = 4
    transmat = np.zeros((n_states, n_states))
    transmat[:] = .3
    for i in range(n_states):
        for j in range(n_states):
            if i==j:
                transmat[i, j] = 0.7
    small = 1e-4
    transmat[0,2] = small
    transmat[1,2:] = small
    transmat[2,0:2] = small
    transmat[3,1] = small
    transmat /=  np.sum(transmat,axis=1)[:,None]
    return transmat

def transmat_5():
    n_states = 5
    transmat = np.zeros((n_states, n_states))
    transmat[:] = .2
    for i in range(n_states):
        for j in range(n_states):
            if i==j:
                transmat[i, j] = 0.8

    small = 1e-4
    transmat[0,3] = small
    transmat[1,2:] = small
    transmat[2,1] = small
    transmat[2,3] = small
    transmat[3,0:3] = small
    transmat[4,1] = small
    transmat /=  np.sum(transmat,axis=1)[:,None]
    return transmat

def transmat_6():
    n_states = 6
    transmat = np.zeros((n_states, n_states))
    transmat[:] = .2
    for i in range(n_states):
        for j in range(n_states):
            if i==j:
                transmat[i, j] = 0.6
    small = 1e-4
    transmat[0,3:5] = small
    transmat[1,2:] = small
    transmat[2,0:2] = small
    transmat[2,3:5] = small
    transmat[3,1:3] = small
    transmat[3,4:] = small
    transmat[4,0:4] = small
    transmat[5,1:3] = small
    transmat /=  np.sum(transmat,axis=1)[:,None]
    return transmat

transmat_dict = {2:transmat_2(), 
                 3:transmat_3(),
                 4:transmat_4(),
                 5:transmat_5(),
                 6:transmat_6(),}

def percentiles_val(n):
    return np.append([20], np.append(50 if n== 3 else np.linspace(45, 55, n-2), np.array([80])))  

def build_model(data ,n_components,  n_iter = 100):
    transmat = transmat_dict[n_components]
    means = np.percentile(data, percentiles_val(n_components)).reshape(-1,1)
    model = hmm.GaussianHMM(n_components,
                        algorithm = 'viterbi',
                        transmat_prior= 1+ (10**(n_components-2))*transmat,
                        means_weight = 10 if n_components==2 else 10e4,
                        means_prior = means,
                        covariance_type='diag', 
                        params='smct',
                        init_params='sc', 
                        n_iter= n_iter)

    if not (type(transmat) == type(1.0)):
        model.transmat_ = transmat

    startprob = np.zeros(n_components)
    startprob[:] = 0.1
    startprob[0]= 0.4
    startprob[-1] = 0.4
    startprob /=  np.sum(startprob)
    model.startprob_ = startprob

    model.means_ = means
    model.fit(data)
    return model


def train_models(data, transmat_type='diag'):
    if transmat_type is 'diag':
        transmat_dict = transmat_dict_diag
    elif transmat_type is 'constrained':
        transmat_dict = transmat_dict_constrained
    models = { n:build_model(data, n, transmat = transmat_dict[n] for n in range(2,7) }
    predicted = { n:models[n].predict(data) for n in range(2,7)}
    return models, predicted



def postprocess(signal):
    sig = signal-1    #signal is -1, 0, 1
    sig_d = np.append(0, np.diff(sig)) #signal of changes
    changes = np.where(sig_d != 0)[0] #where do changes happen
    changes_d = np.append(np.diff(sig_d[changes]), 0) #changes of changes, 
    print(changes_d)
    interest= np.where((np.abs(changes_d) == 2))
    
    correction = np.zeros(sig.shape)
    
    for start, end in zip(changes[interest[0]], changes[interest[0] +1]):
        if sig[start] == 0:
            correction[start:end] = sig[end]
    return sig*2 + correction + 2


class HMMC:
    def __init__(self, df):
        self.loc_eig = df.copy()
        self.mask = ~self.loc_eig.E1.isna()
        
        
        self.data = [self.loc_eig.E1[(self.mask) & (self.loc_eig.chrom == ch)].values  for ch in self.loc_eig.chrom.unique()] 
        
        
    
    def analyze(self):
        indexes = self.mask[self.mask].index
            
        #Create binary model
        conditions, choices = [(self.loc_eig.E1 >= 0), (self.loc_eig.E1 <  0)], [1,0]
        self.loc_eig['binary']  = np.select(conditions, choices, default= np.nan)
        
        
        i = 1
        
        self.predicted = dict()
        for dat in self.data:
            if len(dat) == 0:
                continue
            model, pred = train_models(dat.reshape(-1,1))
            
            #Check if model is shifted
            if self.check_model(model):
                break
            
            for key, value in pred.items():
                
                self.predicted[key] = np.concatenate( (self.predicted.get(key, np.array([])), value))
            
        
            
        for n in range(2,7):
            self.loc_eig['HMM' + str(n)] = pd.Series(data = self.predicted[n], index = indexes)
            
        
            
            
        signal = self.loc_eig.HMM3.values[indexes]
        
        #Create Post Processed 3 to 5 state model
        postsignal = postprocess(signal)
        self.loc_eig['HMM5_2'] = pd.Series(data = postsignal, index = indexes)
        
            
    def check_model(self, model):
        shifted = False
        for n in range(2,7):
            l = model[n].means_
            flag = 0
            if(not np.array_equal(l, np.sort(l, axis = 0))):
                print('Model ' + str(n) + ' Means shifted')
                print('------------------------------------')
                shifted = True
                
        return shifted

def auto_analyze(df):
    a = HMMC(df)
    a.analyze()
    return a.loc_eig

