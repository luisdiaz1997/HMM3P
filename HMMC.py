import numpy as np
from hmmlearn import hmm
import pandas as pd

"""
    Tools for Hidden Markov Models with Hi-C data
"""


def make_transmat(n_states, constrained=False):
    transmat = np.ones((n_states, n_states)) / n_states
    np.fill_diagonal(transmat, (1 - 1 / n_states))

    if n_states is 4 and constrained is True:
        small = 1e-4
        # dis-allow transitions between B and B-in-A
        transmat[0, 2] = small
        # dis-allow transitions between A-in-B and A or B-in-A
        transmat[1, 2:] = small
        # dis-allow transitions between B-in-A and B or A-in-B
        transmat[2, :2] = small
        # dis-allow transitions between A and A-in-B
        transmat[3, 1] = small

    if n_states is 5 and constrained is True:
        small = 1e-4
        transmat[0, 3:] = small
        transmat[1, 2:] = small
        transmat[2, 1] = small
        transmat[2, 3] = small
        transmat[3, :3] = small
        transmat[4, :2] = small

    if n_states is 6 and constrained is True:
        small = 1e-4
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

def init_percentiles(n):
    return np.append(
        np.array([20]),
        np.append(50 if n == 3 else np.linspace(45, 55, n - 2), np.array([80])),
    )

def build_model(data, n_components, n_iter=100, constrain_transmat=False):
    transmat = make_transmat(n_components, constrained=constrain_transmat)
    means = np.percentile(data, init_percentiles(n_components)).reshape(-1, 1)
    model = hmm.GaussianHMM(
        n_components,
        algorithm="viterbi",
        transmat_prior=1 + (10 ** (n_components - 2)) * transmat,
        means_weight=10 if n_components == 2 else 10e4,
        means_prior=means,
        covariance_type="diag",
        params="smct",
        init_params="sc",
        n_iter=n_iter,
    )

    model.transmat_ = transmat

    startprob = np.zeros(n_components)
    startprob[:] = 0.1
    startprob[0] = 0.4
    startprob[-1] = 0.4
    startprob /= np.sum(startprob)
    model.startprob_ = startprob

    model.means_ = means
    model.fit(data)
    return model

def train_models(data, constrain_transmat=False):
    if constrain_transmat is True: print('training constrained models')
    models = {n: build_model(data, n, constrain_transmat=constrain_transmat) for n in range(2, 7)}
    predicted = {n: models[n].predict(data) for n in range(2, 7)}
    return models, predicted

def postprocess_3_to_5_states(signal):
    if len(np.unique(signal)) is not 3:
        raise ValueError("signal must have 3 states")
    sig = signal - 1  # signal is -1, 0, 1
    sig_d = np.append(0, np.diff(sig))  # signal of changes
    changes = np.where(sig_d != 0)[0]  # where do changes happen
    changes_d = np.append(np.diff(sig_d[changes]), 0)  # changes of changes,
    print(changes_d)
    interest = np.where((np.abs(changes_d) == 2))

    correction = np.zeros(sig.shape)

    for start, end in zip(changes[interest[0]], changes[interest[0] + 1]):
        if sig[start] == 0:
            correction[start:end] = sig[end]
    return sig * 2 + correction + 2


class HMMC:
    def __init__(self, df, constrain_transmat=False):
        self.loc_eig = df.copy()
        self.mask = ~self.loc_eig.E1.isna()

        self.data = [
            self.loc_eig.E1[(self.mask) & (self.loc_eig.chrom == ch)].values
            for ch in self.loc_eig.chrom.unique()
        ]

        self.constrain_transmat = constrain_transmat

    def analyze(self):
        indexes = self.mask[self.mask].index
        # Create binary model
        conditions, choices = [(self.loc_eig.E1 >= 0), (self.loc_eig.E1 < 0)], [1, 0]
        self.loc_eig["binary"] = np.select(conditions, choices, default=np.nan)
        i = 1
        self.predicted = dict()
        for dat in self.data:
            if len(dat) == 0:
                continue
            model, pred = train_models(dat.reshape(-1, 1), constrain_transmat=self.constrain_transmat)
            # Check if model is shifted
            if self.check_model(model):
                break
            for key, value in pred.items():
                self.predicted[key] = np.concatenate(
                    (self.predicted.get(key, np.array([])), value)
                )
        for n in range(2, 7):
            self.loc_eig["HMM" + str(n)] = pd.Series(
                data=self.predicted[n], index=indexes
            )
        signal = self.loc_eig.HMM3.values[indexes]

        # Create Post Processed 3 to 5 state model
        postsignal = postprocess_3_to_5_states(signal)
        self.loc_eig["HMM5_2"] = pd.Series(data=postsignal, index=indexes)

    def check_model(self, model):
        shifted = False
        for n in range(2, 7):
            l = model[n].means_
            flag = 0
            if not np.array_equal(l, np.sort(l, axis=0)):
                print("Model " + str(n) + " Means shifted")
                print("------------------------------------")
                shifted = True

        return shifted

def auto_analyze(df, constrain_transmat=False):
    a = HMMC(df, constrain_transmat=constrain_transmat)
    a.analyze()
    return a.loc_eig
