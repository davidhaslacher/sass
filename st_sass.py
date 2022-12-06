import numpy as np
from statsmodels.stats.moment_helpers import cov2corr
from tqdm import tqdm
from scipy import linalg

def euclidean_distance(A,B):
    return np.linalg.norm(A - B, ord='fro')

def find_n_nulls(A,B,D,M):
    mses = []
    for n_nulls in range(A.shape[0]):
        DI = np.ones(M.shape[0])
        DI[:n_nulls] = 0
        DI = np.diag(DI)
        P = M.dot(DI).dot(D)
        mses.append(euclidean_distance(cov2corr(B),cov2corr(P.dot(A).dot(P.conj().T))))
    return np.argmin(mses)

def make_P(X_no_stim,X_stim):
    A = np.cov(X_stim)
    B = np.cov(X_no_stim)
    eigen_values, eigen_vectors = linalg.eig(A,B)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real
    ix = np.argsort(eigen_values)[::-1]
    D = eigen_vectors[:,ix].T
    M = linalg.pinv(D)
    n_nulls = find_n_nulls(A,B,D,M)
    print("n_nulls = {:d}".format(n_nulls))
    DI = np.ones(M.shape[0])
    DI[:n_nulls] = 0
    DI = np.diag(DI)
    P = M.dot(DI).dot(D)
    return P

def make_kmers(sequence,half_k_size):
    kmers = []
    for i in range(half_k_size,len(sequence)-(half_k_size+1)):
        kmer = sequence[i-half_k_size:i+half_k_size+1]
        kmers.append(kmer)
    return np.array(kmers).T

def st_sass(X_no_stim,X_stim,sfreq=100):
    """
    Clean EEG data of electric stimulation artifacts using spatiotemporal stimulation artifact source separation (ST-SASS).

    Parameters
    ----------
    X_no_stim : ndarray, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
        EEG data in absence of electric stimulation. Should be filtered approximately from 1 - 40 Hz.
    X_stim : ndarray, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
        EEG data in the presence of electric stimulation. Should be filtered approximately from 1 - 40 Hz.
    sfreq : float
        Sampling rate of the EEG data. The data should be resampled to about 100 Hz for best performance of ST-SASS.

    Returns
    -------
    X_stim_new : ndarray, shape (n_channels, n_times) or or (n_epochs, n_channels, n_times)
        Clean EEG data in the presence of electric stimulation.
    """
    if X_no_stim.ndim == 2:
        # Apply spatial SASS
        P = make_P(X_no_stim,X_stim)
        X_stim = P@X_stim
        # Apply temporal SASS
        X_stim_new = []
        for ch_ix in tqdm(range(X_no_stim.shape[0])):
            ch_data_no_stim = X_no_stim[ch_ix]
            ch_data_stim = X_stim[ch_ix]
            half_k_size = int(sfreq*2)
            kmers_stim = make_kmers(ch_data_stim,half_k_size)
            kmers_no_stim = make_kmers(ch_data_no_stim,half_k_size)
            P = make_P(kmers_no_stim,kmers_stim)
            kmers_stim = P@kmers_stim
            X_stim_new.append(np.concatenate([kmers_stim[:half_k_size,0],kmers_stim[half_k_size],kmers_stim[-half_k_size:,-1]]))
        return np.array(X_stim_new)
    elif X_no_stim.ndim == 3:
        # Apply spatial SASS
        P = make_P(np.hstack(X_no_stim),np.hstack(X_stim))
        X_stim = np.array([P@X for X in X_stim])
        # Apply temporal SASS
        X_stim_new = []
        for ch_ix in tqdm(range(X_no_stim.shape[1])):
            ch_data_no_stim = X_no_stim[:,ch_ix]
            ch_data_stim = X_stim[:,ch_ix]
            P = make_P(ch_data_no_stim.T,ch_data_stim.T)
            X_stim_new.append(P@ch_data_stim.T)
        return np.array(X_stim_new).transpose((2,0,1))
    else:
        Exception('EEG data should be of shape (n_channels, n_times) or or (n_epochs, n_channels, n_times)')