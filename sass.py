import numpy as np
from scipy import linalg

def find_n_nulls(A,B,D,M):
    mses = []
    for n_nulls in range(A.shape[0]):
        DI = np.ones(M.shape[0])
        DI[:n_nulls] = 0
        DI = np.diag(DI)
        P = M.dot(DI).dot(D)
        mses.append(np.mean((np.diag(B)-np.diag(P.dot(A).dot(P.T)))**2))
    return np.argmin(mses)

def sass(data,A,B):

    """
    Applies the Stimulation Artifact Source Separation to tACS-EEG data

    Parameters
    ----------
    data : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples
        The tACS-EEG data to be processed, bandpass filtered into the frequency band of interest

    A : numpy.ndarray
        m x m ndarray : m: number of channels
        The covariance matrix of EEG calibration data with tACS

    B : numpy.ndarray
        m x m ndarray : m: number of channels
        The covariance matrix of EEG calibration data without tACS

    Returns
    -------
    cleaned_data : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples
        The processed tACS-EEG data
    """

    eigen_values, eigen_vectors = linalg.eig(A,B)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real
    ix = np.argsort(eigen_values)[::-1]
    D = eigen_vectors[:,ix].T
    M = linalg.pinv2(D)
    n_nulls = find_n_nulls(A,B,D,M)
    DI = np.ones(M.shape[0])
    DI[:n_nulls] = 0
    DI = np.diag(DI)
    P = M.dot(DI).dot(D)
    cleaned_data = P.dot(data)
    return cleaned_data
