import numpy as np
from scipy import linalg


def sass(data,CBAN,CBN,n_nulls)

    """
    Applies the Stimulation Artifact Source Separation to tACS-EEG data

    Parameters
    ----------
    data : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples
        The tACS-EEG data to be processed, bandpass filtered into the frequency band of interest

    CBAN : numpy.ndarray
        m x m ndarray : m: number of channels
        The covariance matrix of EEG calibration data with tACS

    CBN : numpy.ndarray
        m x m ndarray : m: number of channels
        The covariance matrix of EEG calibration data without tACS

    n_nulls : int
        The number of artifact components to reject

    Returns
    -------
    cleaned_data : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples
        The processed tACS-EEG data
    """

    eigen_values, eigen_vectors = linalg.eig(CBAN,CBN)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real
    ix = np.argsort(eigen_values)[::-1]
    D = eigen_vectors[:,ix].T
    M = linalg.pinv2(D)
    DI = np.ones(M.shape[0])
    DI[:n_nulls] = 0
    DI = np.diag(DI)
    P = M.dot(DI).dot(D)
    cleaned_data = P.dot(data)
    return cleaned_data