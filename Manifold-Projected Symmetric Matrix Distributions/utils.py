import numpy as np
from sklearn import datasets

def gen_GOE(d):
    B = np.random.randn(d,d)
    return (B + B.T) / 2

def gen_CGSE(d, Phi):
    if Phi is None:
        Phi = datasets.make_spd_matrix(d**2)
    X = np.random.multivariate_normal(mean=np.zeros(d**2), cov=Phi)
    B = X.reshape((d, d), order='F')
    return (B + B.T) / 2

def gen_uniform(d):
    B = np.random.uniform(low=-1, high=1, size=(d, d))
    return (B + B.T) / 2

def exp_sigma(E, Sigma=None):
    if Sigma is None:
        Sigma = np.identity(np.shape(E)[0])
    X = np.linalg.solve(Sigma, E) # for general square matrices
    D, V = np.linalg.eig(X)
    L = np.diag(np.exp(D))
    return Sigma @ V @ L @ np.linalg.inv(V)