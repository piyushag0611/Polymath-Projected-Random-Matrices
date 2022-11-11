import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from sklearn import datasets

def generate_GOE(n=100, d=2):
    """
    Returns a sample of 'n' symmetric matrices ~ GOE(d,d)

    Parameters
    ----------
    n : int
      Size of the sample output. The default is 100.
    d : int
      Size of the Symmetric Matrix in the sample. The default is 2.
      
    Returns
    -------
    matrix_samples : array-like
                   A list of the d*d symmetric matrices generated
                   
    """
    matrix_samples = []
    for i in range(n):
        B = np.random.normal(size=(d, d))
        A = (B+B.T)/2
        matrix_samples.append(A)
    return matrix_samples

def generate_uni(n, d, a, b):
    """

    """
    matrix_samples = []
    for i in range(n):
        B = np.random.uniform(low=a, high=b, size=(d, d))
        A = (B+B.T)/2
        matrix_samples.append(A)
    return matrix_samples   

def generate_cgse(n, d, Phi):
    """

    """
    matrix_samples = []
    for i in range(n):
        X = np.random.multivariate_normal(mean=np.zeros(d**2), cov=Phi)
        B = X.reshape((d,d), order='F')
        A = (B+B.T)/2
        matrix_samples.append(A)
    return matrix_samples

def matrix_arith_mean(matrix_samples):
    """
    
    """
    n = len(matrix_samples)
    matrix_mean = np.sum(matrix_samples, axis=0)/n
    return matrix_mean

def eval_eig_values(matrix_samples):
    """
    
    """
    eig_values = []
    for matrix in matrix_samples:
        eig_val = np.linalg.eigvalsh(matrix)
        eig_values.append(eig_val)
    
    return eig_values

def matrix_viz(matrix_samples, fname='points.png'):
    """
    
    """
    d = int(np.sqrt(matrix_samples[0].size))
    x = y = z = np.array([])
    if(d==2):
        for matrix in matrix_samples:
            x = np.append(x, matrix[0][0])
            y = np.append(y, matrix[0][1])
            z = np.append(z, matrix[1][1])
        mlab.points3d(x, y, z)
        mlab.view()
        mlab.savefig(fname) 
    else:
        # raise a system error that matrix visualization is not possible
        pass  
    
    return None


def eig_value_viz(eig_samples, fname):
    """
    
    """
    d = int(eig_samples[0].size)
    x = y = z = np.array([])
    if(d==3):
        for eig_values in eig_samples:
            x = np.append(x, eig_values[0])
            y = np.append(y, eig_values[1])
            z = np.append(z, eig_values[2])
        mlab.points3d(x, y, z)
        mlab.view()
        mlab.savefig(fname)
    else:
        # raise a system error that matrix visualization is not possible
        pass    
    
    return None







