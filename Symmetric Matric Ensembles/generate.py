import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from sklearn import datasets

def generate_GOE(n=100, d=2)
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
    x = y = z = np.array([])
    for i in range(n):
        B = np.random.normal(size=(d, d))
        A = (B+B.T)/2
        matrix_samples.append(A)
    return matrix_samples

def generate_uni(n, d, a, b)
"""

"""
