import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from sklearn import datasets

def generate_GOE(n, d)
"""
generate_GOE returns a sample of 'n' symmetric matrices ~ GOE(d,d)
"""
    matrix_samples = []
    x = y = z = np.array([])
    for i in range(n):
        B = np.random.normal(size=(d, d))
        A = (B+B.T)/2
        matrix_samples.append(A)
    
