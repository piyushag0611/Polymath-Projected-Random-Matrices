import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from sklearn import datasets

def main():
    n = 100
    d = 2
    matrix_samples = []
    x = y = z = np.array([])
    phi = datasets.make_spd_matrix(d**2)
    for i in range(n):
        X = np.random.multivariate_normal(mean=np.zeros(d**2),cov=phi)
        B = X.reshape((d,d), order='F')
        A = (B+B.T)/2
        matrix_samples.append(A)
    
    matrix_mean = np.sum(matrix_samples, axis=0)/n

    if(d==2):
        for matrix in matrix_samples:
            x = np.append(x, matrix[0][0])
            y = np.append(y, matrix[0][1])
            z = np.append(z, matrix[1][1])
        mlab.points3d(x, y, z)
        mlab.view()
        mlab.savefig('D:\Summer 2022\Polymath Jr\codes\points_cgse.png')
    
if __name__ == '__main__':
    main()