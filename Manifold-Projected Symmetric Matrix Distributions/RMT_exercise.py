# Author : Yi Luo
# Time : 2022/7/3 9:49
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt

# semicircular law
n = 1000
t = 1
v = []
dx = 0.2
for i in range(t):
    A = np.random.randn(n, n)
    S = (A + A.T) / 2
    eigvals = np.linalg.eig(S)[0]

eigvals = [x / sqrt(n / 2) for x in eigvals]

sns.histplot(eigvals, bins=50, stat='probability')
plt.show()

# level density of GUE with beta=2


# Marcenko-Pastur law
