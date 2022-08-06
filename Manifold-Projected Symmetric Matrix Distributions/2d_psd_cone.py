# %%
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets
from utils import gen_GOE, exp_sigma

# %%
def f(x, y):
    return np.sqrt(x * y), -np.sqrt(x * y)

# %%


# %%
x1 = np.linspace(0, 30, 300)
y1 = np.linspace(0, 30, 300)
X1, Y1 = np.meshgrid(x1, y1)
Z1 = f(X1, Y1)[0]
Z2 = f(X1, Y1)[1]


# %%
n = 100000

Sigma = np.array([[1, -1], [-1, 2]])
a_goe = []
b_goe = []
d_goe = []
lambda_goe = []
a_goe_exp1 = []
b_goe_exp1 = []
d_goe_exp1 = []
lambda_goe_exp1 = []
a_goe_exp2 = []
b_goe_exp2 = []
d_goe_exp2 = []
lambda_goe_exp2 = []

for i in range(n):
    X_goe = gen_GOE(2)
    X_goe_exp1 = exp_sigma(X_goe)
    X_goe_exp2 = exp_sigma(X_goe, Sigma)
    lambda_goe.append(list(np.sort(np.linalg.eigvals(X_goe))))
    lambda_goe_exp1.append(list(np.sort(np.linalg.eigvals(X_goe_exp1))))
    lambda_goe_exp2.append(list(np.sort(np.linalg.eigvals(X_goe_exp2))))
    a_goe.append(X_goe[0, 0])
    b_goe.append(X_goe[0, 1])
    d_goe.append(X_goe[1, 1])
    a_goe_exp1.append(X_goe_exp1[0, 0])
    b_goe_exp1.append(X_goe_exp1[0, 1])
    d_goe_exp1.append(X_goe_exp1[1, 1])
    a_goe_exp2.append(X_goe_exp2[0, 0])
    b_goe_exp2.append(X_goe_exp2[0, 1])
    d_goe_exp2.append(X_goe_exp2[1, 1])


# %%
import matplotlib.pyplot as plt
fig = plt.figure()

ax = Axes3D(fig)
# ax = plt.axes(projection="3d")
plt.xlim(-10,30)
plt.ylim(-10,30)
ax.set_zlim(-15,15)

ax.contour3D(X1, Y1, Z1 ,50, cmap='binary')
ax.contour3D(X1, Y1, Z2 ,50, cmap='binary')


ax.scatter3D(a_goe, d_goe, b_goe, s=1, c="#F92301")
ax.scatter3D(a_goe_exp1, d_goe_exp1, b_goe_exp1, s=1, c="#FDB638")
ax.scatter3D(a_goe_exp2, d_goe_exp2, b_goe_exp2, s=1, c="#006AD0")



ax.set_xlabel("a")
ax.set_ylabel("d")
ax.set_zlabel("b")

# if you want to view the plot in an interactive window, deannotate the following command, but in this way it will not be saved.
# plt.show()

plt.savefig("GOE-2D.png")


# %% [markdown]
# Next we plot the eigenvalues on 2D of the exponentiated matrices: starting with $\Sigma=I$ and $\lambda_1>\lambda_2$.

# %%
lambda1 = [x[0] for x in lambda_goe_exp1]
lambda2 = [x[1] for x in lambda_goe_exp1]

fig = plt.figure()
plt.scatter(lambda2, lambda1, s=1)

x = np.linspace(0, 3, 30)
plt.plot(x, x, c="#F92301")

plt.xlabel("$\lambda_2(<\lambda_1)$")
plt.ylabel("$\lambda_1$")
plt.title("Eigenvalues Visualization For Exponentiated \n GOE Matrices With $\Sigma = I$ and $\lambda_1>\lambda_2$")

plt.xlim(0, 7)
# plt.ylim(0,5)
# plt.show()
plt.savefig("GOE_I_1.png")


# %% [markdown]
# Now we plot the eigenvalues on 2D of the exponentiated matrices where $\Sigma=I$ with $\lambda_1>\lambda_2$ or $\lambda_1\leq\lambda_2$ with equal possibility.

# %%
import random

lambda_goe_exp1_shuffle = []
for i, x in enumerate(lambda_goe_exp1):
    coin = random.uniform(0, 1)
    if coin > 0.5:
        lambda_goe_exp1_shuffle.append(x[::-1])
    else:
        lambda_goe_exp1_shuffle.append(x)

lambda1 = [x[0] for x in lambda_goe_exp1_shuffle]
lambda2 = [x[1] for x in lambda_goe_exp1_shuffle]

fig = plt.figure()
plt.scatter(lambda2, lambda1, s=1)
plt.xlim(0, 9)
plt.ylim(0, 9)
plt.xlabel("$\lambda_2$")
plt.ylabel("$\lambda_1$")
plt.title("Eigenvalues Visualization For Exponentiated \n GOE Matrices With $\Sigma = I$ and $\lambda_1>\lambda_2$")
# plt.show()
plt.savefig("GOE_I_2.png")

# %% [markdown]
# Now we plot the eigenvalues on 2D of the exponentiated GOE matrices where $\Sigma=\begin{bmatrix} 1 & -1\\-1 & 2\end{bmatrix}$ and $\lambda_1>\lambda_2$.

# %%
lambda1 = [x[0] for x in lambda_goe_exp2]
lambda2 = [x[1] for x in lambda_goe_exp2]

fig = plt.figure()
plt.scatter(lambda2, lambda1, s=1)

plt.xlabel("$\lambda_2(<\lambda_1)$")
plt.ylabel("$\lambda_1$")
plt.title("Eigenvalues Visualization For Exponentiated GOE Matrices \n With $\Sigma = [[1 -1], [-1 2]]$ and $\lambda_1>\lambda_2$")

plt.xlim(0, 10)
plt.ylim(0, 10)
# plt.show()
plt.savefig("GOE_notI_1.png")

# %% [markdown]
# At last, the eigenvalues on 2D of the exponentiated GOE matrices where $\Sigma=\begin{bmatrix} 1 & -1\\-1 & 2\end{bmatrix}$ and $\lambda_1, \lambda_2$ randomly sorted.

# %%
lambda_goe_exp2_shuffle = []
for i, x in enumerate(lambda_goe_exp2):
    coin = random.uniform(0, 1)
    if coin > 0.5:
        lambda_goe_exp2_shuffle.append(x[::-1])
    else:
        lambda_goe_exp2_shuffle.append(x)

lambda1 = [x[0] for x in lambda_goe_exp2_shuffle]
lambda2 = [x[1] for x in lambda_goe_exp2_shuffle]

fig = plt.figure()
plt.scatter(lambda2, lambda1, s=1)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel("$\lambda_2$")
plt.ylabel("$\lambda_1$")
plt.title("Eigenvalues Visualization For Exponentiated GOE Matrices \n With $\Sigma = [[1 -1], [-1 2]]$ and $\lambda_1, \lambda_2$ Randomly Sorted")
# plt.show()
plt.savefig("GOE_notI_2.png")


