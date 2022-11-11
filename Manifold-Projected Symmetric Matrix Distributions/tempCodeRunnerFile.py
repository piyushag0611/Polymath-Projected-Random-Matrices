ax.scatter3D(X_a, X_d, X_b)

x, y, z = zip(get_adb(X_exp), get_adb(X_exp_Sigma1), get_adb(X_exp_Sigma2), get_adb(invS1_X), get_adb(invS2_X), get_adb(exp_invS1_X), get_adb(exp_invS2_X))
ax.scatter3D(x, y, z)