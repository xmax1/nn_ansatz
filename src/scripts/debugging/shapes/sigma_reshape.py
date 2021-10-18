import numpy as np
n_det = 10
n_j = 7
x = [(1, 2, 3) for i in range(n_det)]
x = [i for lst in x for i in lst]
x = np.array([x for j in range(n_j)])
print(x)
y = x.reshape(n_j, 3, -1, order='F')
print(y[0])
z = x.reshape(n_j, 3, -1)
print(z[0])
print(np.linalg.norm(y, axis=1), '\n', np.linalg.norm(z, axis=1))