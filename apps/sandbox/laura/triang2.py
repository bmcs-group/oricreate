import numpy as np

n=4
    
x, y = np.mgrid[0:n + 1, 0:2]
z = np.zeros_like(x)
X = np.c_[x.flatten(), y.flatten(), z.flatten()]
N = np.arange((2*n)+2).reshape(-1, 2)
L1 = np.c_[N[:-1, 0], N[1:, 1]]
L2 = np.c_[N[0:, 0], N[0:, 1]]
L3 = np.c_[N[:-1, 0], N[1:, 0]]
L4 = np.c_[N[:-1, 1], N[1:, 1]]
L = np.vstack([L1, L2, L3, L4])
F1 = np.c_[N[:-1, 0], N[1:, 0], N[1:, 1]]
F2 = np.c_[N[:-1, 0], N[1:, 1], N[:-1, 1]]
F = np.vstack([F1, F2])
