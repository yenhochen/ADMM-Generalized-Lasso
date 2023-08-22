'''
Author: Yenho Chen

This code corresponds with the following blog post.

    https://yenhochen.github.io/blog/2023/ADMM-Generalized-Lasso/


An excellent resource for algorithmic details is

    Boyd 2010
    Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers
    https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
    
'''

import numpy as np
import numpy.random as npr
from sklearn.datasets import make_sparse_coded_signal
import matplotlib.pyplot as plt

def soft_threshold(x, thresh):
    return np.sign(x)* np.max([np.abs(x)-thresh, np.zeros(len(x))], axis=0)

def ADMM_generalized_lasso(y, D, F, rho, lam, n_iters=100):
    '''
    y: (m,) array. observation
    D: (m, n) array. dictionary
    F: (k, n) array. constraint matrix
    rho: augmented lagrange multiplier
    lam: lagrange multiplier
    
    '''
    n = len(D.T)
    w = npr.randn(n) 
    u = npr.randn(len(F)) 
    z = npr.randn(len(F)) 
    
    FtF = F.T @ F # precompute
    for i in range(n_iters):

        w = np.linalg.lstsq(D.T @ D + rho * FtF, D.T @ y + rho * F.T @ (z-u), rcond=None)[0]
        z = soft_threshold(F @ w + u, lam/rho)
        u = u + F @ w - z
        
    return w

# hyperparameters
n, m = 50, 100
n_nonzero_coefs = 10

# generate the data
y, D, w = make_sparse_coded_signal(
    n_samples=1,
    n_components=n,
    n_features=m,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=1,
)
D = D.T

np.random.seed(1)
# generate structured coefficients
w_true = np.zeros(n)
for i in range(5):
    ix = np.random.choice(n)
    length = np.random.randint(5,10)
    w_true[ix:ix+length] = npr.randn()
    
    
# generate noisy observations 
y_true = D @ w_true
y = y_true + npr.randn(m)*0.2


# define hyperparameters
rho = 0.3 # augmentation multiplier
lam = 0.5 # general multiplier for L1
lam2 = 0.3 # multipler for sparsity in Fused Lasso

# construct F matrices
F_Lasso = np.eye(n) # Lasso solution
F_fusion = (np.diag(np.ones(n),k=1)[:-1,:-1] + np.diag(np.ones(n)*-1,k=0))[:-1] # Fusion Penalty solution
F_fusedLasso = np.concatenate([np.diag(np.ones(n)/lam*lam2), F_fusion]) # Fused Lasso solution


# compute ADMM solution
w_lasso = ADMM_generalized_lasso(y, D, F_Lasso, rho, lam)
w_fusion = ADMM_generalized_lasso(y, D, F_fusion, rho, lam)
w_fusedLasso = ADMM_generalized_lasso(y, D, F_fusedLasso, rho, lam)


    
# Plot
plt.figure(figsize=(6,3), dpi=150)
plt.plot(w_lasso, label="Lasso")
plt.plot(w_fusion, label="Fusion")
plt.plot(w_fusedLasso, label="Fused Lasso")
plt.plot(w_true, 'k--', label="true")
plt.ylabel("Weight")
plt.xlabel("index")
plt.legend()
plt.tight_layout()
plt.savefig("ADMM-lasso-results.png")
plt.show()
    
