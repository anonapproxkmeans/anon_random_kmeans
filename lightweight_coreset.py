import numpy as np

SEED = None

def lightweight_coreset(X, m, seed = SEED, ret_indices = False):
	""" Function that takes in dataset and sample size m, 
	produces a lightweight coreset for k-means as described in 
	the lightweight coresets paper. Taken from youtube."""
	if seed:
		np.random.seed(seed)
	dist = np.power(X - X.mean(axis = 0), 2).sum(axis = 1)	
	
	q = 0.5/X.shape[0] + 0.5*dist/dist.sum()
	indices = np.random.choice(X.shape[0], size = m, replace = True, p = q)
	# indices = np.random.choice(X.shape[0], size = m, replace = True)
	X_lwcs = X[indices, :]
	w_lwcs = 1.0/(m*q[indices])
	# w_lwcs = np.array([1]*m)
	if ret_indices:
		return X_lwcs, w_lwcs, indices 
	return X_lwcs, w_lwcs
