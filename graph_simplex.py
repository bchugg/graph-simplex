import networkx as nx
import numpy as np

def laplacian_decomposition(G, normalized=False):
	# Compute the eigendecomposition of the Laplacian of the connected graph G
	# Return the non-zero eigenvalues and corresponding eigenvectors
	# If normalized=True, perform decomposition on normalized Laplacian
	# Note: G should be a networkx graph

	# Gets eigenvalues and vectors of Laplacian
	if normalized: 
		L = nx.normalized_laplacian_matrix(G).todense()
	else: 
		L = nx.laplacian_matrix(G).todense()
	evals, evecs = np.linalg.eig(L)
	evals = list(evals)

	# Min eigenvalue is only approx 0 due to imprecision
	# Find index of this minimum eigenvalue
	abs_evals = list(map(lambda x: abs(x), evals))
	min_eval_index = abs_evals.index(min(abs_evals))
	
	# Remove evec corresponding to eval 0
	eig_range = list(range(0,len(evals)))
	eig_range.pop(min_eval_index)
	evecs = evecs[:, eig_range] 
	evals.pop(min_eval_index) # Remove eval 0

	return evals,evecs


def simplex_vertices(evals, evecs):
	# Return vertices of simplex and inverse simplex
	# Evals are non-zero eigenvalues of Laplacian, evecs are 
	# corresponding eigenvectors
	# Vertices are encoded in matrix S, S(i,j) is j-th component of i-th vector
	# inverse vertices are similarly encoded in matrix Sinv
 
	n = len(evals) + 1
	S = np.zeros((n,n-1))
	Sinv = np.zeros((n,n-1))
	for i in range(0,n):
		for j in range(0,n-1):
			S[i,j] = evecs[i,j] * np.sqrt(evals[j])
			Sinv[i,j] = evecs[i,j] / float(np.sqrt(evals[j]))

	return S, Sinv

def centroid(S):
	# Compute centroid of simplex S
	cent = np.zeros(len(S[0,:]))
	for i in range(len(cent)):
		cent[i] = sum(S[:,i])
	return cent / float(len(cent)+1) # Normalize by n









