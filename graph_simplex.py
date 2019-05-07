import networkx as nx
import numpy as np

def laplacian_decomposition(G):
	# Compute the eigendecomposition of the Laplacian of the connected graph G
	# Return the non-zero eigenvalues and corresponding eigenvectors
	# Note: G should be a networkx graph

	# Gets eigenvalues and vectors of Laplacian
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
	# Return simplex vertices given non-zero eigenvalues and eigenvectors 
	# Vertices are encoded in matrix S, S(i,j) is j-th component of i-th vector
 
	n = len(evals) + 1
	S = np.zeros((n,n-1))
	for i in range(0,n):
		for j in range(0,n-1):
			S[i,j] = evecs[i,j] * np.sqrt(evals[j])

	return S

G = nx.complete_graph(10)
w,v = laplacian_decomposition(G)
print('Eigenvalues:', w) 
print('Eigenvectors:', v)

S = simplex_vertices(w, v)
print('Simplex vertices', S)


