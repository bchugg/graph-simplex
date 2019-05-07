import networkx as nx
import numpy as np

def laplacian_decomposition(G):
	# Compute the eigendecomposition of the Laplacian of the connected graph G
	# Return the non-zero eigenvalues and corresponding eigenvectors
	# Note: G should be a networkx graph

	# Gets eigenvalues and vectors of Laplacian
	L = nx.laplacian_matrix(G).todense()
	w, v = np.linalg.eig(L)
	w = list(w)

	# Min eigenvalue is only approx 0 due to imprecision
	# Find index of this minimum eigenvalue
	abs_evals = list(map(lambda x: abs(x), w))
	min_eval_index = abs_evals.index(min(abs_evals))
	
	# Remove evec corresponding to eval 0
	eig_range = list(range(0,len(w)))
	eig_range.pop(min_eval_index)
	v = v[:, eig_range] 
	w.pop(min_eval_index) # Remove eval 0

	return w,v

G = nx.complete_graph(10)
w,v = laplacian_decomposition(G)
print('Eigenvalues:', w) 
print('Eigevectors:', v)


