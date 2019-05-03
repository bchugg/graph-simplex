import networkx as nx
import numpy as np

def laplacian_decomposition(G):
	# Compute the eigendecomposition of the Laplacian of the connected graph G
	# Return the non-zero eigenvalues and corresponding eigenvectors
	# Note: G should be a networkx graph
	L = nx.laplacian_matrix(G).todense()
	print(L)
	w, v = np.linalg.eig(L)
	print(w)
	print(v)
	v[:, range(0,len(w)).pop(w.index(0))] # Remove evec corresponding to eval 0
	w.remove(0) # Remove eval 0

	return w,v

G = nx.complete_graph(10)
w,v = laplacian_decomposition(G)
print(w) 
print(v)


