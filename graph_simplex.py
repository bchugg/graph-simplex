import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def plot_simplex(S, d):
	# Plot simplex in two or three dimensions
	# S contains simplex vertices, d is dimension
	if not d==2 and not d==3: 
		print('Dimension must be two or three, brah')
	xdata = S[:,0]
	ydata = S[:,1]
	if d==3: 
		zdata = S[:,2]
		ax = plt.axes(projection='3d')
		for i in range(0,4):
			for j in range(1,4):
				if not i == j:
					x = [xdata[i], xdata[j]]
					y = [ydata[i], ydata[j]]
					z = [zdata[i], zdata[j]]
					ax.plot(x,y,z, '-o')

		#ax.plot(xdata, ydata, zdata, '-o')
		plt.show()
	else: 
		plt.plot(xdata, ydata, '-')
		plt.show()






G = nx.complete_graph(4)
w,v = laplacian_decomposition(G)
print('Eigenvalues:', w) 
print('Eigenvectors:', v)

S = simplex_vertices(w, v)
print('Simplex vertices', S)

plot_simplex(S,3)


