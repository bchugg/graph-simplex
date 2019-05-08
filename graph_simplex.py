import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def laplacian_decomposition(G, normalized=False):
	# Compute the eigendecomposition of the Laplacian of the connected graph G
	# Return the non-zero eigenvalues and corresponding eigenvectors
	# If normalized=True, perform decomposition on normalized Laplacian
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

def plot_simplex(A, d, colors):
	# Plot simplex in two or three dimensions
	# A is array of simplices, d is dimension
	# color is color in which simplex is plotted
	if (not d==2 and not d==3) or not len(colors) == len(A): 
		print('Dimension must be two or three, and supply colors')
	if d == 3: 
		ax = plt.axes(projection='3d')

	for i in range(0,len(A)):
		S = A[i] # i-th simplex
		xdata = S[:,0]
		ydata = S[:,1]
		if d==3: 
			zdata = S[:,2]
			plot_simplex_helper3D(xdata, ydata, zdata, ax, colors[i])
		else: 
			plot_simplex_helper2D(xdata, ydata, c=colors[i])
		
	plt.show()

def plot_simplex_helper3D(xdata, ydata, zdata, ax, c):
	# Helper method for plot_simplex, for R^3 case
	plot_style = c+'-o'
	eps = 0.06

	# Plot lines between each pair of points
	for i in range(0,4):
		for j in range(1,4):
			if not i == j:
				x = [xdata[i], xdata[j]]
				y = [ydata[i], ydata[j]]
				z = [zdata[i], zdata[j]]
				ax.plot(x,y,z, plot_style) # Plot

	# Plot centroid
	ax.plot([0],[0],[0], 'ko')
	ax.text(0,0+eps,0,'c', size=12, color='k')
			
	# Label axes
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')


	# Annotate vertices
	for i in range(4):
		ax.text(xdata[i], ydata[i]+eps, zdata[i], 
			i, size=12, color=c)
		

def plot_simplex_helper2D(xdata, ydata, c='r'):
	# Helper method for plot_simplex, for R^2 case
	
	plot_style = c+'-o'
	eps = 0.06

	# Plot lines between each pair of points
	for i in range(0,3):
		for j in range(1,3):
			if not i == j:
				x = [xdata[i], xdata[j]]
				y = [ydata[i], ydata[j]]
				plt.plot(x,y,plot_style)

	# Plot centroid
	plt.plot(0,0,'o')
	plt.text(0,0+eps, 'c', size=12, color='k')

	# Label axes
	plt.xlabel('x')
	plt.ylabel('y')

	# Annotate vertices
	for i in range(3):
		plt.text(xdata[i], ydata[i]+eps,
			i, size=12, color=c)



G = nx.complete_graph(4)
w,v = laplacian_decomposition(G)
#print('Eigenvalues:', w) 
#print('Eigenvectors:', v)

S, Sinv = simplex_vertices(w, v)
print('Simplex vertices', S)
print('Inverse vertices', Sinv)

plot_simplex([S,Sinv],3, ['b', 'r'])




