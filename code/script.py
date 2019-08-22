import networkx as nx
import numpy as np
from graph_simplex import *
from simplex_plotter import *
import matplotlib.pyplot as plt 



d = 3
dynamics = False

if d == 2: 
	G = nx.Graph()
	G.add_nodes_from(range(1,3))
	G.add_edge(1,2)
	G.add_edge(2,3)


	#G = nx.complete_graph(3)
	evals,evecs = laplacian_decomposition(G)
	S, Sinv = simplex_vertices(evals, evecs)
	evalsn, evecsn = laplacian_decomposition(G,True)
	Sn, Sinvn = simplex_vertices(evalsn, evecsn)
	f = plot_simplex([S,Sn],2, ['b','g'])

	#c = centroid(S)
	#print(c)

	inits = [[1,0,0], [1,1,0], [0,0,1]]
	its,  eps = 10, 0.1
	# for i in range(its):
	# 	inits.append([1, eps*np.random.rand(), eps*np.random.rand()])


elif d== 3:
	
	G = nx.Graph()
	G.add_nodes_from(range(1,4))
	G.add_edge(1,2)
	#G.add_edge(1,3)
	G.add_edge(2,3)
	G.add_edge(3,4)
	#G.add_edge(4,2)
	G.add_edge(1,4)
	#G = nx.complete_graph(4)
	evals,evecs = laplacian_decomposition(G)
	S, Sinv = simplex_vertices(evals, evecs)
	evalsn, evecsn = laplacian_decomposition(G,True)
	Sn, Sinvn = simplex_vertices(evalsn, evecsn)
	f = plot_simplex([Sinv,Sinvn],3, ['r','y'])
	
	inits = [[1,0,0,0],
			 [0,1,0,1],
			 [0,0,1,1]] 
			# [0,1,1,1]]

	

iters = 100
#styles = ['r.', 'b.', 'g.', 'y.']
if dynamics: 
	for i,x in enumerate(inits): 
		s = sum(x)
		x0 = list(map(lambda i: i/float(s), x))
		x = ctrw(G, x0, iters)
		print(x)
		f = plot_rw(x[::1,:], S, f)


#f.title('Simplex of K4')
fig = plt.gcf()
f.show()
fig.savefig('../thesis/figures/ex')


#print('Centroid', centroid(S))




