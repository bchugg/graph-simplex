import networkx as nx
import numpy as np
from graph_simplex import *
from simplex_plotter import *
import matplotlib.pyplot as plt 



d = 3

if d == 2: 
	G = nx.Graph()
	G.add_nodes_from(range(1,3))
	G.add_edge(1,2)
	G.add_edge(2,3)

	#G = nx.complete_graph(3)
	evals,evecs = laplacian_decomposition(G, True)
	S, Sinv = simplex_vertices(evals, evecs)
	plot_simplex([S,Sinv],2, ['b', 'r'])
elif d== 3:
	#G = nx.complete_graph(4)
	G = nx.Graph()
	G.add_nodes_from(range(1,4))
	G.add_edge(1,2)
	G.add_edge(2,3)
	G.add_edge(3,4)
	G.add_edge(1,4)
	evals,evecs = laplacian_decomposition(G, True)
	S, Sinv = simplex_vertices(evals, evecs)
	f = plot_simplex([S],3, ['b'])
	#f.show()


#G = nx.binomial_graph(30,0.6)
#print('Simplex vertices', S)
#print('Inverse vertices', Sinv)

#print('Centroid', centroid(S))

# Test random walk 

#N = 4
#G = nx.complete_graph(N)
iters = 10000
#x0 = np.random.rand(d+1)
x0 = [1, 0,0, 0]
s = sum(x0)
x0 = list(map(lambda i: i/float(s), x0))

x = ctrw(G, x0, iters)
print(x[iters-1,:])


fig = plot_rw(x[::250,:], S, f)
#fig.show()


