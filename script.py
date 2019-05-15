import networkx as nx
import numpy as np
from graph_simplex import *
from simplex_plotter import *



d = 4

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
	plot_simplex([S,Sinv],3, ['b', 'r'])


#G = nx.binomial_graph(30,0.6)
#print('Simplex vertices', S)
#print('Inverse vertices', Sinv)

#print('Centroid', centroid(S))

# Test random walk 

N = 20
G = nx.complete_graph(N)
iters = 10000
x0 = np.random.rand(N)
s = sum(x0)
x0 = list(map(lambda i: i/float(s), x0))

x = ctrw(G, x0, iters)
print(x[iters-1,:])


