import networkx as nx
import numpy as np
from graph_simplex import *
from simplex_plotter import *



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
	evals,evecs = laplacian_decomposition(G, True)
	S, Sinv = simplex_vertices(evals, evecs)
	plot_simplex([S,Sinv],3, ['b', 'r'])


#G = nx.binomial_graph(30,0.6)
#print('Simplex vertices', S)
#print('Inverse vertices', Sinv)

#print('Centroid', centroid(S))


