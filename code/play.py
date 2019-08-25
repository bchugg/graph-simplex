import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 
from functools import reduce

def Tcomb(G):
	# Compute combinatorial metric of graph  G

	L = nx.laplacian_matrix(G).todense() 

	# Delete i-th row, i-th col from L
	Q = np.delete(np.delete(L,1,1),1,0) 

	return np.abs(np.linalg.det(Q))

def Tnorm(G):
	# Compute normalized metric of G

	L = nx.normalized_laplacian_matrix(G).todense() 

	# Delete i-th row, i-th col from L
	Q = np.delete(np.delete(L,1,1),2,0) 

	di = G.degree(1)
	dj = G.degree(2)
	return  np.abs(np.linalg.det(Q) / float(np.sqrt(di) * np.sqrt(dj)))



# Weights increase uniformly

# G = nx.karate_club_graph()
# weight_range = [i for i  in range(1,15)]
# Tnorms = []

# for w in weight_range:
# 	F  = nx.Graph()
# 	for (u,v) in G.edges():
# 		F.add_edge(u,v,weight=w)
# 	Tnorms.append(Tnorm(F))

# plt.plot([i for i in range(1,15)], Tnorms)
# plt.show()
# print(Tnorms)


# Single node increases  in weight

# G = nx.complete_graph(15)
# baseline = 1
# for (u,v) in  G.edges():
# 	G.edges[u,v].update(weight=baseline)

# weight_range = [i for i  in range(1,100)]
# Tnorms = []
# Tnorms2 = []


# for w in weight_range:
# 	G.edges[0,4].update(weight=w)
# 	Tnorms.append(Tnorm(G))
# 	G.edges[0,4].update(weight=baseline)
	
# print(Tnorms)
# plt.plot([i for i in range(1,100)], Tnorms)
# plt.show()


G = nx.complete_graph(10)
num_edges = len(G.edges())
Tnorms = []

for i in range(20):
	edges = list(G.edges())
	while True: 
		r = np.random.randint(1,len(edges))
		G.remove_edge(*edges[r])
		if nx.is_connected(G):
			break
	Tnorms.append(Tnorm(G))

plt.plot([i for i in range(num_edges,num_edges-20,-1)], Tnorms)
plt.show()






# Tcombs = []
# Tnorms = []

# for n in range(5,15):
# 	G = nx.complete_graph(n) 
# 	volG = sum([G.degree(i) for i in range(1,n)])
# 	Tcombs.append(Tcomb(G))
# 	Tnorms.append(Tnorm(G))
	

# plt.plot([i for i  in range(5,15)], Tnorms)
# plt.show()
#print(Tcombs)
#print(Tnorms)




# n = 50
# # G = nx.karate_club_graph()
# # #G = nx.davis_southern_women_graph()
# G =  nx.complete_graph(n)

# L = nx.normalized_laplacian_matrix(G).todense() # Normalized 
# #L = nx.laplacian_matrix(G).todense() # Combinatorial

# indices = [1,5,7,15,23]
# for i in indices:
# 	for j in indices: 
# 		#Q = np.delete(L,i,0) # Delete i-th row of L  
# 		#Q = np.delete(Q,j,1) # Delete j-th column  of L 

# 		Q = np.delete(np.delete(L,i,0),j,1)

# 		det = np.linalg.det(Q)
# 		di = G.degree(i)
# 		dj = G.degree(j)
# 		print(det / float(np.sqrt(di) * np.sqrt(dj)))
# 		print(Tnorm(G))




	
