import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np  


def plot_simplex(A, d, colors):
	# Plot simplex in two or three dimensions
	# A is array of simplices, d is dimension
	# color is color in which simplex is plotted
	f = plt.figure()
	#f.add_subplot(111)

	if (not d==2 and not d==3) or not len(colors) == len(A): 
		print('Dimension must be two or three, and supply colors')
		return
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

	
	return plt

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
	#ax.plot([0],[0],[0], 'ko')
	#ax.text(0,0+eps,0,'c', size=12, color='k')
			
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
	#plt.plot(0,0,'ko')
	#plt.text(0,0+eps, 'c', size=12, color='k')

	# Label axes
	plt.xlabel('x')
	plt.ylabel('y')
	plt.axis('scaled')

	# Annotate vertices
	for i in range(3):
		plt.text(xdata[i], ydata[i]+eps,
			i, size=12, color=c)



def plot_rw(x, S, fig, style='r.'):
	# Plot the random walk as points in simplex
	# In particular, each x[i] is a barycentric coordinate of simplex
	# described by vertices in S. 

	color = np.random.rand(3) # Generate random color
	N = len(x[0,:])
	St = np.transpose(S) 
	if N == 4: 
		for i in range(len(x[:,0])):
			coords = np.dot(St, x[i,:])
			fig.plot([coords[0]], [coords[1]], [coords[2]], color=color, marker='.')
	elif N == 3: 
		for i in range(len(x[:,0])):
			coords = np.dot(St, x[i,:])
			fig.plot([coords[0]], [coords[1]], color=color, marker='.')

	#fig.show()
	return fig

