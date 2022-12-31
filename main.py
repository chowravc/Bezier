### Importing useful packages
import operator as op
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

### Combinations function
def ncr(n, r):
	r = min(r, n-r)
	numer = reduce(op.mul, range(n, n-r, -1), 1)
	denom = reduce(op.mul, range(1, r+1), 1)
	return numer // denom  # or / in Python 2

### Object of type bezier curve
class bezier:

	## Constructor for bezier curve
	def __init__(self, points):

		# Bezier curve order
		self.order = points.shape[0] - 1

		# Dimensionality of curve
		self.dims = points.shape[1]

		# Points defining curve
		self.points = points

	## Function to obtain curve
	def curve(self, nP=int(1e3)):

		# Creating list of parameters
		t = np.linspace(0, 1, num=nP)

		# Order of curve
		n = self.order

		# List to store coordinates of curve
		coords = []

		# Go through each dimension
		for dim in range(self.dims):

			# Initialize coordinates in dimension
			dCords = np.zeros(nP)

			# Go through explicit formula for bezier curve
			for i in range(n+1):

				# Add interpolated terms
				dCords += ncr(n, i)*((1-t)**(n-i))*(t**i)*(self.points.T[dim][i])

			# Add dimension to coordinates
			coords.append(dCords)

		# Return coordinates
		return np.asarray(coords)


### Run when script is run directly
if __name__ == '__main__':

	## Points to interpolate
	# ps = np.array([[0, 0], [1, 0], [1, 1]])
	ps = np.array([[-10, -10], [-6, 6], [2, -6], [4, 4], [8, -4]])
	
	## Creating bezier curve
	b1 = bezier(ps)

	## Obtaining coords for curve
	pbs = b1.curve()

	## Plotting bezier curve
	plt.plot(pbs[0], pbs[1], zorder=1, label='curve')
	
	## Transpose of points
	ps = ps.T

	## Plotting points
	plt.scatter(ps[0], ps[1], s=5, color='k', zorder=2, label='points')

	# plt.xlim(-0.1, 1.1)
	# plt.ylim(-0.1, 1.1)

	## Equal axes
	plt.gca().set_aspect('equal', adjustable='box')

	## Show legend
	plt.legend()

	## Add axes labels
	plt.xlabel('x')
	plt.ylabel('y')

	## Add plot title
	plt.title('Bezier curve')

	## Displaying plot
	plt.show()