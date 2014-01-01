# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *

def computeNumericalGradient(J, theta):
	
	numgrad = zeros(theta.shape)
	
	EPSILON = 1e-04
	
	bases = eye(numgrad.shape[0])
	
	for i in range(numgrad.shape[0]):
		(value1, grad1) = J(theta + EPSILON*bases[:,i])
		(value2, grad2) = J(theta - EPSILON*bases[:,i])
		numgrad[i] = (value1 - value2) / (2*EPSILON)

	return numgrad


if __name__ == "__main__":
	""" Check correctness of implemenation of computeNumericalGradient
	on an example of simple quadratic function
	"""
	
	def simpleQuadraticFunction(x):
		value = x[0]**2 + 3*x[0]*x[1]
	
		grad = zeros(2)
		grad[0]  = 2*x[0] + 3*x[1]
		grad[1]  = 3*x[0]
	
		return (value, grad)
	
	
	x = array([4, 10]).T
	
	(value, grad) = simpleQuadraticFunction(x);
	
	numgrad = computeNumericalGradient(simpleQuadraticFunction, x)
	
	diff = linalg.norm(numgrad-grad)/linalg.norm(numgrad+grad)
	
	print('%s' % diff)
	print('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n')