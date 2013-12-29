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
