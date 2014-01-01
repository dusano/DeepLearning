# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *
from scipy.sparse import *

from compute_numerical_gradient import computeNumericalGradient


def cost(thetaParam, numClasses, inputSize, lambdaParam, data, labels):
	"""Compute the cost and gradient for softmax regression.
	
	Keyword arguments:
	thetaParam -- a vector of parameters
	numClasses -- the number of classes 
	inputSize -- the size N of the input vector
	lambdaParam -- weight decay parameter
	data - the N x M input matrix, where each column data(:, i) corresponds to a single test set
	labels - an M x 1 matrix containing the labels corresponding for the input data
	
	"""

	# Unroll the parameters from theta
	thetaParam = thetaParam.reshape(numClasses, inputSize)
	
	m = data.shape[0]

	groundTruth = csc_matrix( (ones(m),(labels,range(m))), shape=(numClasses,m) ).todense()
	cost = 0

	M = thetaParam.dot(data.T)
	M = M - amax(M, 0)
	h_data = exp(M)
	h_data = h_data / sum(h_data, 0)
	
	cost = -sum(multiply(groundTruth, log(h_data)))/m + lambdaParam/2 * sum(thetaParam**2)

	thetaGrad = -((groundTruth - h_data).dot(data))/m + lambdaParam*thetaParam

	return (cost, squeeze(array(thetaGrad.ravel())))


def predict(thetaParam, data):
	"""Compute pred using theta
	
	Keyword arguments:
	optTheta -- this provides a numClasses x inputSize matrix
	data -- the N x M input matrix, where each column data(:, i) corresponds to a single test set
	"""	
	h_data = exp(thetaParam.dot(data.T))
	h_data = h_data / sum(h_data, 0)
	return argmax(h_data, axis=0)
	

if __name__ == "__main__":
	""" Check correctness of implemenation of softmax cost function
	using gradient check
	"""
	numClasses = 10			# Number of classes (MNIST images fall into 10 classes)
	lambdaParam = 1e-4		# Weight decay parameter
	inputSize = 8
	inputData = random.normal(size=(100,inputSize))
	labels = random.randint(10, size=100)

	def softmaxCostCallback(x):
		return cost(x, numClasses, inputSize, lambdaParam, inputData, labels) 
	
	# Randomly initialise theta
	thetaParam = 0.005 * random.normal(size=numClasses * inputSize)

	(cost_value, grad) = softmaxCostCallback(thetaParam)
	
	numGrad = computeNumericalGradient(softmaxCostCallback, thetaParam)
	
	# Compare numerically computed gradients with those computed analytically
	diff = linalg.norm(numGrad-grad)/linalg.norm(numGrad+grad)
	
	print('%s' % diff)
	print('Norm of the difference between numerical and analytical gradient (should be < 1e-7)\n\n')
