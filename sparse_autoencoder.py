# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *

from compute_numerical_gradient import computeNumericalGradient
from image_patches import getPatches


def sigmoid(x):
	return 1 / (1 + exp(-x))
	

def feedForward(thetaParam, hiddenSize, visibleSize, data):
	"""Compute the activation of the hidden layer for the Sparse Autoencoder.
	
	Keyword arguments:
	thetaParam -- trained weights from the autoencoder
	visibleSize -- the number of input units (probably 64) 
	hiddenSize -- the number of hidden units (probably 25) 
	data -- our matrix containing the training data as columns. So, data[i,:] is the i-th training example. 

	"""
	
	W1 = thetaParam[0:hiddenSize*visibleSize].reshape(hiddenSize, visibleSize)
	b1 = thetaParam[2*hiddenSize*visibleSize:2*hiddenSize*visibleSize+hiddenSize]

	return sigmoid(data.dot(W1.T) + b1)


def cost(thetaParam, visibleSize, hiddenSize, lambdaParam, sparsityParam, betaParam, data):
	""" Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
	and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
	
	Keyword arguments:
	thetaParam -- a vector of parameters (W1, W2, b1, b2)
	visibleSize -- the number of input units (probably 64) 
	hiddenSize -- the number of hidden units (probably 25) 
	lambdaParam -- weight decay parameter
	sparsityParam -- the desired average activation for the hidden units
	betaParam -- weight of sparsity penalty term
	data -- a matrix containing the training data. So, data[i,:] is the i-th training example. 
	
	"""

	W1 = thetaParam[0:hiddenSize*visibleSize].reshape(hiddenSize, visibleSize)
	W2 = thetaParam[hiddenSize*visibleSize:2*hiddenSize*visibleSize].reshape(visibleSize, hiddenSize)
	b1 = thetaParam[2*hiddenSize*visibleSize:2*hiddenSize*visibleSize+hiddenSize]
	b2 = thetaParam[2*hiddenSize*visibleSize+hiddenSize:]
	
	m = data.shape[0]

	# Forward propagation
	a2 = sigmoid(data.dot(W1.T) + b1)
	a3 = sigmoid(a2.dot(W2.T) + b2)
		
	# Back propagation
	mean_a2 = mean(a2,0)
	
	sparsity_delta = (-sparsityParam / mean_a2) + (1-sparsityParam)/(1-mean_a2)
	
	delta3 = -(data - a3) * (a3 * (1-a3))
	delta2 = (delta3.dot(W2) + betaParam*sparsity_delta) * (a2 * (1-a2))

	W1grad = (delta2.T.dot(data))/m + lambdaParam * W1
	b1grad = sum(delta2, 0)/m
	W2grad = (delta3.T.dot(a2))/m + lambdaParam * W2
	b2grad = sum(delta3, 0)/m
	
	cost = sum((a3 - data)**2)/2
		
	weight_decay = sum(W1**2) + sum(W2**2)
	
	sparsity_penalty = sparsityParam*log(sparsityParam/mean_a2) + \
						(1-sparsityParam)*log((1-sparsityParam) / (1-mean_a2))
						
	cost = cost/m + (lambdaParam/2) * weight_decay + betaParam * sum(sparsity_penalty)
	
	grad = concatenate([W1grad.ravel(), W2grad.ravel(), b1grad.ravel(), b2grad.ravel()])

	return (cost, grad)


def initializeParameters(hiddenSize, visibleSize):
	# Initialize parameters randomly based on layer sizes.
	
	# we'll choose weights uniformly from the interval [-r, r]
	r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1)
	
	W1 = random.rand(hiddenSize, visibleSize) * 2 * r - r;
	W2 = random.rand(visibleSize, hiddenSize) * 2 * r - r;
	
	b1 = zeros((hiddenSize, 1));
	b2 = zeros((visibleSize, 1));
	
	# Convert weights and bias gradients to the vector form.
	# This step will "unroll" (flatten and concatenate together) all 
	# your parameters into a vector, which can then be used with minFunc. 
	theta = concatenate([W1.ravel(), W2.ravel(), b1.ravel(), b2.ravel()])
	
	return theta


if __name__ == "__main__":
	""" Check correctness of implemenation of sparse_autoencoder cost function
	using gradient check
	"""
	patchSize=8
	visibleSize = patchSize*patchSize		# number of input units 
	hiddenSize = 25							# number of hidden units
	sparsityParam = 0.01					# desired average activation of the hidden units.
	lambdaParam = 0.0001					# weight decay parameter       
	betaParam = 3							# weight of sparsity penalty term

	patches = getPatches(numPatches=10, patchSize=patchSize)

	# Obtain random parameters theta
	thetaParam = initializeParameters(hiddenSize, visibleSize)
	
	def sparseAutoencoderCostCallback(x):
		return cost(x, visibleSize, hiddenSize, lambdaParam, sparsityParam,
					betaParam, patches) 
	
	(cost_value, grad) = sparseAutoencoderCostCallback(thetaParam)
	
	numgrad = computeNumericalGradient(sparseAutoencoderCostCallback, thetaParam);
	diff = linalg.norm(numgrad-grad)/linalg.norm(numgrad+grad)
	
	print('%s' % diff)
	print('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n')