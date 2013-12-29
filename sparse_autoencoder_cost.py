# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *


def sigmoid(x):
	return 1 / (1 + exp(-x))


def sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambdaParam, 
		sparsityParam, betaParam, data):
	""" Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
	and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
	
	Keyword arguments:
	theta -- a vector of parameters (W1, W2, b1, b2)
	visibleSize -- the number of input units (probably 64) 
	hiddenSize -- the number of hidden units (probably 25) 
	lambdaParam -- weight decay parameter
	sparsityParam -- the desired average activation for the hidden units
	betaParam -- weight of sparsity penalty term
	data -- a matrix containing the training data. So, data(:,i) is the i-th training example. 
	
	"""

	W1 = theta[0:hiddenSize*visibleSize].reshape(hiddenSize, visibleSize)
	W2 = theta[hiddenSize*visibleSize:2*hiddenSize*visibleSize].reshape(visibleSize, hiddenSize)
	b1 = theta[2*hiddenSize*visibleSize:2*hiddenSize*visibleSize+hiddenSize]
	b2 = theta[2*hiddenSize*visibleSize+hiddenSize:]
	
	# Cost and gradient variables Here, we initialize them to zeros. 
	cost = 0
	W1grad = zeros(W1.shape) 
	W2grad = zeros(W2.shape)
	b1grad = zeros(b1.shape) 
	b2grad = zeros(b2.shape)
	
	m = data.shape[1]
	
	a2 = zeros((W1.shape[0], m))
	a3 = zeros((W2.shape[0], m))
	
	for i in range(m):
		z2 = W1.dot(data[:,i]) + b1
		a2[:,i] = sigmoid(z2)
		
		z3 = W2.dot(a2[:,i]) + b2
		a3[:,i] = sigmoid(z3)
		
	mean_a2 = mean(a2,1)
	
	for i in range(m):
		delta3 = -(data[:,i] - a3[:,i]) * (a3[:,i] * (1-a3[:,i]))
		sparsify = (-sparsityParam / mean_a2) + (1-sparsityParam) / (1-mean_a2)
		delta2 = (W2.T.dot(delta3) + betaParam*sparsify) * (a2[:,i] * (1-a2[:,i]))

		W1grad += outer(delta2, data[:,i])
		b1grad += delta2
		W2grad += outer(delta3, a2[:,i])
		b2grad += delta3
		
		cost = cost + (linalg.norm(a3[:,i] - data[:,i])**2) / 2
		
	weight_decay = sum(W1.ravel()**2) + sum(W2.ravel()**2)
	
	sparsity_penalty = sparsityParam*log(sparsityParam/mean_a2) + \
						(1-sparsityParam)*log((1-sparsityParam) / (1-mean_a2))
						
	cost = cost/m + (lambdaParam/2) * weight_decay + betaParam * sum(sparsity_penalty)
 
	W1grad = W1grad/m + lambdaParam * W1
	b1grad = b1grad/m
	W2grad = W2grad/m + lambdaParam * W2
	b2grad = b2grad/m
	
	grad = concatenate([W1grad.ravel(), W2grad.ravel(), b1grad.ravel(), b2grad.ravel()])

	return (cost, grad)
