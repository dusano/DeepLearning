# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *

import sparse_autoencoder


class Layer:
	W = None
	b = None
	
	def __init__(self, num):
		self.num = num
	
	
class NetConfig:
	inputSize = 0
	layerSizes = []


def stack2params(stack):
	"""Converts a "stack" structure into a flattened parameter vector and also
	stores the network configuration. This is useful when working with
	optimization toolboxes such as minFunc.
	
	Keyword arguments:
	stack - the stack structure, where stack{1}.w = weights of first layer
	                                   stack{1}.b = weights of first layer
	                                   stack{2}.w = weights of second layer
	                                   stack{2}.b = weights of second layer
	                                   ... etc.
	 
	""" 
	params = array([])
	prev_layer = None
	for layer in stack:
		params = concatenate([params, layer.W.ravel(), layer.b.ravel()])
	
		assert layer.W.shape[0] == layer.b.shape[0], 'The bias should be a *column* vector of %i x1' % layer.W.shape[0]
		
		if prev_layer is not None:
			assert prev_layer.W.shape[0] == layer.W.shape[1], \
					'The adjacent layers L%i and L%i should have matching sizes.' % (prev_layer.num, layer.num)

		prev_layer = layer
	
	netConfig = NetConfig()
	if len(stack) > 0:
		netConfig.inputSize = stack[0].W.shape[1]
		for layer in stack:
			netConfig.layerSizes.append(layer.W.shape[0])
	
	return (params, netConfig)


def params2stack(params, netConfig):
	"""Converts a flattened parameter vector into a nice "stack" structure 
	for us to work with. This is useful when you're building multilayer
	networks.
	
	Keyword arguments:
	params -- flattened parameter vector
	netConfig -- auxiliary variable containing the configuration of the network
	
	"""
	stack = []
	layerNum = 0
	prevLayerSize = netConfig.inputSize
	curPos = 0
	for layerSize in netConfig.layerSizes:
		layerNum += 1
		layer = Layer(layerNum)
		
		wlen = layerSize * prevLayerSize
		layer.W = params[curPos:curPos+wlen].reshape(layerSize, prevLayerSize)
		curPos += wlen
		
		blen = layerSize
		layer.b = params[curPos:curPos+blen].ravel()
		curPos += blen
		
		prevLayerSize = layerSize
		
		stack.append(layer)
	
	return stack


def predict(thetaParam, inputSize, hiddenSize, numClasses, netConfig, data):
	"""Takes a trained theta and a test data set, and returns the predicted labels for each example.
	
	Keyword arguments:
	thetaParam -- trained weights from the autoencoder
	inputSize -- the number of input units
	hiddenSize -- the number of hidden units *at the 2nd layer*
	numClasses -- the number of categories
	netConfig - configuration of the neural network
	data -- our matrix containing the training data as columns.  So, data[i,:] is the i-th training example.
	
	"""

	softmaxTheta = thetaParam[0:hiddenSize*numClasses].reshape(numClasses, hiddenSize)
	stack = params2stack(thetaParam[hiddenSize*numClasses:], netConfig)

	activation = data
	for layer in stack:
		activation = sparse_autoencoder.sigmoid(activation.dot(layer.W.T) + layer.b)
		
	h_data = exp(softmaxTheta.dot(activation.T))
	h_data = h_data / sum(h_data, 0)
	return argmax(h_data, axis=0)
