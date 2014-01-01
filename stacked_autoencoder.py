# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *
from scipy.sparse import *

from compute_numerical_gradient import computeNumericalGradient
import sparse_autoencoder


class Layer:
	
	def __init__(self, num):
		self.num = num
		self.W = None
		self.b = None
	
	
class NetConfig:
	
	def __init__(self):
		self.inputSize = 0
		self.layerSizes = []


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
	netConfig = NetConfig()
	if len(stack) > 0:
		prev_layer = None
		netConfig.inputSize = stack[0].W.shape[1]
		for layer in stack:
			assert layer.W.shape[0] == layer.b.shape[0], 'The bias should be a *column* vector of %i x1' % layer.W.shape[0]
			
			if prev_layer is not None:
				assert prev_layer.W.shape[0] == layer.W.shape[1], \
						'The adjacent layers L%i and L%i should have matching sizes.' % (prev_layer.num, layer.num)
						
			params = concatenate([params, layer.W.ravel(), layer.b.ravel()])
			netConfig.layerSizes.append(layer.W.shape[0])
	
			prev_layer = layer
	
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

		layer.W = params[curPos:curPos+layerSize * prevLayerSize].reshape(layerSize, prevLayerSize)
		curPos += layerSize * prevLayerSize

		layer.b = params[curPos:curPos+layerSize].ravel()
		curPos += layerSize
		
		prevLayerSize = layerSize
		
		stack.append(layer)
	
	return stack


def cost(thetaParam, inputSize, hiddenSize, numClasses, netConfig, lambdaParam, data, labels, corruptionLevel=0.0):
	"""Takes a trained softmaxTheta and a training data set with labels, and returns cost
	and gradient using a stacked autoencoder model. Used for finetuning.
	
	Keyword arguments:
	thetaParam -- trained weights from the autoencoder
	visibleSize -- the number of input units
	hiddenSize --  the number of hidden units *at the 2nd layer*
	numClasses --  the number of categories
	netConfig --   the network configuration of the stack
	lambdaParam -- the weight regularization penalty
	data -- our matrix containing the training data as columns.  So, data[i,:] is the i-th training example. 
	labels -- a vector containing labels, where labels[i] is the label for the i-th training example
	corruptionLevel -- how much of the input will get corrupted (denoising autoencoder)
	
	"""
	
	# We first extract the part which compute the softmax gradient
	softmaxTheta = thetaParam[0:hiddenSize*numClasses].reshape(numClasses, hiddenSize)
	stack = params2stack(thetaParam[hiddenSize*numClasses:], netConfig)
	
	m = data.shape[0]
	groundTruth = array(csc_matrix( (ones(m),(labels,range(m))), shape=(numClasses,m) ).todense())

	activation = data
	
	# Corrupt input data (so that denoising autoencoder can fix it)
	if corruptionLevel > 0.0:
		corruptionMatrix = random.binomial(1,1-corruptionLevel, size=activation.shape)
		activation = activation * corruptionMatrix

	# Forward propagation
	activations = []
	for layer in stack:
		activations.append(activation)
		activation = sparse_autoencoder.sigmoid(activation.dot(layer.W.T) + layer.b)

	# Back propagation
	M = softmaxTheta.dot(activation.T)
	M = M - amax(M, 0)
	h_data = exp(M)
	h_data = h_data / sum(h_data, 0)
	
	cost = -1.0/numClasses * sum(multiply(groundTruth, log(h_data))) + lambdaParam/2 * sum(softmaxTheta**2)
	softmaxThetaGrad = -1.0/numClasses * ((groundTruth - h_data).dot(activation)) + lambdaParam*softmaxTheta
	
	stackGrad = []
	delta = multiply(-(softmaxTheta.T.dot(groundTruth - h_data)), (activation * (1-activation)).T)
	idx = len(activations)
	while activations != []:
		activation = activations.pop()
		layer = Layer(idx)
		layer.W = (1.0/numClasses) * delta.dot(activation)
		layer.b = (1.0/numClasses) * sum(delta, 1)
		stackGrad.insert(0, layer)

		delta = multiply(stack[idx-1].W.T.dot(delta), (activation * (1-activation)).T)
		
		idx -= 1

	(params, netConfig) = stack2params(stackGrad)
	grad = concatenate([softmaxThetaGrad.ravel(), params])
	
	return (cost, grad)


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


if __name__ == "__main__":
	inputSize = 4
	hiddenSize = 5
	lambdaParam = 0.01
	data = random.normal(size=(5, inputSize))
	labels = array([0, 1, 0, 1, 0])
	numClasses = 2
	
	stack = [Layer(1), Layer(2)]
	stack[0].W = 0.1 * random.normal(size=(3, inputSize))
	stack[0].b = zeros(3)
	stack[1].W = 0.1 * random.normal(size=(hiddenSize, 3))
	stack[1].b = zeros(hiddenSize)

	softmaxTheta = 0.005 * random.normal(size=hiddenSize * numClasses)
	
	(stackParams, netConfig) = stack2params(stack)
	stackedAETheta = concatenate([softmaxTheta, stackParams])

	def stackedAutoencoderCostCallback(x):
		return cost(x, inputSize, hiddenSize, numClasses, netConfig,
				lambdaParam, data, labels)
				
	(cost_value, grad) = stackedAutoencoderCostCallback(stackedAETheta)

	numgrad = computeNumericalGradient(stackedAutoencoderCostCallback, stackedAETheta)
	
	diff = linalg.norm(numgrad-grad)/linalg.norm(numgrad+grad)

	print('%s' % diff)
	print('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n')
