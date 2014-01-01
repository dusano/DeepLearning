# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *
from scipy import optimize

from image_patches import getPatches
from visualize_network import visualizeNetwork
import sparse_autoencoder

patchSize=8
visibleSize = patchSize*patchSize		# number of input units 
hiddenSize = 25							# number of hidden units 
sparsityParam = 0.01					# desired average activation of the hidden units.
lambdaParam = 0.0001					# weight decay parameter       
betaParam = 3							# weight of sparsity penalty term

def sparseAutoencoderCostCallback(x):
	return sparse_autoencoder.cost(x, visibleSize, hiddenSize, lambdaParam, sparsityParam,
				betaParam, patches) 

patches = getPatches(numPatches=10000, patchSize=patchSize)

thetaParam = sparse_autoencoder.initializeParameters(hiddenSize, visibleSize)

options = {
		'maxiter': 400,
		'disp': True,
	}

result = optimize.minimize(sparseAutoencoderCostCallback, thetaParam, method='L-BFGS-B', jac=True, options=options)

W1 = result.x[0:hiddenSize*visibleSize].reshape(hiddenSize, visibleSize)

image_filename = 'images.png'
print 'Saving learned features to %s' % image_filename
visualizeNetwork(W1.T, image_filename)
