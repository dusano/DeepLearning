# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *
import scipy.io
from scipy import optimize

from compute_numerical_gradient import computeNumericalGradient
from sparse_autoencoder_cost import sparseAutoencoderCost
from visualize_network import visualizeNetwork


def getPatches(numPatches, patchSize):
	
	images = scipy.io.loadmat('IMAGES.mat')['IMAGES']
	
	patches = zeros((patchSize*patchSize, numPatches))
	
	numImages = images.shape[2]
	imageIdxs = random.randint(numImages, size=numPatches)
	sortedImageIdxs = argsort(imageIdxs)
	
	lastImageIdx = -1
	for i in range(numPatches):
		imageIdx = imageIdxs[sortedImageIdxs[i]]
		if lastImageIdx != imageIdx:
			img = images[:,:,imageIdx]
			lastImageIdx = imageIdx
			
		x = random.randint(img.shape[0] - patchSize)
		y = random.randint(img.shape[1] - patchSize)
	
		patch = img[x:x+patchSize, y:y+patchSize]
	
		patches[:, sortedImageIdxs[i]] = patch.reshape(1, patchSize*patchSize)
	
	# Remove DC (mean of images)
	patches = patches - mean(patches)
	
	# Truncate to +/-3 standard deviations and scale to -1 to 1
	pstd = 3 * std(patches)
	patches = maximum(minimum(patches, pstd), -pstd) / pstd
	
	# Rescale from [-1,1] to [0.1,0.9]
	patches = (patches + 1) * 0.4 + 0.1
	
	return patches


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


patchSize=8
visibleSize = patchSize*patchSize		# number of input units 
hiddenSize = 25							# number of hidden units 
sparsityParam = 0.01					# desired average activation of the hidden units.
lambdaParam = 0.0001					# weight decay parameter       
betaParam = 3							# weight of sparsity penalty term

patches = getPatches(numPatches=10000, patchSize=patchSize)

def sparseAutoencoderCostCallback(x):
	return sparseAutoencoderCost(x, visibleSize, hiddenSize, lambdaParam, sparsityParam,
				betaParam, patches) 

if False:
	# Obtain random parameters theta
	theta = initializeParameters(hiddenSize, visibleSize)
	
	(cost, grad) = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambdaParam,
						sparsityParam, betaParam, patches)
	
	numgrad = computeNumericalGradient(sparseAutoencoderCostCallback, theta);
	diff = linalg.norm(numgrad-grad)/linalg.norm(numgrad+grad)
	
	print('%s' % diff)
	print('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n')


theta = initializeParameters(hiddenSize, visibleSize)

options = {
		'maxiter': 400,
		'disp': True,
	}
	
result = optimize.minimize(sparseAutoencoderCostCallback, theta, method='L-BFGS-B', jac=True, options=options)

W1 = result.x[0:hiddenSize*visibleSize].reshape(hiddenSize, visibleSize)

visualizeNetwork(W1.T, 'images.png')
