# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *
import scipy.io
from scipy import optimize

from compute_numerical_gradient import computeNumericalGradient
import sparse_autoencoder
from visualize_network import visualizeNetwork


def getPatches(numPatches, patchSize):
	
	images = scipy.io.loadmat('IMAGES.mat')['IMAGES']
	
	patches = zeros((numPatches, patchSize*patchSize))
	
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
	
		patches[sortedImageIdxs[i], :] = patch.reshape(1, patchSize*patchSize)
	
	# Remove DC (mean of images)
	patches = patches - mean(patches)
	
	# Truncate to +/-3 standard deviations and scale to -1 to 1
	pstd = 3 * std(patches)
	patches = maximum(minimum(patches, pstd), -pstd) / pstd
	
	# Rescale from [-1,1] to [0.1,0.9]
	patches = (patches + 1) * 0.4 + 0.1
	
	return patches


patchSize=8
visibleSize = patchSize*patchSize		# number of input units 
hiddenSize = 25							# number of hidden units 
sparsityParam = 0.01					# desired average activation of the hidden units.
lambdaParam = 0.0001					# weight decay parameter       
betaParam = 3							# weight of sparsity penalty term

def sparseAutoencoderCostCallback(x):
	return sparse_autoencoder.cost(x, visibleSize, hiddenSize, lambdaParam, sparsityParam,
				betaParam, patches) 

patches = getPatches(numPatches=10, patchSize=patchSize)

# Obtain random parameters theta
thetaParam = sparse_autoencoder.initializeParameters(hiddenSize, visibleSize)

(cost, grad) = sparse_autoencoder.cost(thetaParam, visibleSize, hiddenSize, lambdaParam,
					sparsityParam, betaParam, patches)

numgrad = computeNumericalGradient(sparseAutoencoderCostCallback, thetaParam);
diff = linalg.norm(numgrad-grad)/linalg.norm(numgrad+grad)

print('%s' % diff)
print('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n')

print('Minimizing costs...')
patches = getPatches(numPatches=10000, patchSize=patchSize)

thetaParam = sparse_autoencoder.initializeParameters(hiddenSize, visibleSize)

options = {
		'maxiter': 400,
		'disp': False,
	}

result = optimize.minimize(sparseAutoencoderCostCallback, thetaParam, method='L-BFGS-B', jac=True, options=options)

W1 = result.x[0:hiddenSize*visibleSize].reshape(hiddenSize, visibleSize)

visualizeNetwork(W1.T, 'images.png')
