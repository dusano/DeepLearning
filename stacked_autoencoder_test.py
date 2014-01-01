# Based on CS294A/CS294W Programming Assignment Starter Code
import os

from numpy  import *
from scipy import optimize

from MNIST_images import loadMNISTImages, loadMNISTLabels
from visualize_network import visualizeNetwork
import sparse_autoencoder
import softmax
import stacked_autoencoder


results_dir = 'results/sae2/'

inputSize = 28 * 28		# MNIST inputs are 28x28
numClasses = 10			# MNIST dataset consists of 10 digits
hiddenSizeL1 = 200		# Layer 1 Hidden Size
hiddenSizeL2 = 200		# Layer 2 Hidden Size
sparsityParam = 0.1		# desired average activation of the hidden units.
lambdaParam = 3e-3		# weight decay parameter       
betaParam = 3			# weight of sparsity penalty term       
corruptionLevel = 0.3	# how much of the input to denoising autoencoder will get corrupted.

if not os.path.exists(results_dir):
	os.makedirs(results_dir)

trainData = loadMNISTImages('mnist/train-images-idx3-ubyte')
trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte')

# Train the first sparse autoencoder
options = {
		'maxiter': 400,
		'disp': True,
	}

sae1OptThetaFilename = results_dir + 'sae1OptTheta.npy'

if os.path.exists(sae1OptThetaFilename):
	sae1OptTheta = load(sae1OptThetaFilename)
else:
	def sparseAutoencoderCostCallbackL1(x):
		return sparse_autoencoder.cost(x, inputSize, hiddenSizeL1, lambdaParam, sparsityParam,
					betaParam, trainData)
	
	sae1Theta = sparse_autoencoder.initializeParameters(hiddenSizeL1, inputSize)
	result = optimize.minimize(sparseAutoencoderCostCallbackL1, sae1Theta, method='L-BFGS-B', jac=True, options=options)
	
	sae1OptTheta = result.x
	save(sae1OptThetaFilename, sae1OptTheta)
	
	W1 = sae1OptTheta[0:hiddenSizeL1*inputSize].reshape(hiddenSizeL1, inputSize)
	visualizeNetwork(W1.T, results_dir + 'sae1.png')

# Train the second sparse autoencoder
sae1Features = sparse_autoencoder.feedForward(sae1OptTheta, hiddenSizeL1, inputSize, trainData)

sae2OptThetaFilename = results_dir + 'sae2OptTheta.npy'

if os.path.exists(sae2OptThetaFilename):
	sae2OptTheta = load(sae2OptThetaFilename)
else:
	def sparseAutoencoderCostCallbackL2(x):
		return sparse_autoencoder.cost(x, hiddenSizeL1, hiddenSizeL2, lambdaParam, sparsityParam,
					betaParam, sae1Features)
	
	sae2Theta = sparse_autoencoder.initializeParameters(hiddenSizeL2, hiddenSizeL1)
	result = optimize.minimize(sparseAutoencoderCostCallbackL2, sae2Theta, method='L-BFGS-B', jac=True, options=options)
	
	sae2OptTheta = result.x
	save(sae2OptThetaFilename, sae2OptTheta)

# Train the softmax classifier
saeSoftmaxOptThetaFilename = results_dir + 'saeSoftmaxOptTheta.npy'

if os.path.exists(saeSoftmaxOptThetaFilename):
	saeSoftmaxOptTheta = load(saeSoftmaxOptThetaFilename)
else:
	sae2Features = sparse_autoencoder.feedForward(sae2OptTheta, hiddenSizeL2, hiddenSizeL1, sae1Features)
	
	softmax_lambda = 1e-4
	
	def softmaxCostCallback(x):
		return softmax.cost(x, numClasses, hiddenSizeL2, softmax_lambda, sae2Features, trainLabels) 
	
	# Randomly initialise theta
	thetaParam = 0.005 * random.normal(size=numClasses * hiddenSizeL2)
	
	softmax_options = {
			'maxiter': 100,
			'disp': False,
		}
	
	result = optimize.minimize(softmaxCostCallback, thetaParam, method='L-BFGS-B', jac=True, options=softmax_options)
	
	saeSoftmaxOptTheta = result.x[0:numClasses*hiddenSizeL2]
	
	save(saeSoftmaxOptThetaFilename, saeSoftmaxOptTheta)

# Finetune softmax model

stack = [stacked_autoencoder.Layer(1), stacked_autoencoder.Layer(2)]
stack[0].W = sae1OptTheta[0:hiddenSizeL1*inputSize].reshape(hiddenSizeL1, inputSize)
stack[0].b = sae1OptTheta[2*hiddenSizeL1*inputSize:2*hiddenSizeL1*inputSize+hiddenSizeL1]
stack[1].W = sae2OptTheta[0:hiddenSizeL2*hiddenSizeL1].reshape(hiddenSizeL2, hiddenSizeL1)
stack[1].b = sae2OptTheta[2*hiddenSizeL2*hiddenSizeL1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2]

(stackParams, netConfig) = stacked_autoencoder.stack2params(stack)
stackedAETheta = concatenate([saeSoftmaxOptTheta, stackParams])

saeOptThetaFilename = results_dir + 'saeOptTheta.npy'

if os.path.exists(saeOptThetaFilename):
	stackedAEOptTheta = load(saeOptThetaFilename)
else:
	def stackedAutoencoderCostCallback(x):
		return stacked_autoencoder.cost(x, inputSize, hiddenSizeL2, numClasses, netConfig,
				lambdaParam, trainData, trainLabels)

	result = optimize.minimize(stackedAutoencoderCostCallback, stackedAETheta, method='L-BFGS-B', jac=True, options=options)
	
	stackedAEOptTheta = result.x
	save(saeOptThetaFilename, stackedAEOptTheta)

# Test

testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte')
testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte')

pred_no_fine_tuning = stacked_autoencoder.predict(stackedAETheta, inputSize, hiddenSizeL2, numClasses, netConfig, testData)

acc_no_fine_tuning = mean(testLabels==pred_no_fine_tuning)
print('Before Fine-tuning Test Accuracy: %0.3f%%\n' % (acc_no_fine_tuning * 100))

pred_fine_tuned = stacked_autoencoder.predict(stackedAEOptTheta, inputSize, hiddenSizeL2, numClasses, netConfig, testData)

acc_fine_tuned = mean(testLabels==pred_fine_tuned)
print('Adter Fine-tuning Test Accuracy: %0.3f%%\n' % (acc_fine_tuned * 100))