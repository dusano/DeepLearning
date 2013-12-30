# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *
from scipy import optimize

from MNIST_images import loadMNISTImages, loadMNISTLabels
from softmax import softmaxCost, softmaxPredict
from compute_numerical_gradient import computeNumericalGradient


inputSize = 28 * 28		# Size of input vector (MNIST images are 28x28)
numClasses = 10			# Number of classes (MNIST images fall into 10 classes)
lambdaParam = 1e-4		# Weight decay parameter

images = loadMNISTImages('mnist/train-images-idx3-ubyte')
labels = loadMNISTLabels('mnist/train-labels-idx1-ubyte')

inputData = images

# For debugging purposes, you may wish to reduce the size of the input data
# in order to speed up gradient checking. 
# Here, we create synthetic dataset using random data for testing

# Set DEBUG to true when debugging.
DEBUG = False
if DEBUG:
	inputSize = 8
	inputData = random.normal(size=(100,inputSize))
	labels = random.randint(10, size=100)
	
# Randomly initialise theta
thetaParam = 0.005 * random.normal(size=numClasses * inputSize)

def softmaxCostCallback(x):
	return softmaxCost(x, numClasses, inputSize, lambdaParam, inputData, labels) 

(cost, grad) = softmaxCost(thetaParam, numClasses, inputSize, lambdaParam, inputData, labels)

if DEBUG:
	numGrad = computeNumericalGradient(softmaxCostCallback, thetaParam)
	
	# Compare numerically computed gradients with those computed analytically
	diff = linalg.norm(numGrad-grad)/linalg.norm(numGrad+grad)
	
	print('%s' % diff)
	print('Norm of the difference between numerical and analytical gradient (should be < 1e-7)\n\n')

# Randomly initialise theta
thetaParam = 0.005 * random.normal(size=numClasses * inputSize)

options = {
		'maxiter': 100,
		'disp': True,
	}

result = optimize.minimize(softmaxCostCallback, thetaParam, method='L-BFGS-B', jac=True, options=options)

optTheta = result.x[0:numClasses*inputSize].reshape(numClasses, inputSize)

images = loadMNISTImages('mnist/t10k-images-idx3-ubyte')
labels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte')

inputData = images

pred = softmaxPredict(optTheta, inputData)

acc = mean(labels==pred)
print('Accuracy: %0.3f%%\n' % (acc * 100))
