# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *
from scipy import optimize

from MNIST_images import loadMNISTImages, loadMNISTLabels
import softmax


inputSize = 28 * 28		# Size of input vector (MNIST images are 28x28)
numClasses = 10			# Number of classes (MNIST images fall into 10 classes)
lambdaParam = 1e-4		# Weight decay parameter

trainData = loadMNISTImages('mnist/train-images-idx3-ubyte')
trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte')

def softmaxCostCallback(x):
	return softmax.cost(x, numClasses, inputSize, lambdaParam, trainData, trainLabels) 

# Randomly initialise theta
thetaParam = 0.005 * random.normal(size=numClasses * inputSize)

options = {
		'maxiter': 100,
		'disp': True,
	}

result = optimize.minimize(softmaxCostCallback, thetaParam, method='L-BFGS-B', jac=True, options=options)

optTheta = result.x[0:numClasses*inputSize].reshape(numClasses, inputSize)

# Evaluating performance of the softmax classifier
testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte')
testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte')

pred = softmax.predict(optTheta, testData)

acc = mean(testLabels==pred)
print('Accuracy: %0.3f%%\n' % (acc * 100))
