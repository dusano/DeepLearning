# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *


def loadMNISTImages(filename):
	"""loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
	the raw MNIST images.
	"""
	
	f = open(filename, 'rb')

	assert f != -1, 'Could not open %s' % filename

	magic = fromfile(f, dtype='>i4', count=1)
	assert magic == 2051, 'Bad magic number in %s' % filename

	numImages = fromfile(f, dtype='>i4', count=1)
	numRows = fromfile(f, dtype='>i4', count=1)
	numCols = fromfile(f, dtype='>i4', count=1)

	images = fromfile(f, dtype='B')
	images = images.reshape(numImages, numCols, numRows)
	
	f.close()

	# Reshape to #pixels x #examples
	images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
	# Convert to double and rescale to [0,1]
	images = double(images) / 255

	return images
	
	
def loadMNISTLabels(filename):
	"""loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
	the labels for the MNIST images
	"""

	f = open(filename, 'rb')
	assert f != -1, 'Could not open %s' % filename

	magic = fromfile(f, dtype='>i4', count=1)
	assert magic == 2049, 'Bad magic number in %s' % filename
	
	numLabels = fromfile(f, dtype='>i4', count=1)
	
	labels = fromfile(f, dtype='B')
	
	assert labels.shape[0] == numLabels, 'Mismatch in label count'
	
	f.close()
	
	return labels
