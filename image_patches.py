# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *
import scipy.io


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