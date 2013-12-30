# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *
import scipy.misc

def visualizeNetwork(A, filename):
	"""This function visualizes filters in matrix A"""

	# rescale
	A = A - mean(A)
 
	# compute rows, cols
	L = A.shape[0]
	M = A.shape[1]
	sz = int(sqrt(L))
	buf = 1
	if floor(sqrt(M))**2 != M:
		n = ceil(sqrt(M))
		while ((M % n) !=0) and (n < 1.2*sqrt(M)):
			n += 1
		m = ceil(M/n)
	else:
		n=sqrt(M);
		m=n;

	array = -ones((buf+m*(sz+buf), buf+n*(sz+buf))) 
	k = 0
	for i in range(int(m)):
		for j in range(int(n)):
			if k >= M: 
				continue 
			
			clim = max(abs(A[:,k]))
			array[buf+i*(sz+buf):buf+i*(sz+buf)+sz, buf+j*(sz+buf):buf+j*(sz+buf)+sz] = A[:,k].reshape(sz,sz)/clim
			k += 1
	
	scipy.misc.toimage(array).save(filename)

	return
