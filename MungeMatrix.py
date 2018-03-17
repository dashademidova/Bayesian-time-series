import numpy as np

def MungeMatrix(v):
	vScale = np.sqrt(np.abs(np.diag(v)))
	vScale = np.outer(vScale, vScale)
	vScale[vScale == 0] = 1e-8
	standardizedV = v / vScale
	m = np.min(np.linalg.eigvals(standardizedV))
	if m < 0:
		standardizedV -= 2 * m * np.eye(v.shape[0])
		standardizedV /= 1 - 2 * m 
		v = standardizedV * vScale
	return v
