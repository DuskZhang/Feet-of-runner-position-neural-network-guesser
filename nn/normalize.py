import numpy as np

def normalize(data):
	for i in range(data.shape[1]):
		mx = max(data[:,i])
		mi = min(data[:,i])
		data[:,i] = (data[:,i] - mi) / (mx - mi)
	return data 

def normalizemn(data):
	for i in range(data.shape[1]):
		mn = np.mean(data[:,i])
		data[:,i] -= mn
		csq = np.mean(np.square(data[i:,i]))
		data[:,i] /= np.sqrt(csq)
	return data 

def tru(data):
	m = data.shape[1]
	n = data.shape[0]
	mns = []
	squs = []
	for i in range(n):
		mn = sum(data[i,:]) / m
		data[i,:] -= mn
		mns = np.append(mns, mn)
		#get positive sqrt of max
		mx = np.max(abs(data[i,:]))
		squ = np.sqrt(mx ** 2)
		squs = np.append(squs, squ)
		data[i,:] /= squ
	return data, mns, squs