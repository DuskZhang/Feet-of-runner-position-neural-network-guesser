import numpy as np
import csv
import matplotlib.pyplot as plt
from normalize import tru
def train(guessY):
	num_hidden = 4
	num_inputs = 2
	num_outputs = 1
	momentum = 0.9
	learning_rate = 0.01
	#get inputs
	x = np.empty((num_inputs, 0), float)
	y = np.array([])
	
	with open("rundata.txt") as f:
		c = csv.reader(f, delimiter='\t')
		for line in c:
			x = np.append(x, [[float(line[0])] ,[float(line[1])]], axis = 1)
	
	x, mnsx, squsx = tru(x)
	
	if guessY:
		with open("rundataprey.txt") as f:
			c = csv.reader(f)
			for line in c:
				y = np.append(y, float(line[0]))
		
	else:
		with open("rundataprex.txt") as f:
			c = csv.reader(f)
			for line in c:
				y = np.append(y, float(line[0]))
	m = y.shape[0]
	y = np.reshape(y, (num_outputs,m))

	y, mnsy, squsy = tru(y)
	if guessY:
		np.savetxt("normalizedxy.txt", [mnsy, squsy] )
	else:
		np.savetxt("normalizedxx.txt", [mnsy, squsy] )

	temp = np.ravel(mnsx)
	temp = np.append(temp,np.ravel(squsx) )
	np.savetxt("normalizedx.txt", temp )
	num_iterations = 350000
	num_layers = 3
	#initialization

	weights1 = np.random.rand(num_hidden, num_inputs) * 0.12
	weights2 = np.random.rand(num_outputs, num_hidden) * 0.12
	b1 = np.zeros((num_hidden, 1))
	b2 = np.zeros((num_outputs, 1))
	parameters = {
				"W1" : weights1,#4, 2
				"W2" : weights2,#1, 4
				"b1" : b1,#4,1
				"b2" : b2 #1,1
	}
	vW1 = np.zeros(parameters["W1"].shape) 
	vW2 = np.zeros(parameters["W2"].shape) 
	vb1 = np.zeros(parameters["b1"].shape) 
	vb2 = np.zeros(parameters["b2"].shape) 
	#training
	for epoch in range(num_iterations):
		A_prev = x
		xz = [] #x of the second cache is input after first tanh
		for l in range(1, num_layers):
			#forward propagation
			weights = parameters["W" + str(l)]
			bias = parameters["b" + str(l)]
			if l == num_layers - 1: #output layer
				A_prev, xznew = linear_forward(A_prev, weights, bias) #regression
				xz = np.append(xz, xznew)
			else: #hidden layer
				A_prev, xznew = linear_activation_forward(A_prev, weights, bias, "tanh")
				xz = np.append(xz, xznew)
		#compute cost
		print("cost:",computeCost(A_prev, y))

		#backward propagation	
		dZ2 = A_prev - y # cost wrt z2, shape of 1, 66
		#print(xz[1]["x"]) #shape of 4, 66
		dW2 = dZ2.dot(xz[1]["x"].T) #dW2 shape of 1, 4 - tf 1, 66 * 66, 4 
		db2 = 1/m * np.sum(dZ2, axis = 1, keepdims=True) #shape 1,1
		#print("dta", dtanh(xz[0]['z']))
		dZ1 = parameters["W2"].T * dZ2 * dtanh(xz[0]['z']) #shape should be 4, 66 -  4, 1 * 1, 66 * 4,1 element wise 4, 66
		#print(dZ1.shape)
		dW1 = dZ1.dot(xz[0]['x'].T) / m
		#print('dw1',dW1)
		db1 = np.sum(dZ1, axis = 1, keepdims = True) / m
		#print('db1', db1)
		#updating weights
		vW1 = vW1 * momentum + (1- momentum) * dW1
		vW2 = vW2 * momentum + (1- momentum) * dW2 
		vb1 = vb1 * momentum + (1- momentum) * db1 
		vb2 = vb2 * momentum + (1- momentum) * db2  
		#print(parameters['W2'].shape)
		#print(dW2.shape)
		#print(vW2.shape)
		parameters['W1'] -= learning_rate * vW1
		parameters['W2'] -= learning_rate * vW2
		parameters['b1'] -= learning_rate * vb1
		parameters['b2'] -= learning_rate * vb2

	if(guessY == True):
		temp = np.ravel(parameters["W1"])
		temp = np.append(temp, np.ravel(parameters["W2"]), axis = 0)
		np.savetxt("weights_y_prediction.txt", temp )
		temp = np.ravel(parameters["b1"])
		temp = np.append(temp, np.ravel(parameters["b2"]), axis = 0)
		np.savetxt("biases_y_prediction.txt", temp)

	else:
		temp = np.ravel(parameters["W1"])
		temp = np.append(temp, np.ravel(parameters["W2"]), axis = 0)
		np.savetxt("weights_x_prediction.txt", temp )
		temp = np.ravel(parameters["b1"])
		temp = np.append(temp, np.ravel(parameters["b2"]), axis = 0)
		np.savetxt("biases_x_prediction.txt", temp)

	
def dtanh(z):
	return 1 - np.square(tanh(z))

def tanh(z):
	return 2 / (1 + np.exp(-2 * z)) - 1

def computeCost(X, Y):
	mse = np.square(X- Y)
	mse = np.mean(mse) * 0.5
	return mse

def linear_forward(x, weights, bias):
	x = x.astype(float)
	weights = weights.astype(float)
	bias = bias.astype(float)
	z = np.dot(weights,x)
	return z + bias, {'x' : x,'z': z}

def linear_activation_forward(x, weights, bias, activation):
	if activation == 'tanh':
		z, c = linear_forward(x, weights, bias)
		return tanh(z), c 

def predict(inputX, inputY, num_inputs, num_hidden, num_outputs):
	normals = np.array([])
	with open("normalizedx.txt") as f:
		c = csv.reader(f)
		for line in c:
			normals = np.append(normals, float(line[0]))
	mn1x = normals[0] 
	inputX -= mn1x
	mn2x = normals[1]
	inputY -= mn2x 
	sq1x = normals[2]
	inputX /= sq1x 
	sq2x = normals[3]
	inputY /= sq2x 
	normals = np.array([])
	with open("normalizedxx.txt") as f:
		c = csv.reader(f)
		for line in c:
			normals = np.append(normals, float(line[0]))

	mnx = normals[0]
	sqx = normals[1]

	normals = np.array([])
	with open("normalizedxy.txt") as f:
		c = csv.reader(f)
		for line in c:
			normals = np.append(normals, float(line[0]))

	mny = normals[0]
	sqy = normals[1]

	weights = np.array([])
	biases = np.array([])
	with open("weights_x_prediction.txt") as f:
		c = csv.reader(f)
		for line in c:
			weights = np.append(weights, float(line[0]))

	with open("biases_x_prediction.txt") as f:
		c = csv.reader(f)
		for line in c:
			biases = np.append(biases, float(line[0]))

	W1 = np.reshape(weights[0:num_hidden * num_inputs] ,(num_hidden, num_inputs))
	W2 = np.reshape(weights[num_hidden * num_inputs: ] ,(num_outputs, num_hidden))
	b1 = np.reshape(biases[0:num_hidden] ,(num_hidden, 1))
	b2 = np.reshape(biases[num_hidden: ] ,(num_outputs, 1))
	#first x needa be (num_inputs, 1)
	x = np.vstack((inputX, inputY))
	#print(x.shape)
	x, _ = linear_activation_forward(x, W1, b1, "tanh")
	x, _ = linear_forward(x, W2, b2)

	weights = np.array([])
	biases = np.array([])
	with open("weights_y_prediction.txt") as f:
		c = csv.reader(f)
		for line in c:
			weights = np.append(weights, float(line[0]))

	with open("biases_y_prediction.txt") as f:
		c = csv.reader(f)
		for line in c:
			biases = np.append(biases, float(line[0]))

	W1 = np.reshape(weights[0:num_hidden * num_inputs] ,(num_hidden, num_inputs))
	W2 = np.reshape(weights[num_hidden * num_inputs: ] ,(num_outputs, num_hidden))
	b1 = np.reshape(biases[0:num_hidden] ,(num_hidden, 1))
	b2 = np.reshape(biases[num_hidden: ] ,(num_outputs, 1))

	y = np.vstack((inputX, inputY))
	y, _ = linear_activation_forward(y, W1, b1, "tanh")
	y, _ = linear_forward(y, W2, b2)
	
	return (x * sqx + mnx), (y * sqy + mny)

desireTraining = False
if desireTraining:
	trainingY = True
	train(trainingY)
	trainingY = False
	train(trainingY)

def main():
	stopguessing = False
	xs = []
	ys = []
	xo = []
	yo = []
	while(stopguessing == False):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.axis([0, 1290, 0, 2160])
		x = np.empty((2, 0), float)
		with open("rundata.txt") as f:
			c = csv.reader(f, delimiter='\t')
			for line in c:
				x = np.append(x, [[float(line[0])] ,[float(line[1])]], axis = 1)
		for i in range(x.shape[1]):
			x_in = x[0,i]
			y_in = x[1,i]
			xs.append(x_in)
			ys.append(y_in)
			x_out, y_out = predict(x_in, y_in, 2, 4, 1)
			xo.append(float(x_out))
			yo.append(float(y_out))
		plt.xlabel("X-coord of foot")
		plt.ylabel("Y-coord of foot")
		plt.legend(['g^', 'bs'], ["Left Foot", "Right Foot"])
		plt.plot(xs, ys, 'g^', xo, yo, 'bs')
		i = 0
		for xy in zip(xs, ys):
			i+=1
			if(i % 5 == 0):
				ax.annotate('(%s)' % i, xy = xy, textcoords ='data', color='red')
		i = 0
		for xy in zip(xo, yo):
			i+=1
			if(i % 5 == 0):
				ax.annotate('(%s)' % i, xy = xy, textcoords ='data', color='brown')
		plt.grid()
		plt.show()
		cmd = input("End Program? Y/ N\n")
		if cmd == 'Y':
			stopguessing = True

main()