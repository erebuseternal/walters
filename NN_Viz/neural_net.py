import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

class tanh(object):
	def forward(self, Z):
		return np.tanh(Z)

	def backward(self, A):
		return 1 - np.power(A, 2)

	def predict(self, A):
		return np.where(A > 0, 1., -1.)

class sigmoid(object):
	def forward(self, Z):
		return 1/(1 + np.exp(-Z)) 

	def backward(self, A):
		return A * (1 - A)

	def predict(self, A):
		return np.where(A > 0.5, 1., 0.)

def initiate_layers(layer_nodes, X):
	m = X.shape[0]
	layers = []
	for n, g in layer_nodes:
		W = np.random.randn(n, m) * 0.01
		B = np.zeros((n, 1))
		layers.append((W, B, g))
		m = n
	return layers

def forward_propogation(X, layers):
	activation = X
	activations = [X]
	for W, B, g in layers:
		Z = np.dot(W, activation) + B
		activation = g.forward(Z)
		activations.append(activation)
	return activations

def backward_propogation(Y, learning_rate, activations, layers):
	current_A = activations[-1]
	prev_activations = activations[:-1]
	dA = -np.divide(Y, current_A) + np.divide(1 - Y, 1 - current_A)
	new_layers = []
	for (W, B, g), previous_A in reversed(zip(layers, prev_activations)):
		m = previous_A.shape[1]
		dZ = dA * g.backward(current_A)
		dW = np.dot(dZ, previous_A.T) / m
		dB = np.sum(dZ, axis=1, keepdims=True) / m
		dA = np.dot(W.T, dZ)
		W = W - learning_rate * dW
		B = B - learning_rate * dB
		new_layers.append((W, B, g))
		current_A = previous_A
	return list(reversed(new_layers))

def loss(Y, A):
	m = Y.shape[1]
	return -(np.dot(Y, np.log(A).T)+np.dot((1-Y), np.log(1-A).T))/m

def plot_decision_boundary(layers, X, Y, cmap='Paired_r', layer=-1, node=0):
    h = 0.02
    x_min, x_max = X[0,:].min() - 10*h, X[0,:].max() + 10*h
    y_min, y_max = X[1,:].min() - 10*h, X[1,:].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    D = np.c_[xx.ravel(), yy.ravel()].T
    _, _, g = layers[layer]
    Z = g.predict(forward_propogation(D, layers)[layer+1][node])
    #Z = np.where(forward_propogation(D, layers)[layer][node] > 0.5, 1., 0.)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.title('Layer %s, Node %s' % (layer, node))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[0], X[1], c=Y[0], cmap=cmap, edgecolors='k')
    plt.savefig('decision_boundary_%s_%s.png' % (layer, node))

if __name__ == '__main__':
	X1 = np.random.randn(2, 40)
	X1 /= 1.5 * np.linalg.norm(X1, axis=0)
	Y1 = np.array([1.] * 40).reshape((1, 40))
	X2 = np.random.randn(2, 40)
	X2 /= np.linalg.norm(X2, axis=0)
	Y2 = np.array([0.] * 40).reshape((1, 40))
	X = np.concatenate([X1, X2], axis=1)
	Y = np.concatenate([Y1, Y2], axis=1)
	layers = initiate_layers([(3, tanh()), (1, sigmoid())], X)
	num_iterations = 10000
	losses = []
	for i in range(num_iterations):
		activations = forward_propogation(X, layers)
		layers = backward_propogation(Y, 1.0, activations, layers)
		P = np.where(activations[-1] > 0.5, 1., 0)
		losses.append(np.squeeze(loss(Y, activations[-1])))
	print(metrics.accuracy_score(Y[0], P[0]))
	for i, (W, B, g) in enumerate(layers):
		for j in range(W.shape[0]):
			plot_decision_boundary(layers, X, Y, layer=i, node=j)
	plt.figure(figsize=(5,5))
	plt.plot(losses)
	plt.savefig('loss.png')





