from __future__ import print_function, division
from builtins import range
import gym
import tensorflow as tf
import numpy as np


class HiddenLayer(object):
	def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
		self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
		self.use_bias = use_bias
		self.f = f
		if use_bias:
			self.b = tf.Variable(np.zeros(M2).astype(np.float32))

	def forward(self, X):
		if self.use_bias:
			a = tf.matmul(X, self.W) + self.b
		else:
			a = tf.matmul(X, self.W)
		return self.f(a)


class PolicyModel(object):
	def __init__(self, D, K, hidden_layer_sizes):
		self.layers = []
		M1 = D
		for M2 in hidden_layer_sizes:
			self.layers.append(HiddenLayer(M1, M2))
			M1 = M2

		self.layers.append(HiddenLayer(M1, K, tf.nn.softmax, 
									   use_bias=False))

		self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
		self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
		self.advantages = tf.placeholder(tf.float32, shape=(None,), 
										 name='advantages')

		Z = self.X
		for layer in self.layers:
			Z = layer.forward(Z)
		p_a_given_s = Z
		self.predict_op	= p_a_given_s

		selected_log_probs = tf.log(
			tf.reduce_sum(
				p_a_given_s * tf.one_hot(self.actions, K),
				reduction_indices=[1]
			)
		)

		cost = -tf.reduce_sum(self.advantages * selected_log_probs)
		self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)

	def set_session(self, session):
		self.session = session

	def predict(self, X):
		X = np.atleast_2d(X)
		return self.session.run(self.predict_op, feed_dict={self.X: X})

	def sample_action(self, X):
		p = self.predict(X)[0]
		return np.random.choice(len(p), p=p)

	def partial_fit(self, X, actions, advantages):
		X = np.atleast_2d(X)
		actions = np.atleast_1d(actions)
		advantages = np.atleast_1d(advantages)
		self.session.run(
			self.train_op,
			feed_dict={
				self.X: X,
				self.advantages: advantages,
				self.actions: actions
			}
		)


class ValueModel(object):
	def __init__(self, D, hidden_layer_sizes):
		self.layers = []
		M1 = D
		for M2 in hidden_layer_sizes:
			self.layers.append(HiddenLayer(M1, M2))
			M1 = M2
		self.layers.append(HiddenLayer(M1, 1, lambda x: x))

		self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
		self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

		Z = self.X
		for layer in self.layers:
			Z = layer.forward(Z)
		Y_hat = tf.reshape(Z, [-1])
		self.predict_op = Y_hat

		cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
		self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

	def set_session(self, session):
		self.session = session

	def predict(self, X):
		X = np.atleast_2d(X)
		return self.session.run(self.predict_op, feed_dict={self.X: X})

	def partial_fit(self, X, Y):
		X = np.atleast_2d(X)
		Y = np.atleast_1d(Y)
		self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})


def play_game_td(env, pmodel, vmodel, gamma):
	s = env.reset()
	done = False
	while not done:
		a = pmodel.sample_action(s)
		prev_s = s
		s, r, done, _ = env.step(a)
		v = vmodel.predict(s)[0]
		G = r + gamma * v
		advantage = G - vmodel.predict(prev_s)
		pmodel.partial_fit(prev_s, a, advantage)
		vmodel.partial_fit(prev_s, G)

def play_game_mc(env, pmodel, vmodel, gamma):
	states = []
	actions = []
	rewards = []
	s = env.reset()
	done = False
	r = 0
	total_rewards = 0.
	i = 0
	while not done:
		i += 1
		a = pmodel.sample_action(s)
		states.append(s)
		actions.append(a)
		rewards.append(r)
		prev_s = s
		s, r, done, _ = env.step(a)
		if done and i < 200:
			r = -200
		total_rewards += r if r != -200 else 0
	a = pmodel.sample_action(s)
	states.append(s)
	actions.append(a)
	rewards.append(r)

	returns = []
	advantages = []
	G = 0
	for s, r in zip(reversed(states), reversed(rewards)):
		returns.append(G)
		advantages.append(G - vmodel.predict(s)[0])
		G = r + gamma*G
	returns.reverse()
	advantages.reverse()

	pmodel.partial_fit(states, actions, advantages)
	vmodel.partial_fit(states, returns)

	return total_rewards


if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	D = env.observation_space.shape[0]
	K = env.action_space.n
	pmodel = PolicyModel(D, K, [])
	vmodel = ValueModel(D, [10])
	init = tf.global_variables_initializer()
	session = tf.InteractiveSession()
	session.run(init)
	pmodel.set_session(session)
	vmodel.set_session(session)
	gamma = 0.99

	N = 1000
	total_rewards = []
	for n in range(N):
		total_rewards.append(play_game_mc(env, pmodel, vmodel, gamma))
		if (n + 1) % 50 == 0:
			print(n + 1)
			print(np.mean(total_rewards))
			total_rewards = []
