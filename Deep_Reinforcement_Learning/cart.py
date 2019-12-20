import gym
import numpy as np


class FeatureTransformer(object):
	def __init__(self):
		self.cart_pos_bins = np.linspace(-2.4, 2.4, 9)
		self.cart_vel_bins = np.linspace(-2, 2, 9)
		self.pole_ang_bins = np.linspace(-0.4, 0.4, 9)
		self.pole_vel_bins = np.linspace(-3.5, 3.5, 9)

	def transform(self, s):
		cart_pos, cart_vel, pole_ang, pole_vel = s
		cart_pos = np.digitize(cart_pos, self.cart_pos_bins)
		cart_vel = np.digitize(cart_vel, self.cart_vel_bins)
		pole_ang = np.digitize(pole_ang, self.pole_ang_bins)
		pole_vel = np.digitize(pole_vel, self.pole_vel_bins)
		return cart_pos * 1 + cart_vel * 10 + pole_ang * 100 + pole_vel * 1000


class Model(object):
	def __init__(self, env):
		self.env = env
		self.feature_transformer = FeatureTransformer()
		num_states = 10 ** 4
		num_actions = env.action_space.n
		self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

	def predict(self, s):
		s = self.feature_transformer.transform(s)
		return self.Q[s]

	def update(self, s, a, G, alpha=10**-3):
		s = self.feature_transformer.transform(s)
		self.Q[s, a] += alpha * (G - self.Q[s, a])

	def sample_action(self, s, eps):
		if np.random.random() <= eps:
			return self.env.action_space.sample()
		else:
			p = self.predict(s)
			return np.argmax(p)


def play_game(env, model, eps, gamma):
	s = env.reset()
	done = False
	i = 0
	total_reward = 0.
	while not done:
		a = model.sample_action(s, eps)
		s_prev = s
		s, r, done, _ = env.step(a)
		if done and i < 199:
			r = -300
		total_reward += r
		G = r + gamma * np.max(model.predict(s))
		model.update(s_prev, a, G)
		i += 1
	return total_reward, i

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	model = Model(env)
	gamma = 0.9
	total_rewards = []
	iters = []
	for i in range(10000):
		eps = 1/np.sqrt(i+1)
		r, n = play_game(env, model, eps, gamma)
		total_rewards.append(r)
		iters.append(n)
		if i % 100 == 0:
			print(i)
			print(np.sum(total_rewards)/len(total_rewards))
			print(np.max(iters))
			total_rewards = []
			iters = []

	total_rewards = []
	iters = []
	for i in range(100):
		r, n = play_game(env, model, 0., gamma)
		total_rewards.append(r)
		iters.append(n)
	print(np.sum(total_rewards)/len(total_rewards))
	print(np.max(iters))
	print(np.mean(iters))
