import numpy as np
import gym
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

class FeatureTransformer(object):
	def __init__(self, env, num_exemplars=500):
		observation_examples = np.array([env.observation_space.sample() 
										 for x in range(500)])
		scaler = StandardScaler()
		scaler.fit(observation_examples)

		featurizer = FeatureUnion([
				('rbf1', RBFSampler(gamma=5.0, n_components=num_exemplars)),
				('rbf2', RBFSampler(gamma=2.0, n_components=num_exemplars)),
				('rbf3', RBFSampler(gamma=1.0, n_components=num_exemplars)),
				('rbf4', RBFSampler(gamma=0.5, n_components=num_exemplars)),
			])
		featurizer.fit(scaler.transform(observation_examples))

		self.scaler = scaler
		self.featurizer = featurizer

	def transform(self, s):
		return self.featurizer.transform(self.scaler.transform(s))


class Model(object):
	def __init__(self, env, feature_transformer, learning_rate):
		self.env = env
		self.models = []
		self.feature_transformer = feature_transformer
		for i in range(env.action_space.n):
			model = SGDRegressor(learning_rate=learning_rate)
			model.partial_fit(feature_transformer.transform(
								[env.reset()]), [0])
			self.models.append(model)

	def predict(self, s):
		s = self.feature_transformer.transform([s])
		return np.array([m.predict(s)[0] for m in self.models])

	def sample_action(self, s, eps):
		if np.random.random() <= eps:
			return env.action_space.sample()
		else:
			return np.argmax(self.predict(s))

	def update(self, s, a, G):
		s = self.feature_transformer.transform([s])
		self.models[a].partial_fit(s, [G])


def play_game(env, model, eps, gamma):
	s = env.reset()
	done = False
	total_reward = 0.
	while not done:
		a = model.sample_action(s, eps)
		s_prev = s
		s, r, done, _ = env.step(a)
		G = r + gamma * np.max(model.predict(s))
		model.update(s_prev, a, G)
		total_reward += r
	return total_reward


if __name__ == '__main__':
	env = gym.make('MountainCar-v0')
	feature_transformer = FeatureTransformer(env)
	model = Model(env, feature_transformer, "constant")
	gamma = 0.99

	total_rewards = []
	for n in range(300):
		eps = 0.1 * (0.97 ** n)
		total_rewards.append(play_game(env, model, eps, gamma))
		if (n + 1) % 10 == 0:
			print(n + 1)
			print(np.mean(total_rewards))
			total_rewards = []

	total_rewards = []
	for n in range(100):
		total_rewards.append(play_game(env, model, 0, gamma))
	print(np.mean(total_rewards))