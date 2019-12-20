import numpy as np
import gym
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


class BaseModel(object):
	def __init__(self, D):
		self.w = np.random.randn(D) / np.sqrt(D)

	def partial_fit(self, input_, target, eligibility, lr=10**-2):
		#print(np.mean(lr * (target - input_.dot(self.w)) * eligibility))
		self.w += lr * (target - input_.dot(self.w)) * eligibility

	def predict(self, X):
		X = np.array(X)
		return X.dot(self.w)


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
		examples = featurizer.fit_transform(scaler.transform(observation_examples))

		self.scaler = scaler
		self.featurizer = featurizer
		self.dimensions = examples.shape[1]

	def transform(self, s):
		return self.featurizer.transform(self.scaler.transform(s))


class Model(object):
	def __init__(self, env, feature_transformer):
		self.env = env
		self.models = []
		self.feature_transformer = feature_transformer
		D = feature_transformer.dimensions
		self.eligibilities = np.zeros((env.action_space.n, D))
		for i in range(env.action_space.n):
			model = BaseModel(D)
			self.models.append(model)

	def predict(self, s):
		s = self.feature_transformer.transform([s])
		return np.array([m.predict(s)[0] for m in self.models])

	def sample_action(self, s, eps):
		if np.random.random() <= eps:
			return env.action_space.sample()
		else:
			return np.argmax(self.predict(s))

	def update(self, s, a, G, gamma, lambda_):
		s = self.feature_transformer.transform([s])
		self.eligibilities *= gamma * lambda_
		self.eligibilities[a] += s[0]
		self.models[a].partial_fit(s[0], G, self.eligibilities[a])


def play_game(env, model, eps, gamma, lambda_):
	s = env.reset()
	done = False
	total_reward = 0.
	while not done:
		a = model.sample_action(s, eps)
		s_prev = s
		s, r, done, _ = env.step(a)
		G = r + gamma * np.max(model.predict(s)[0])
		model.update(s_prev, a, G, gamma, lambda_)
		total_reward += r
	return total_reward


if __name__ == '__main__':
	env = gym.make('MountainCar-v0')
	feature_transformer = FeatureTransformer(env)
	model = Model(env, feature_transformer)
	gamma = 0.99
	lambda_ = 0.1

	total_rewards = []
	for n in range(300):
		eps = 0.1 * (0.97 ** n)
		total_rewards.append(play_game(env, model, eps, gamma, lambda_))
		if (n + 1) % 10 == 0:
			print(n + 1)
			print(np.mean(total_rewards))
			total_rewards = []

	total_rewards = []
	for n in range(100):
		total_rewards.append(play_game(env, model, 0, gamma, lambda_))
	print(np.mean(total_rewards))