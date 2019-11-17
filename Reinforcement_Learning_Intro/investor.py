import random
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

def load_data():
	df = pd.read_csv('aapl_msi_sbux.csv')
	return np.concatenate(([df['AAPL']], [df['MSI']], [df['SBUX']]))

def get_scaler(env, number_of_stocks):
  states = []
  done = False
  investor = Investor(number_of_stocks, 1, 1, None)
  while not done:
    action = investor.Q.random_action()
    state, reward, done = env.take_action(action)
    states.append(state)

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler

class Portfolio(object):
	def __init__(self, stock_prices, initial_investments, cash):
		self.stock_prices = stock_prices
		self.time = 0
		self.cash = cash
		self.investments = np.array(initial_investments)

	def take_action(self, action):
		starting_value = self.portfolio_value()
		new_investments = np.zeros(self.investments.shape[0])
		min_buy_price = float('inf')
		buys = []
		for i, choice in enumerate(action):
			if choice == 'H':
				new_investments[i] = self.investments[i]
				continue
			price = self.stock_prices[i, self.time]
			number = self.investments[i]
			if choice == 'B':
				buys.append((i, price))
				if min_buy_price > price:
					min_buy_price = price
			self.cash += price * number
		while self.cash >= min_buy_price:
			i, price = random.choice(buys)
			if price > self.cash:
				continue
			new_investments[i] += 1
			self.cash -= price
		self.time += 1
		self.investments = new_investments
		reward = self.portfolio_value() - starting_value
		done = self.time + 1 == self.stock_prices.shape[1]
		return self.get_state(), reward, done

	def get_state(self):
		return np.array(
				[self.cash]
				+ list(self.investments)
				+ list(self.stock_prices[:, self.time])
			)

	def portfolio_value(self):
		return (self.cash
				+ np.dot(
					self.investments,
					self.stock_prices[:, self.time]
					)
				)

class QApproximator(object):
	def __init__(self, number_of_stocks, gamma, alpha, scalar):
		self.gamma = gamma
		self.alpha = alpha
		self.scalar = scalar
		self.index_to_action = {}
		self._create_index_to_action_map(number_of_stocks)
		self.params = np.zeros((3 ** number_of_stocks, 2 * number_of_stocks + 2))

	def _action_to_index(self, a):
		c2i = {
			'H': 0,
			'B': 1,
			'S': 2
		}
		index = 0
		for i, c in enumerate(a):
			index += c2i[c] * (3 ** i)
		return index

	def _create_index_to_action_map(self, number_of_stocks, current_action=[]):
		if len(current_action) == number_of_stocks:
			self.index_to_action[self._action_to_index(current_action)] = tuple(current_action)
		else:
			for c in ['H', 'B', 'S']:
				self._create_index_to_action_map(number_of_stocks, current_action + [c])

	def update(self, s1, a1, r, s2, a2):
		s1 = self._convert_state(s1)
		s2 = self._convert_state(s2)
		target = r + self.gamma * np.max(np.dot(self.params, s2))
		t_hat = np.dot(self.params, s1)[self._action_to_index(a1)]
		self.params[self._action_to_index(a1),:] -= self.alpha*(t_hat - target) * s1

	def update_terminal(self, s, a, r):
		s = self._convert_state(s)
		target = r
		t_hat = np.dot(self.params, s)[self._action_to_index(a)]
		self.params[self._action_to_index(a),:] -= self.alpha*(t_hat - target) * s

		"""
		s = self._convert_state(s)
		actual = np.dot(self.params, s)
		adjustment = np.dot(self.params, s)
		adjustment[self._action_to_index(a)] = (r)
		coef = np.reshape((self.alpha * (adjustment - actual)), (self.params.shape[0], 1))
		dp = np.zeros((27, 8)) + s
		self.params += coef * dp
		"""

	def _convert_state(self, s):
		return np.array(list(scalar.transform([s])[0]) + [1])

	def maximal_action(self, s):
		s = self._convert_state(s)
		inference = np.dot(self.params, s)
		index = np.argmax(inference)
		return self.index_to_action[index]

	def random_action(self):
		index = random.randint(0, len(self.index_to_action) - 1)
		return self.index_to_action[index]

class Investor(object):
	def __init__(self, number_of_stocks, gamma, alpha, scalar):
		self.Q = QApproximator(number_of_stocks, gamma, alpha, scalar)
		self.t = 0
		self.number_of_stocks = number_of_stocks
		self.last_sar = None

	def choose_action(self, state, practice=False):
		r = random.random()
		if r <= 0.5 / (1 + self.t/10000) and practice:
			action = self.Q.random_action()
		else:
			action = self.Q.maximal_action(state)
		self.t += 1
		return action

	def train(self, s, a, r, done):
		if not self.last_sar:
			self.last_sar = (s, a, r)
		else:
			self.Q.update(*(self.last_sar + (s, a)))
			self.last_sar = (s, a, r)
		if done:
			self.Q.update_terminal(*self.last_sar)

def invest(investor, stock_prices, initial_investment_func, 
		   cash_func, num_episodes, practice=False):
	portfolio_values = []
	for i in range(num_episodes):
		env = Portfolio(stock_prices, initial_investment_func(), cash_func())
		done = False
		state = env.get_state()
		while not done:
			action = investor.choose_action(state, practice)
			new_state, reward, done = env.take_action(action)
			if practice:
				investor.train(state, action, reward, done)
			state = new_state
		portfolio_values.append(env.portfolio_value())
	return portfolio_values, scalar

def basic_investment_func():
	return [0., 0., 0.]

def basic_cash_func():
	return 15000.

if __name__ == '__main__':
	data = load_data()
	training_data = data[:,:int(data.shape[1]*0.5)]
	testing_data = data[:,int(data.shape[1]*0.5):]
	# create scalar
	env = Portfolio(training_data, basic_investment_func(), basic_cash_func())
	scalar = get_scaler(env, 3)
	# create investor
	investor = Investor(3, 0.9, 0.001, scalar)
	# we train the investor
	print('train')
	pvs = invest(investor, training_data, basic_investment_func, 
		   		 basic_cash_func, 100, practice=True)
	print(pvs)
	print('test')
	# we test
	pvs = invest(investor, testing_data, basic_investment_func, 
		   		 basic_cash_func, 1, practice=False)
	print(pvs)
