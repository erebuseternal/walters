import random
import numpy as np

class Portfolio(object):
	def __init__(self, stock_prices, initial_investments, cash):
		self.stock_prices = stock_prices
		self.time = 0
		self.cash = cash
		self.investments = np.array(initial_investments)

	def take_action(self, action):
		starting_value = self.portfolio_value()
		new_investments = np.array(self.investments.shape[0])
		min_buy_price = float('inf')
		buys = []
		for i, action in enumerate(actions):
			if action == 'H':
				new_investments[i] = self.investments[i]
			price = self.stock_prices[i, self.time]
			number = self.investments[i]
			if action == 'B':
				buys.append(i, price)
				if min_buy_price > price:
					min_buy_price = price
			self.cash += price * number
		while self.cash >= min_buy_price:
			i, price = random.choice(buys)
			if price > cash:
				continue
			new_investments[i] += 1
			self.cash -= price
		self.time += 1
		self.investments = new_investments
		reward = self.portfolio_value - starting_value
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
	def __init__(self, number_of_stocks, gamma, alpha):
		self.gamma = gamma
		self.alpha = alpha
		self.index_to_action = {}
		self._create_index_to_action_map(number_of_stocks)
		self.params = np.random.randn((3 ** number_of_stocks, 2 * number_of_stocks + 2))

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
		actual = np.dot(self.params, s1)
		adjustment = np.dot(self.params, s1)
		adjustment[self._action_to_index[a1]] = (r + 
			self.gamma * np.dot(self.params, s2)[self._action_to_index[a2]])
		self.params += self.alpha * (np.dot((adjustment - actual), s1))

	def update_terminal(self, s, a, r):
		s1 = self._convert_state(s1)
		s2 = self._convert_state(s2)
		actual = np.dot(self.params, s1)
		adjustment = np.dot(self.params, s1)
		adjustment[self._action_to_index[a1]] = (r)
		self.params += self.alpha * (np.dot((adjustment - actual), s1))

	def _convert_state(self, s):
		return np.array(list(s) + [1])

	def maximal_action(self, s):
		s = self._convert_state(s)
		inference = np.dot(self.params, s)
		index = np.argmax(inference)
		return self.index_to_action[index]

	def random_action(self):
		index = random.randint(0, len(self._create_index_to_action_map) - 1)
		return self.index_to_action[index]

class Investor(object):
	def __init__(self, number_of_stocks, gamma, alpha):
		self.Q = QApproximator(number_of_stocks, gamma, alpha)
		self.t = 0
		self.number_of_stocks = number_of_stocks
		self.last_sar = None

	def choose_action(self, state, practice=False)
		r = random.random()
		if r <= 0.5 / (1 + t/10000) and practice:
			action = self.Q.random_action()
		else:
			action = self.Q.maximal_action(state)
		return action

	def train(self, s, a, r, done):
		if not self.last_sar:
			self.last_sar = (s, a, r)
		else:
			self.Q.update(*(self.last_sar + (s, a)))
			self.last_sar = (s, a, r)
		if done:
			self.Q.update(*self.last_sar)

def invest(investor, stock_prices_func, initial_investment_func, 
		   cash_func, num_episodes, practice=True):
	for _ in num_episodes:
		env = Portfolio(stock_prices_func(), initial_investment_func(), cash_func())
		done = False
		state = env.get_state()
		while not done:
			action = investor.choose_action(state, practice)
			new_state, reward, done = env.take_action(action)
			if practice:
				investor.train(state, action, reward, done)
			state = new_state
