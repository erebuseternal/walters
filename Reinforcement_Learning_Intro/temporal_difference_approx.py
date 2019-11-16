import random
import numpy as np
from collections import defaultdict
from grid_world import GridWorld

def basic_env_builder():
	rewards = {(i, j): 0.0 for i in range(4) for j in range(3)}
	rewards[(3, 2)] = 1.
	rewards[(3, 1)] = -1.
	env = GridWorld(rewards)
	return env

def costly_walk_builder():
	rewards = {(i, j): -0.5 for i in range(4) for j in range(3)}
	rewards[(3, 2)] = 1.
	rewards[(3, 1)] = -1.
	env = GridWorld(rewards)
	return env

def view_returns(returns, size):
	# initialize every cell to empty
	rows = [[' N/A'] * size[0] for j in range(size[1])]
	for state, _return in returns.items():
		rows[state[1]][state[0]] = str(round(_return, 2))
	# cobble together the string
	rep = ''
	for row in reversed(rows):
		rep += ' ' + ' '.join(['------'] * size[0]) + ' \n'
		rep += '| ' + ' | '.join(row) + ' |\n'
	rep += ' ' + ' '.join(['------'] * size[0]) + ' \n'
	print(rep)

def basic_state_to_vector_func(s, a):
	"""
	Builds a vector of length 25
	"""
	return ([
		s[0] * (1. if a == action else 0.)
		for action in 'UDLR'
	]
	+ [
		s[1] * (1. if a == action else 0.)
		for action in 'UDLR'
	]
	+ [
		(s[0] ** 2) * (1. if a == action else 0.)
		for action in 'UDLR'
	]
	+ [
		(s[1] ** 2) * (1. if a == action else 0.)
		for action in 'UDLR'
	]
	+ [
		(s[0] * s[1]) * (1. if a == action else 0.)
		for action in 'UDLR'
	]
	+ [
		s[0],
		s[1],
		s[0] ** 2,
		s[1] ** 2,
		s[0] * s[1]
	])

class QClass(object):
	"""
	This class needs the ability to be updated with a new
	sars'a' and also return the maximal action
	"""
	def __init__(self, alpha, gamma, state_to_vector_func, num_features):
		self.alpha = alpha
		self.gamma = gamma
		self.state_to_vector_func = state_to_vector_func
		self.params = np.random.randn(num_features + 1)

	def update(self, s, a, r, s2, a2):
		self.params += (self.alpha 
						* (r + self.gamma * self.infer(s2, a2)
						   - self.infer(s, a)
						   ) 
						* self.state_to_vector(s, a))

	def update_terminal(self, s, a, r):
		self.params += (self.alpha 
						* (r - self.infer(s, a)) 
						* self.state_to_vector(s, a))

	def infer(self, s, a):
		return np.dot(self.params, self.state_to_vector(s, a)) 

	def state_to_vector(self, s, a):
		return np.array(list(self.state_to_vector_func(s, a)) + [1.])

	def maximal_action(self, s, actions):
		best_action = None
		best_inference = -float('inf')
		for a in actions:
			inference = self.infer(s, a)
			if inference > best_inference:
				best_action = a
				best_inference = inference
		return best_action

def run_temporal_difference_approx(env_builder, state_to_vector_func, 
							num_features, gamma=0.9, alpha=0.001,
							iterations=10000, max_steps=100):
	Q = QClass(alpha, gamma, state_to_vector_func, num_features)
	t = 0
	for _ in range(iterations):
		# build the environment
		env = env_builder()
		# select a random start
		env.state = random.choice(list(env.legal_states))
		steps = 0
		last_state_action_reward = None
		# play the game
		while not env.game_over() and steps < max_steps:
			state = env.get_state()
			r = random.random()
			if r <= 0.5 / (1 + t/10000):
				action = random.choice(list(env.get_actions()))
			else:
				action = Q.maximal_action(state, env.get_actions())
			reward = env.act(action)
			t += 1
			steps += 1
			if not last_state_action_reward:
				last_state_action_reward = (state, action, reward)
				continue
			last_state, last_action, last_reward = last_state_action_reward
			Q.update(last_state, last_action, last_reward, state, action)
			last_state_action_reward = (state, action, reward)
		# capture the last action that 
		# led us to the terminal state
		if last_state_action_reward:
			last_state, last_action, last_reward = last_state_action_reward
			Q.update_terminal(last_state, last_action, last_reward)
	# finally we print out the decided
	# upon actions
	# initialize every cell to empty
	rows = [[' '] * env.size[0] for j in range(env.size[1])]
	# add in the tokens for blocked cells
	for state in env.legal_states:
		env.state = state
		if not env.get_actions():
			continue
		rows[state[1]][state[0]] = Q.maximal_action(state, env.get_actions())
	# cobble together the string
	rep = ''
	for row in reversed(rows):
		rep += ' ' + ' '.join(['-----'] * env.size[0]) + ' \n'
		rep += '|  ' + '  |  '.join(row) + '  |\n'
	rep += ' ' + ' '.join(['-----'] * env.size[0]) + ' \n'
	print(rep)
	return Q

def visualize_values(Q, env_builder):
	env = env_builder()
	rows = [[' N/A'] * env.size[0] for j in range(env.size[1])]
	# add in the tokens for blocked cells
	for state in env.legal_states:
		env.state = state
		if not env.get_actions():
			continue
		action = Q.maximal_action(state, env.get_actions())
		rows[state[1]][state[0]] = str(round(Q.infer(state, action), 2))
	rep = ''
	for row in reversed(rows):
		rep += ' ' + ' '.join(['------'] * env.size[0]) + ' \n'
		rep += '| ' + ' | '.join(row) + ' |\n'
	rep += ' ' + ' '.join(['-----'] * env.size[0]) + ' \n'
	print(rep)

if __name__ == '__main__':
	Q = run_temporal_difference_approx(costly_walk_builder, 
									   basic_state_to_vector_func,
									   25)
	visualize_values(Q, costly_walk_builder)
