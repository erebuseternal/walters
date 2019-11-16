import random
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

def run_temporal_difference(env_builder, gamma=0.9, learning_rate=0.1,
					iterations=10000, max_steps=10):
	Q = defaultdict(lambda: defaultdict(lambda: [0, 0]))
	t = 1.
	for _ in range(iterations):
		# build the environment
		env = env_builder()
		# select a random start
		env.state = random.choice(list(env.legal_states))
		steps = 0
		last_state_action_reward = None
		# play the game
		states = set()
		while not env.game_over() and steps < max_steps:
			state = env.get_state()
			states.add(state)
			r = random.random()
			if state not in Q or r <= 0.5 / (1 + t/10000):
				action = random.choice(list(env.get_actions()))
			else:
				action = sorted((action for action in Q[state]), 
							 	key=lambda action: Q[state][action],
							 	reverse=True)[0]
			reward = env.act(action)
			t += 1
			steps += 1
			if not last_state_action_reward:
				last_state_action_reward = (state, action, reward)
				continue
			last_state, last_action, last_reward = last_state_action_reward
			last_Q, current_Q = Q[last_state][last_action][0], Q[state][action][0]
			new_learning_rate = learning_rate / (1 + Q[last_state][last_action][1])
			Q[last_state][last_action][0] = (last_Q + new_learning_rate 
												* (last_reward + gamma * current_Q - last_Q))
			Q[last_state][last_action][1] += 0.01
			last_state_action_reward = (state, action, reward)
		# capture the last action that 
		# led us to the terminal state
		if last_state_action_reward:
			last_state, last_action, last_reward = last_state_action_reward
			last_Q = Q[last_state][last_action][0]
			new_learning_rate = learning_rate / (1 + Q[last_state][last_action][1])
			Q[last_state][last_action][0] = (last_Q + new_learning_rate 
												* (reward - last_Q))
			Q[last_state][last_action][1] += 1

	# finally we print out the decided
	# upon actions
	# initialize every cell to empty
	rows = [[' '] * env.size[0] for j in range(env.size[1])]
	# add in the tokens for blocked cells
	for state in Q:
		rows[state[1]][state[0]] = sorted((action for action in Q[state]), 
										  key=lambda action: Q[state][action],
										  reverse=True)[0]
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
	for state in Q:
		best_action = sorted((action for action in Q[state]), 
							  key=lambda action: Q[state][action],
							  reverse=True)[0]
		rows[state[1]][state[0]] = str(round(Q[state][best_action][0], 2))
	rep = ''
	for row in reversed(rows):
		rep += ' ' + ' '.join(['------'] * env.size[0]) + ' \n'
		rep += '| ' + ' | '.join(row) + ' |\n'
	rep += ' ' + ' '.join(['-----'] * env.size[0]) + ' \n'
	print(rep)

if __name__ == '__main__':
	Q = run_temporal_difference(costly_walk_builder)
	visualize_values(Q, costly_walk_builder)
