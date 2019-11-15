import random
from collections import defaultdict
from grid_world import GridWorld

def play_game(policy, env, max_actions=100):
	sars = []
	while not env.game_over() and len(sars) < 100:
		state = env.get_state()
		action = policy[state]
		reward = env.act(action)
		sars.append((state, action, reward, env.get_state()))
	return sars

def back_compute_returns(sars, gamma):
	# initialize terminal state to 0 returns
	returns = {
		sars[-1][-1]: 0
	}
	# back compute returns
	for state, action, reward, state2 in reversed(sars):
		returns[state] = reward + gamma * returns[state2]
	return returns

def evaluate_policy(policy, env_builder, gamma=0.9, iterations=100):
	return_dists = defaultdict(list)
	for _ in range(iterations):
		env = env_builder()
		sars = play_game(policy, env)
		for state, _return in back_compute_returns(sars, gamma).items():
			return_dists[state].append(_return)
	returns = {state: sum(return_dists[state]) / len(return_dists[state]) 
			   for state in return_dists}
	return returns

def basic_env_builder():
	rewards = {(i, j): 0.0 for i in range(4) for j in range(3)}
	rewards[(3, 2)] = 1.
	rewards[(3, 1)] = -1.
	env = GridWorld(rewards)
	return env

def random_start_env_builder():
	rewards = {(i, j): 0.0 for i in range(4) for j in range(3)}
	rewards[(3, 2)] = 1.
	rewards[(3, 1)] = -1.
	options = [(i, j) for i in range(4) for j in range(3)
				if (i, j) not in [(1, 1), (3, 2), (3, 1)]]
	env = GridWorld(rewards, starting_state=random.choice(options))
	return env

def basic_policy():
	# go right if you can,
	# otherwise go up
	return {
		(0, 0): 'R',
		(1, 0): 'R',
		(2, 0): 'R',
		(3, 0): 'U',
		(0, 1): 'U',
		(2, 1): 'R',
		(0, 2): 'R',
		(1, 2): 'R',
		(2, 2): 'R'
	}

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


if __name__ == '__main__':
	policy = basic_policy()
	returns = evaluate_policy(policy, random_start_env_builder)
	view_returns(returns, (4, 3))




