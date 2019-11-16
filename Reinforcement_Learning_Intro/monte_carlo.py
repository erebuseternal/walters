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

def run_monte_carlo(policy, env_builder, gamma=0.9, epsilon=0.1, iterations=1000, max_steps=25):
	Q = defaultdict(lambda: defaultdict(lambda: (0, 0)))
	for _ in range(iterations):
		# build the environment
		env = env_builder()
		# select a random start
		env.state = random.choice(list(env.legal_states))
		# play the game
		sars = []
		while not env.game_over() and len(sars) < max_steps:
			state = env.get_state()
			# if the policy doesn't know what to
			# do with this state, assign a random
			# action
			r = random.random()
			if state not in policy:
				policy[state] = random.choice(list(env.get_actions()))
			action = policy[state]
			if r <= epsilon:
				action = random.choice(list(env.get_actions()))
			reward = env.act(action)
			sars.append((state, action, reward, env.get_state()))
		# if we started in a terminal position,
		# play a new game
		if not sars:
			continue
		# initialize the returns by setting
		# the return of the terminal state to
		# zero
		returns = {
			sars[-1][-1]: 0
		}
		for state, action, reward, state2 in reversed(sars):
			returns[state] = reward + gamma * returns[state2]
		# update Q
		for state, action, _, _ in reversed(sars):
			mean, N = Q[state][action]
			Q[state][action] = (
				(returns[state] + N * mean) / (N + 1),
				N + 1
			)
		# update the policy with the argmax for each
		# known state
		for state in Q:
			action = sorted((action for action in Q[state]), 
							 key=lambda action: Q[state][action],
							 reverse=True)[0]
			policy[state] = action
	# finally we print out the decided
	# upon actions
	# initialize every cell to empty
	rows = [[' '] * env.size[0] for j in range(env.size[1])]
	# add in the tokens for blocked cells
	for state, action in policy.items():
		rows[state[1]][state[0]] = action
	# cobble together the string
	rep = ''
	for row in reversed(rows):
		rep += ' ' + ' '.join(['-----'] * env.size[0]) + ' \n'
		rep += '|  ' + '  |  '.join(row) + '  |\n'
	rep += ' ' + ' '.join(['-----'] * env.size[0]) + ' \n'
	print(rep)
	return policy, Q

if __name__ == '__main__':
	policy = {}
	run_monte_carlo(policy, costly_walk_builder)
	policy = {}
	run_monte_carlo(policy, basic_env_builder)




