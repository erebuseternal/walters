class GridWorld(object):
	def __init__(self, reward_map=None, starting_state=(0, 0), 
				 size=(4, 3), blocked_states=((1, 1),), 
				 terminal_states=((3, 2), (3, 1))):
		# initialize the actions map
		self.size = size
		self.blocked_states = blocked_states
		self.terminal_states = terminal_states
		self.actions_map = self._initialize_actions_map(size, blocked_states, terminal_states)
		self.legal_states = set([(i, j) for i in range(size[0]) for j in range(size[1])])
		self.legal_states -= set(blocked_states)
		# initialize the state
		if starting_state not in self.legal_states:
			raise Exception("GridWorld Exception: starting state isn't legal")
		self.state = starting_state
		# check the rewards match the grid world
		if not reward_map:
			raise Exception('GridWorld Exception: no rewards specified')
		states_not_covered = self.legal_states - set(reward_map)
		if states_not_covered:
			raise Exception('GridWorld Exception: \
				these states have no assigned reward: %s' % states_not_covered)
		self.reward_map = reward_map

	@staticmethod
	def _initialize_actions_map(size, blocked_states, 
									 terminal_states):
		# initialize all all states to all actions
		actions_map = {(i, j): set('UDRL')
					   for i in range(size[0])
					   for j in range(size[1])}
		# remove left from left edge
		for j in range(size[1]):
			actions_map[(0, j)] -= set('L')
		# remove right from right edge
		for j in range(size[1]):
			actions_map[(size[0] - 1, j)] -= set('R')
		# remove down from bottom edge
		for i in range(size[0]):
			actions_map[(i, 0)] -= set('D')
		# remove up from top edge
		for i in range(size[0]):
			actions_map[(i, size[1] - 1)] -= set('U')
		# remove actions on terminal states
		for state in terminal_states:
			actions_map[state] = set()
		# remove actions that would put you on a
		# blocked state
		for state in blocked_states:
			left = (state[0] - 1, state[1])
			if left in actions_map:
				actions_map[left] -= set('R')
			right = (state[0] + 1, state[1])
			if right in actions_map:
				actions_map[right] -= set('L')
			below = (state[0], state[1] - 1)
			if below in actions_map:
				actions_map[below] -= set('U')
			above = (state[0], state[1] + 1)
			if above in actions_map:
				actions_map[above] -= set('D')
		return actions_map

	def get_state(self):
		return self.state

	def get_actions(self):
		return self.actions_map[self.state]

	def game_over(self):
		# game is over if we're in a terminal state
		return self.actions_map[self.state] == set()

	def act(self, action):
		if action not in self.get_actions():
			raise Exception('GridWorld Exception: Tried to make an illegal move')
		if action == 'L':
			self.state = (self.state[0] - 1, self.state[1])
		elif action == 'R':
			self.state = (self.state[0] + 1, self.state[1])
		elif action == 'D':
			self.state = (self.state[0], self.state[1] - 1)
		else:
			self.state = (self.state[0], self.state[1] + 1)
		return self.reward_map[self.state]

	def __str__(self):
		# initialize every cell to empty
		rows = [[' '] * self.size[0] for j in range(self.size[1])]
		# add in the tokens for blocked cells
		for state in self.blocked_states:
			rows[state[1]][state[0]] = 'B'
		# add in tokens for terminal states
		for state in self.terminal_states:
			rows[state[1]][state[0]] = 'T'
		# place a token for the current state
		rows[self.state[1]][self.state[0]] = 'x'
		# cobble together the string
		rep = ''
		for row in reversed(rows):
			rep += ' ' + ' '.join(['-----'] * self.size[0]) + ' \n'
			rep += '|  ' + '  |  '.join(row) + '  |\n'
		rep += ' ' + ' '.join(['-----'] * self.size[0]) + ' \n'
		return rep
