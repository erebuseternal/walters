import numpy as np


class GridWorld(object):
	"""
	-------------------------
	|     |     |     |  1  |
	-------------------------
	|     |  x  |     | -1  |
	-------------------------
	|  o  |     |     |     |
	-------------------------
	"""

	def __init__(self, start=(0, 0)):
		self.start = start
		self.i = start[0]
		self.j = start[1]
		self.values = self._initialize_values()

	def _initialize_values(self):
		self.values = {(i, j): 0.5 for i in range(4) for j in range(3)}
		del self.values[(1, 1)]
		self.values[(3, 2)] = 1
		self.values[(3, 1)] = -1

	def act(self, action, undo=False):
		step = (1 if not undo else -1)
		if action == 'U':
			self.j += step
		elif action == 'D':
			self.j -= step
		elif action == 'R':
			self.i += step
		else:
			self.i -= step
		if (not (-1 < self.i < 4) or not (-1 < self.j < 3) 
			or (self.i == 1 and self.j == 1)):
			self.act(action, undo=(not undo))
			return False
		else:
			return True

	def reset(self, start=None):
		if not start:
			start = self.start
		self.i = start[0]
		self.j = start[1]

	def game_over(self):
		print(self.i, self.j)
		if self.i == 3 and self.j == 2:
			return 1
		elif self.i == 3 and self.j == 1:
			return -1
		else:
			return 0

	def __str__(self):
		rows = [[' '] * 4 for j in range(3)]
		rows[self.j][self.i] = 'o'
		rows[1][1] = 'x'
		rep = ''
		for row in reversed(rows):
			rep += ' '.join(['-'*5 for i in range(4)]) + '\n'
			rep += '  ' + '  |  '.join(row) + '  \n'
		rep += ' '.join(['-'*5 for i in range(4)]) + '\n'
		return rep
