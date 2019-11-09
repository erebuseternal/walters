import numpy as np
import random


NUM_TO_TOKEN = {
    1: 'x',
    -1: 'o',
    0: ' '
}
TOKEN_TO_NUM = {token: num for num, token in NUM_TO_TOKEN.items()}


class Environment(object):
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.winner = 0
        
    def is_empty(self, i, j):
        return self.board[i][j] == 0
    
    def play(self, i, j, num):
        self.board[i][j] = num
        
    def try_play(self, i, j, num):
        self.board[i][j] = num
        h = self.state_hash()
        self.board[i][j] = 0
        return h
        
    def state_hash(self):
        h = 0.
        for i in range(3):
            for j in range(3):
                bit = i * 3 + j
                h += (self.board[i][j] + 1) * (3 ** bit)
        return int(h)
    
    def get_winner(self, force_recalculation=False):
        if self.winner and not force_recalculation:
            return self.winner
        
        self.winner = 0
        
        for i in range(3):
            s = np.sum(self.board[i])
            if s == -3 or s == 3:
                self.winner = s / 3
            
        for j in range(3):
            s = np.sum(self.board[:,j])
            if s == -3 or s == 3:
                self.winner = s / 3
        
        s = np.trace(self.board)
        if s == -3 or s == 3:
            self.winner = s / 3
        
        s = np.trace(np.fliplr(self.board))
        if s == -3 or s == 3:
            self.winner = s / 3
        
        return self.winner
        
    def game_over(self):
        return self.get_winner() or 0 not in self.board
                
    def draw(self):
        for i in range(3):
            print('---------')
            print(' ' 
                  + '  '.join([NUM_TO_TOKEN[num] 
                               for num in self.board[i]])
                  + ' ')
        print('---------')
        
    def reset(self):
        self.board = np.zeros((3, 3))
        self.winner = 0
        
    def find_winning_positions(self, i=0, j=0, winners=None):
        if winners is None:
            winners = np.zeros(3**9)
        if i == 3:
            winners[self.state_hash()] = self.get_winner(force_recalculation=True)
            return winners
        for fill in range(-1, 2):
            self.board[i][j] = fill
            if j == 2:
                self.find_winning_positions(i+1, 0, winners)
            else:
                self.find_winning_positions(i, j+1, winners)
        return winners
        
        
class Agent(object):
    def __init__(self, token):
        self.num = TOKEN_TO_NUM[token]
        self._initialize_values()
        self.history = []
        self.last_state_hash = None
    
    def _initialize_values(self):
        winning_positions = Environment().find_winning_positions()
        agent_wins = np.where(winning_positions == self.num, 1, 0)
        agent_loses = np.where(winning_positions == -self.num, -1, 0)
        wins_or_loses = agent_wins + agent_loses
        self.values = np.where(wins_or_loses == 0, 0.0, 0) + wins_or_loses
        
    def play(self, env, epsilon, verbose=False):
        starting_state_hash = env.state_hash()
        if self.last_state_hash is not None:
            self.history.append((
                self.last_state_hash,
                starting_state_hash
            ))
        r = np.random.random()
        possible_plays = []
        for i in range(3):
            for j in range(3):
                if env.is_empty(i, j):
                    possible_plays.append((i, j))
        if r < epsilon:
            if verbose:
                print('AI is making random move, cause why the fuck not')
            i, j = random.choice(possible_plays)
        else:
            play_values = []
            best_play = None
            best_play_value = -float('inf')
            for i, j in possible_plays:
                play_hash = env.try_play(i, j, self.num)
                play_value = self.values[play_hash]
                play_values.append((i, j, play_hash, round(play_value, 3)))
                if play_value > best_play_value:
                    best_play = i, j
                    best_play_value = play_value
            if verbose:
                print('AI is using the following values: ' + str(play_values))
            i, j = best_play
        env.play(i, j, self.num)
        final_state_hash = env.state_hash()
        self.history.append((
            starting_state_hash,
            final_state_hash
        ))
        self.last_state_hash = final_state_hash
        
    def update(self, env, learning_rate):
        final_state_hash = env.state_hash()
        winner = env.get_winner()
        reward = winner / self.num
        reward = reward if reward == 1 else 0
        self.values[final_state_hash] = reward
        if final_state_hash != self.last_state_hash:
            self.history.append((
                self.last_state_hash,
                final_state_hash
            ))
        for start, final in reversed(self.history):
            self.values[start] = self.values[start] + learning_rate * (self.values[final] - self.values[start])
        self.history = []
            
class Human(object):
    
    def __init__(self, token):
        self.num = TOKEN_TO_NUM[token]
        
    def play(self, env, *args, **kwargs):
        legal_move = False
        while not legal_move:
            move = input("Enter your move in coordinates i,j: ")
            i, j = move.split(',')
            i = int(i.strip())
            j = int(j.strip())
            legal_move = env.is_empty(i, j)
        env.play(i, j, self.num)
        
    def update(self, *args):
        pass
                
        
def play_game(env, player1, player2, learning_rate=0.01, epsilon=0.01, verbose=True):
    env.reset()
    current_player = player1
    while not env.game_over():
        if verbose: 
            env.draw()
        current_player.play(env, epsilon, verbose=verbose)
        if current_player is player1:
            current_player = player2
        else:
            current_player = player1
    if verbose: 
        env.draw()
        print('GAME OVER!')
    player1.update(env, learning_rate)
    player2.update(env, learning_rate)
        

if __name__ == '__main__':
    env = Environment()
    agent1 = Agent('x')
    agent2 = Agent('o')
    for i in range(10000):
        if i and i % 100 == 0:
            print(i)
        play_game(env, agent1, agent2, epsilon=0.2, learning_rate=0.5, verbose=False)
    """
    agent3 = Agent('o')
    for i in range(10000):
        if i and i % 100 == 0:
            print(i)
        play_game(env, agent1, agent3, epsilon=0.2, learning_rate=0.5, verbose=False)
    for i in range(10000):
        if i and i % 100 == 0:
            print(i)
        play_game(env, agent1, agent2, epsilon=0.2, learning_rate=0.5, verbose=False)
    """
    
    human = Human('o')
    play_another = True
    while play_another:
        play_game(env, agent1, human)
        decision = input('Would you like to play another game? Y/N: ').strip()
        if decision == 'N':
            play_another = False