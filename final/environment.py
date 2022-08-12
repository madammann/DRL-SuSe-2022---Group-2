import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box

class ConnectFourEnv:
    '''
    A OpenAi Gym inspired environment, but not one built in OpenAi Gym.
    This is supposed to have the same functions, but creating one in OpenAI Gym seems like a bit overkill at the moment.
    '''
    
    def __init__(self, size : tuple = (6,7), reward_setting={'step' : 1, 'draw' : 10, 'win' : 100}):
        '''
        Initialisation method for the environment.
        
        :param size (tuple): The size of the grid to be used in this environment.
        :param reward_setting (dict): A dictionary with keys "step", "draw", and "win" with their respective rewards as integer.
        '''
        
        self.size = size
        
        if len(size) == 2 and size[0] >= 4 and size[1] >= 4:
            self.size = size
            
        self._grid = np.zeros(self.size)
        self.action_space = Discrete(self.size[0]) #discrete action space over all columns of grid
        #TODO: Change observation space to include whose turn it is
        self.observation_space = Box(low=-1, high=1, shape=self.size, dtype='int') #discrete space with 0 for no plate, -1 for red, 1 for blue
        self.terminal = False
        self.winner = None
        self.reward_setting = reward_setting
        self.turn = True
        
    def step(self, action : int) -> tuple:
        '''
        The step method which is used to take an action in the environment.
        
        :param (int): A discrete action sampled from all actions in the action space for the size of this specific grid.
        
        :returns (tuple): A tuple of observation (grid, turn), reward, and terminal boolean for the resulting state after the action taken.
        '''
        
        pos = self._action(action)
        
        if pos == None:
            return self.grid2obs(), tf.constant([self.reward_setting['draw']]*2, dtype='float32'), tf.constant(self.terminal, dtype='bool')
        
        self.terminal = self._verify_terminal(pos)
        reward = self._calculate_reward()
        
        return (self.grid2obs(), int(self.turn)), reward, tf.constant(self.terminal, dtype='bool')
    
    def render(self):
        '''
        Here we could later create a pygame-based rendering or some form of printout of the environment observation.
        '''
        print(np.flip(self._grid,0).astype(int),"\n")

    
    def reset(self):
        '''
        Resets the environment to the starting state.

        :returns (tuple): A tuple of observation (grid, turn) and terminal boolean for the resulting state after the action taken.
        '''
        
        self._grid = np.zeros(self.size)
        self.terminal = False
        self.winner = None

        return (self.grid2obs(), int(self.turn)), tf.constant(self.terminal, dtype='bool')
    
    def _action(self, action : int) -> tuple:
        '''
        The method which actually takes the action in the environment.
        Unavailable actions due to a full column automatically become the action higher, or in case of rightmost column, the leftmost option.
        
        :param action (int): The action propagated from the step method.
        
        :returns (tuple): The coordinates in self._grid where a change has taken place for validation whether the game has terminated.
        '''
        
        start = action
        while self._grid[self.size[0]-1][action] != 0:
            action += 1
            
            if action >= self.size[0]:
                action = 0
                
            if action == start:
                self.terminal = True
                
                return None
        
        pos = (None, action)
        for i, space in enumerate(self._grid.T[action]):
            if space == 0:
                pos = (i, pos[1])
                
                break
        
        self._grid[pos] = 1 + int(self.turn)
        self.turn = not self.turn
        
        return pos
    
    def grid2obs(self):
        '''
        The method for transforming the internal numpy grid into a tensor to be used as observation.
        Here the conversion may be a bit inefficient but since Connect-4 is still relatively simple it should be efficient enough.
        
        :returns (tf.Tensor): An binary observation of shape (x, y, 2) will be returned in float32.
        '''
        
        blue = np.array(self._grid == 2).astype('int16')
        red = np.array(self._grid == 1).astype('int16')
        obs = np.dstack([blue,red])

        #First color in return (here blue) and a 2 in grid is the player who has opened the game
        
        return tf.constant(obs,'float32')

    
    def _get_chain_length(self, col : int, pos : tuple, delta : tuple) -> int:
        '''
        Method for checking the chain length from a specific position recursively.
        
        :param col (int): An integer representing the color in the grid to check.
        :param pos (tuple): A tuple representing the position from which to check.
        :param delta (tuple): A tuple representing the step to be taken from pos.
        
        :returns (int): An integer representing the chain length given a specific delta from pos.
        '''
        
        dpose = (pos[0]+delta[0],pos[1]+delta[1])
        
        if dpose[0] >= self.size[0] or dpose[0] < 0 or dpose[1] >= self.size[1] or dpose[1] < 0:
            return 1
        
        else:
            if self._grid[dpose] == col:
                val = 1
                val += self._get_chain_length(col, dpose, delta)
                
                return val
            
            else:
                return 1 # we return one here since we know the chain is at least of length one at pos
    
    def _verify_terminal(self, pos):
        '''
        Method for checking whether the grid state is terminal given a position from which to check for a line of four.
        
        :param pos (tuple): Position of the latest change in the grid as tuple of integers.
        
        :return (bool): True if terminal, False if not.
        '''
        
        col = self._grid[pos]
        
        chains = []
        for delta in [(1,0),(-1,0),(0,1),(0,-1),(-1,-1),(1,1),(1,-1),(-1,1)]:
            chains += [self._get_chain_length(col, pos, delta)]
        
        chains = [chains[0]+chains[1]-1,chains[2]+chains[3]-1,chains[4]+chains[5]-1,chains[6]+chains[7]-1] # one is substracted for each chain since each chain contains the element at pos
        
        if any([chain >= 4 for chain in chains]):
            self.winner = True if col == 2 else False
            
            return True
        
        return False
    
    def _calculate_reward(self):
        '''
        Method for calculating the reward for a step, draw or win and returning it as a tuple for all players.
        
        :returns (tuple): A float32 tuple of length 2 for player 1 and player 2's reward.
        '''
        
        if self.terminal:
            if self.winner:
                return tf.constant([self.reward_setting['win'],-self.reward_setting['win']], dtype='float32')
            
            else:
                return tf.constant([-self.reward_setting['win'],self.reward_setting['win']], dtype='float32')
            
        else:
            return tf.constant([self.reward_setting['step']]*2, dtype='float32')