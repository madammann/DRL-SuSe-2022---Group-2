import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box

class ConnectFourEnv:
    '''
    A OpenAi Gym inspired environment, but not one built in OpenAi Gym.
    This is supposed to have the same functions, but creating one in OpenAI Gym seems like a bit overkill at the moment.
    '''
    
    def __init__(self, size, reward_setting={'step' : 1, 'draw' : 10, 'win' : 100}):
        self.size = (4, 4) #size is column, then row in this implementation
        
        if len(size) == 2 and size[0] >= 4 and size[1] >= 4:
            self.size = size
            
        self._grid = np.zeros(self.size)
        self.action_space = Discrete(self.size[0]) #discrete action space over all columns of grid
        self.observation_space = Box(low=-1, high=1, shape=self.size, dtype='int') #discrete space with 0 for no plate, -1 for red, 1 for blue
        self.terminal = False
        self.winner = None
        self.reward_setting = reward_setting
        self.turn = True
        
    def step(self, action):
        '''
        ADD
        '''
        
        pos = self._action(action)
        
        if pos == None:
            return self.grid2obs(), tf.constant([self.reward_setting['draw']]*2, dtype='float32'), tf.constant(self.terminal, dtype='bool')
        
        self.terminal = self._verify_terminal(pos)
        reward = self._calculate_reward()
        
        return self.grid2obs(), reward, tf.constant(self.terminal, dtype='bool')
    
    def render(self):
        '''TBD'''
        pass
    
    def reset(self):
        '''Resets the environment to the starting state.'''
        
        self._grid = np.zeros(self.size)
        self.terminal = False
        self.winner = None
    
    def _action(self, action):
        '''
        ADD
        Unavailable actions due to a full column automatically become the action higher, or in case of rightmost column, the leftmost option.
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
        ADD
        '''
        
        obs = np.zeros(self.size)
        blue = np.array(self._grid == 2).astype('int16')
        red = np.array(self._grid == 1).astype('int16')
        obs = obs - red + blue
        
        return tf.constant(obs,'float32')
    
    def _get_chain_length(self, col, pos, delta):
        '''
        ADD
        '''
        
        dpose = (pos[0]+delta[0],pos[1]+delta[1])
        if dpose[0] >= self.size[0] or dpose[0] < 0 or dpose[1] >= self.size[1] or dpose[1] < 0:
            return 0
        else:
            if self._grid[dpose] == col:
                val = 1
                val += self._get_chain_length(col, dpose, delta)
                return val
            else:
                return 0
    
    def _verify_terminal(self, pos):
        '''
        ADD
        '''
        
        col = self._grid[pos]
        
        chains = []
        for delta in [(1,0),(-1,0),(0,1),(0,-1),(-1,-1),(1,1),(1,-1),(-1,1)]:
            chains += [self._get_chain_length(col, pos, delta)]
        
        chains = [chains[0]+chains[1],chains[2]+chains[3],chains[4]+chains[5],chains[6]+chains[7]]
        
        if any([chain >= 4 for chain in chains]):
            self.winner = True if col == 2 else False
            return True
        
        return False
    
    def _calculate_reward(self):
        '''
        ADD
        '''
        
        if self.terminal:
            if self.winner:
                return tf.constant([self.reward_setting['win'],-self.reward_setting['win']], dtype='float32')
            
            else:
                return tf.constant([-self.reward_setting['win'],self.reward_setting['win']], dtype='float32')
            
        else:
            return tf.constant([self.reward_setting['step']]*2, dtype='float32')