import numpy as np

from environment import ConnectFourEnv

class EvaluationFunction:
    '''
    ADD
    '''
    
    def __init__(self, params):
        '''
        ADD
        '''
        
        self.weights = np.ones((3),dtype='float32')
#         self.meta_weights = np.ones((3),dtype='float32') #may be implemented later
        
    def __call__(self, env, player : bool):
        '''
        ADD
        '''
        
        obs = env.grid2obs()
        
        values = np.zeros((3),dtype='float32')
        
        values[0] = self.get_possible_wins_val(obs, player) * self.weights[0]
        values[1] = self.get_options_left_val(obs, player) * self.weights[1]
        values[2] = self.get_open_chains_val(obs, player) * self.weights[2]
        
        return np.sum(values)
        
    def get_possible_wins_val(self, obs, player : bool):
        '''
        ADD
        '''
        
        pass
    
    def get_options_left_val(self, obs, player : bool):
        '''
        ADD
        '''
        
        pass
    
    def get_open_chains_val(self, obs, player : bool):
        '''
        ADD
        '''
        
        pass