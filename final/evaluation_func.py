import numpy as np

from environment import ConnectFourEnv

#this script was work in progress but has been scrapped since minimax is already good enough in it's current form and there is not enough time or priority for it.

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
        
        return np.divide(np.sum(values),3)
        
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
        Evaluation function for calculating the value for open chain.
        An open chain is any position where two or more plates align and there is still room to add more to either side.
        The value for player A will be the normalized difference between open chains of player A and player B, with two plates weighted as +1 and three as +2.
        
        :obs (tf.Tensor): A tensor of shape (M,N,2) representing the current state.
        :player (bool): Player from whose position to consider the evaluation (true == player one).
        
        :returns (float): A float value between -1 and 1.
        '''
        
        state = obs.numpy().astype('float32')
        
        #create states for player in 2D with oponents pieces as 2 and empty as 3 
        state_flat = None #ADD
        
        #create the necessary structuring elements for checking a 2D board for matches for two plates
#         horizontal_d, diagonal_d = np.array([[1,1,0]]), np.array([[1,0],[0,1]])
        
        #create the necessary structuring elements for checking a 2D board for matches for two plates
#         horizontal_t, diagonal_t = np.array([[1,1,1,0]]), np.array([[1,0,0],[0,1,0]])
        
        hits = np.zeros((8))
        
        
        #calculate weighted difference of sums for each player
        value = (hits[0]+hits[1])*2 + (hits[2]+hits[3])*3 - (hits[4]+hits[5])*2 - (hits[6]+hits[7])*3
        value = None #ADD normalize
        
        return value
    
#     def convolve_1D_hits(lines : list, element):
#         '''
#         ADD
#         '''
        
#         element_len = element.shape[0]
        
#         for i, line in enumerate(lines):
#             np.pad
            
        
#         hits = 0
#         for line in lines:
#             for i, val in enumerate(line):
#                 if line[i:]
        
#         return hits
    
#     def lineify(grid):
#         '''
#         ADD
#         '''
        
#         lines = []
#         manips = []
        
        
        
#         for line in grid:
#             lines += line
        
        