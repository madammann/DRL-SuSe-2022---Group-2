import numpy as np
import tensorflow as tf

from environment import ConnectFourEnv

from minimax import MinimaxNode, Minimax
from model import ConnectFourModel

from copy import deepcopy

class AgentBase:
    '''
    ADD
    '''
    
    def __init__(self):
        self.winrate = np.nan()
        self.games = 0

class MinimaxAgent(AgentBase):
    '''
    ADD
    '''
    
    def policy(self, env):
        '''
        ADD
        '''
        
        self.depthmax = self.select_appropriate_depthmax(env)
        tree = Minimax(env, depthmax=self.depthmax)
        
        policy = tree(env.turn)
        
        return policy
    
    def select_move(self, env):
        '''
        ADD
        '''
        
        policy = self.policy(env)
        
        action = np.argmax(policy)
        
        return action
    
    def select_appropriate_depthmax(self, env):
        '''
        ADD
        '''
        
        branching_factor = len(env.action_range)
        node_max = 10e8
        
        nodes = branching_factor
        depth = 1
        while True:
            nodes += nodes*branching_factor
            
            if nodes <= node_max:
                depth += 1
                
            else:
                return depth

class RandomAgent(AgentBase):
    '''
    ADD
    '''
    
    def policy(self, env):
        '''
        ADD
        '''
        
        policy = np.random.uniform(low=0, high=1, shape=(len(env.action_range)), dtype='float32')
        
        return policy
    
    def select_move(self, env):
        '''
        ADD
        '''
        
        policy = self.policy(env)
        
        return np.argmax(policy)

class AvoidNextLossAgent(AgentBase):
    '''
    ADD
    '''
    
    def policy(self, env):
        '''
        ADD
        '''

        policy = np.full((10), 1, dtype='float32')
    
        args = [deepcopy(env) for _ in range(len(env.action_range))]
        
        return policy
    
    def select_move(self, env):
        '''
        ADD
        '''
        
        policy = self.policy(env)
        choice = np.argmax(policy)
        
        return choice

    def create_all_child_envs(self, env):
        '''
        ADD
        '''
        
        child_envs = []
        
        for action in range(len(env.action_range)):
            child_env = deepcopy(env)
            child_env.step(action)
            child_env += [child_env]
        
        return child_envs

class ModelAgent(AgentBase):
    '''
    ADD
    '''
    
    #TBD Implenet a __init__ overwrite of parent class which allows for specific model loading
    def __init__(self, model_name : str):
        '''
        ADD
        '''
        
        super(ModelAgent, self).__init__()
        
        self.model = None #load a specific model here
        self.input_size = None #add input sized for models by name here
        
    def policy(self, env):
        '''
        ADD
        '''
        
        obs = env.grid2obs()
        input_size = tuple(obs.shape) #may not work change after testing
        
        if input_size != self.input_size:
            raise ValueError('Model does not support input shape of passed environment.')
        
        policy = self.model(obs).numpy()
        
        return policy
    
    def select_move(self, env):
        '''
        ADD
        '''
        
        policy = self.policy(env)
        
        action = np.argmax(policy)
        
        return action