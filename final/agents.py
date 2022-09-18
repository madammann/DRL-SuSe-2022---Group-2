import numpy as np
import tensorflow as tf

from environment import ConnectFourEnv

from minimax import MinimaxNode, Minimax
from model import ConnectFourModel
import neat
import pickle

from copy import deepcopy

class AgentBase:
    '''
    ADD
    '''
    
    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.average_speed = 0
    
    def reset(self):
        self.wins = 0
        self.losses = 0
        self.average_speed = 0
        
    def __str__(self):
        '''
        ADD
        '''
        
        return f'Agent performance {self.wins, self.losses, self.wins+self.losses} with avg speed {self.average_speed} secs/move.'

class MinimaxAgent(AgentBase):
    '''
    ADD
    '''
    
    def policy(self, env, eval_func=None):
        '''
        ADD
        '''
        
        tree = Minimax(env, depthmax=3)
        
        policy = tree(env.turn, eval_func=eval_func)
        
        return policy
    
    def select_move(self, env, eval_func=None):
        '''
        ADD
        '''
        
        policy = self.policy(env, eval_func=eval_func)
        
        action = np.argmax(policy)
        
        return action
    
#     def select_appropriate_depthmax(self, env):
#         '''
#         ADD
#         '''
        
#         branching_factor = len(env.action_range)
#         node_max = 10e8
        
#         nodes = branching_factor
#         depth = 1
#         while True:
#             nodes += nodes*branching_factor
            
#             if nodes <= node_max:
#                 depth += 1
                
#             else:
#                 return depth

class RandomAgent(AgentBase):
    '''
    Agent playing a (valid) random move
    '''
    
    def select_move(self, env):

        action = env.get_random_valid_action()
        
        return action

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
        argsorted = np.argsort(policy)[::-1]
        
        #if there two or more choices are equally good, make a random choice between the n max elements
        if policy[argsorted[0]] == policy[argsorted[1]]:
            indices = np.where(policy == policy[argsorted[0]])[0]
            choice = np.random.choice(indices)
        
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
    
    #TBD Implement a __init__ overwrite of parent class which allows for specific model loading
    def __init__(self, model_name : str):
        '''
        ADD
        '''
        
        super(ModelAgent, self).__init__()
        
        self.model = None
        if model_name == 'd6by7':
            self.model = ConnectFourModel((6,7), 0.002)
            self.model(tf.random.normal((1, 6, 7, 3)))
            self.model.load('./weights/d6by7.h5')
            
        elif model_name == 'd8by9':
            self.model = ConnectFourModel((8,9), 0.002)
            self.model(tf.random.normal((1, 8, 9, 3)))
            self.model.load('./weights/d8by9.h5')
        
        elif model_name == 'd10by11':
            self.model = ConnectFourModel((10,11), 0.002)
            self.model(tf.random.normal((1, 10, 11, 3)))
            self.model.load('./weights/d10by11.h5')
        
        elif model_name == 'd12by13':
            self.model = ConnectFourModel((12,13), 0.002)
            self.model(tf.random.normal((1, 12, 13, 3)))
            self.model.load('./weights/d12by13.h5')
        
        else:
            raise ValueError('The model which was requested is not supported or does not exist.')
        
    def policy(self, env):
        '''
        ADD
        '''
        
        obs = env.grid2obs()
        
        policy = self.model(tf.expand_dims(obs,axis=0)).numpy()
        
        return policy
    
    def select_move(self, env):
        '''
        ADD
        '''
        
        policy = self.policy(env)
        
        action = np.argmax(policy)
        
        return int(action)


class NeuroevolutionAgent(AgentBase):
    '''
    Agent that uses an ANN previously trained via neuroevolution to make their move
    '''

    def __init__(self, path: str):
        '''
        ADD
        '''

        super(NeuroevolutionAgent, self).__init__()

        #load model resulting from neuroevolution from file
        genome_storage = open(path, 'rb')
        model_genome, neat_config = pickle.load(genome_storage)
        self.model = neat.nn.FeedForwardNetwork.create(model_genome, neat_config)

        self.input_size = None  # add input sized for models by name here

    def policy(self, env):
        '''
        ADD
        '''
        
        # change environment observation format to 1-grid representation
        save_obs_setting = env.grid2obs
        env.grid2obs = env.grid2obs_1grid

        observation = env.grid2obs()

        flat_observation = tf.reshape(observation, [-1]).numpy()  # Prepare observation for input
        output = self.model.activate(flat_observation)

        #restore observation format setting for environment
        env.grid2obs = save_obs_setting

        return output

    def select_move(self, env):
        '''
        ADD
        '''

        policy = self.policy(env)

        action = np.argmax(policy)

        return int(action)
