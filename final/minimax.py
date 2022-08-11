import numpy as np
from copy import deepcopy
from env import ConnectFourEnv

class MinimaxNode:
    def __init__(self, move : int, parent=None):
        self.move = move
        self.value = float(0)
        self.proven = False
        self.children = []
        self.parent = parent

class LeafGenerator:
    '''
    Generator object which receives a current position and generates the grids of all children of it.
    '''
    
    def __init__(self, pos, depth, action_space):
        self.pos = pos
        self.depth = depth
        self.idx = 0
        self.action_space = action_space
        
    def __iter__(self):
        '''
        ADD
        '''
        
        
        
        
        yield 0
    
class Minimax:
    def __init__(self, env):
        self.starting = env
        self.action_space = range(len(env.action_space))
        self.tree = {0 : [MinimaxNode(None)]}
    
    def __call__(self, depthmax=4, eval_func=None):
        depth = 0
        while depth =< depthmax:
            pass
        
    def create_child_envs(self, env):
        child_envs = [deepcopy(env) for _ in range(len(env.action_space))]
        for i, env in enumerate(child_envs):
            