import numpy as np
from copy import deepcopy
from env import ConnectFourEnv

from multiprocessing.Pool import ThreadPool

#write warning for larger action space or larger depth choice

class MinimaxNode:
    def __init__(self, move : int, parent=None):
        '''
        ADD
        '''
        
        self.move = move
        self.value = float(0)
        self.proven = False
        self.children = []
        self.parent = parent
    
class Minimax:
    def __init__(self, env, depthmax=6):
        '''
        ADD
        '''
        
        self.starting = env
        self.action_space = range(len(env.action_space))
        self.depthmax = depthmax
        self.tree = {0 : [MinimaxNode(None)]}
        
        for depth in range(depthmax):
            self.tree[depth] = []
            
            for i, node in enumerate(self.tree[depth-1]):
                self.tree[depth] += self._add_child_nodes(i)
    
    def __call__(self, player : bool, eval_func=None):
        '''
        ADD
        '''
        
        #integrate test for terminal base state and for true depthmax for close to terminal states
        
        depth = 0
        
        while depth =< self.depthmax:
            for i, node in enumerate(self.tree[depth]):
                None
                
                #propagate true result of s_t upward as soon as all successor states S_t+1 have been evaluated
                if i % len(self.action_space) == 0:
                    self._uppropagate(depth-1,node.parent)
                
    
    def _uppropagate(self, startdepth : int, parent : int):
        '''
        ADD
        '''
        
        depth = startdepth
        
        #unless broken uppropagate in this loop until before reaching the root
        while depth > 0:
            
            #test if all information for current propagation is available
            if not startdepth and all([True]):
                pass
            
            #break the loop in case there is not enough information available for the next propagation
            else:
                pass
            
    def _downpropagate(self, depth : int, index : int):
        pass
            
    
    def _get_parent_chain(self, depth, index, chain=[]):
        '''
        ADD
        '''
        
        chain = chain
        
        chain += [self.tree[depth][index].move]
        
        return chain[::-1]
    
    def _get_value_at(self, chain, eval_func=None):
        '''
        ADD
        '''
        
        env = deepcopy(self.starting)
        
        for move in chain:
            if not env.terminal:
                env.step(move)
                
                if env.terminal:
                    return 'ADD'
            
            else:
                pass
            pass
        pass
    
        def _add_child_nodes(self, parent_idx) -> list:
        '''
        ADD
        '''
        
        children = []
        for action in self.action_space:
            children += [MinimaxNode(action,parent_idx)]
            
        return children