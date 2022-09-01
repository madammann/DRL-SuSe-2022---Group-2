import numpy as np
from copy import deepcopy
from environment import ConnectFourEnv

from multiprocessing.pool import ThreadPool

class MinimaxNode:
    def __init__(self, move : int, parent=None):
        '''
        A node object for the Minimax search tree.
        
        :att move (int): An integer representing the action taken to arrive at this node's position.
        :att value (float): A float value representing the value for the current player between -1 and 1.
        :att proven (bool): A boolean value whether this is the game-theoretical value.
        :att visited (bool): A boolean value whether this node was visited already, used in minimax search for efficient traversal.
        :att children (list): A list of all integer indices for all children.
        :att parent (int): Either none if root, else an integer representing the parent of the depth-1 layer of the tree.
        '''
        
        self.move = move
        self.value = float(0)
        self.proven = False
        self.visited = False
        self.children = []
        self.parent = parent
    
class Minimax:
    '''
    Minimax tree search class for Connect-4 environments, written mostly unspecific so other enviornments may also work.
    To use this class create an object by calling the __init__ method useing an environment of the starting state and a maximum search depth.
    Make sure to watch that the action space branching factor is not too high so the maximum depth does not make it explode.
    
    ABOUT THE ENVIRONMENT
    
    USAGE EXPL
    
    :att starting (ConnectFourEnv): The starting state environment passed (certain attributes of env are required).
    :att action_space (func): A range function for the action space of the environment.
    :att depthmax (int): The maximum search depth.
    :att tree (dict): The search tree as a mapping from depth to a list of node objects.
    '''
    def __init__(self, env, depthmax=6):
        '''
        The initialization method for the class.
        Checks if the to-be-created tree has less or equal to 100.000.000 nodes and if yes creates the whole tree with empty values.
        If not it will produce an error.
        To get the actual search results the call method has to be used.
        
        Warning: Using a depthmax too large or even a smaller depthmax with a high branching factor may result in exploding complexity!
        
        :param env (ConnectFourEnv): A Connect-4 environment representing the starting or root state.
        :param depthmax (int): The maximum depth of the tree.
        '''
        
        self.starting = env
        self.action_space = env.action_range
        self.depthmax = depthmax
        self.tree = {0 : [MinimaxNode(None)]}
        
        if len(self.action_space)**depthmax >= 10e8:
            raise ValueError('The tree to be created would be too large, reduce branching factor or depth.')
        
        # here range(1,d+1) is chosen to match dictionary keys properly
        for depth in range(1,depthmax+1):
            self.tree[depth] = []
            
            #loop for adding children and creating a bidirectional reference
            for i, node in enumerate(self.tree[depth-1]):
                c_idx = len(self.tree[depth])
                self.tree[depth] += self.add_child_nodes(i)
                self.tree[depth-1][i].children = list(range(c_idx,c_idx+len(self.action_space))) #reference from parent to children
                
#     def generate_children_tree(moves : tuple):
#         '''
#         This method generates an impartial tree for the next minimax search two moves after the parent tree.
        
#         :param moves (tuple): A tuple of the two last moves as integers.
        
#         :returns (Minimax): A Minimax object with the new parameters ready to receive a call.
#         '''
        
#         if len(moves) == 2:
#             self.starting = self.starting.step(moves[0])
#             self.starting = self.starting.step(moves[1])
#         else:
#             raise ValueError('The provided parameter for moves has incorrect length.')
        
#         #add self.tree manip here
        
#         return self
    
    def __call__(self, player : bool, eval_func=None):
        '''
        The call method for Minimax tree search.
        
        ::
        ::
        
        :returns (list): A list.
        '''
        
        while True:
            depth = 1
            args = self.gather_next_startpoints(depth)
            results = ThreadPool(len(args)).starmap(self.generate_endpoint_and_evaluate,args)
            print(results)
            queue = [sublst for lst in results for sublst in lst]
            self.process_queue(queue)
            
            break
            depth += 1
    
    def gather_next_startpoints(self, depth : int) -> list:
        '''
        :returns (list): A list of tuples for arguments to use in a multithread call of form [(depth,index)]
        '''
        
        args = []
        for i, node in enumerate(self.tree[depth]):
            if not node.visited:
                args += [(depth,i)]
        
        # if no nodes where colleted, assume all were visited and go to the next depth unless at max depth
        if len(args) == 0:
            if depth < self.depthmax:
                return self.gather_next_startpoints(depth+1)
            
        else:
            return args
        
    def generate_endpoint_and_evaluate(self, depth : int, index : int, eval_func=None):
        '''
        ADD
        '''
        
        node = index
        for depth in range(depth, self.depthmax):
            node = self.tree[depth][node].children[0] #chose first child of current node until at leaf
        
        #get the current state's value
        return self.get_value_at(self.depthmax, node, eval_func=eval_func)
        
    def downpropagate(self, depth : int, index : int, value : float):
        '''
        Method for downpropagating the value of a node to all its children.
        Important note: This method must be called on the first child first since it used negative value of parent.
        
        :param depth (int): The depth of the current state in the tree.
        :param index (int): The index of the node for the current state at selected depth.
        :param value (float): The value of the parent state
        '''
        
        #value assignments for the node at depth, index
        self.tree[depth][index].value = -value #use the negative value of the parent
        self.tree[depth][index].proven = True
        self.tree[depth][index].visited = True
        
        children = self.tree[depth][index].children
        
        #if the current depth is not the maximum depth
        if depth < max(list(self.tree.keys())):
            for idx in children:
                self.downpropagate(depth+1, idx, -value)
    
    def get_parent_chain(self, depth : int, index : int, chain=[]) -> list:
        '''
        Method for recursively reconstructing a list of moves to reach the current node in the tree.
        This list includes the move of the node at index at depth and all it's parental moves.
        This enables playing the list down from the starting environment to arrive at the current environment.
        
        :param depth (int): The depth of the current state in the tree.
        :param index (int): The index of the node for the current state at selected depth.
        
        :returns (list): An ordered list of moves to be played from the starting state to arrive at current state.
        '''
        
        chain = deepcopy(chain) #not ideal but fixes error
        chain += [self.tree[depth][index].move]
        
        #recursively get the next element by going down in depth and getting the parental index
        if depth > 1:
            return self.get_parent_chain(depth-1, self.tree[depth][index].parent, chain=chain)
        
        else:
            return chain[::-1]
    
    def process_queue(self, queue):
        '''
        ADD
        '''
        
        for d, idx, var, val in queue:
            if var == 'visited':
                self.tree[d][idx].visited = val
            
            elif var == 'proven':
                self.tree[d][idx].proven = val
            
            elif var == 'value':
                self.tree[d][idx].value = val
    
    def validate_proven(self):
        pass
    
    def get_value_at(self, depth : int, index : int, eval_func=None):
        '''
        Method for getting the value of a state.
        Either uses an evaluation function or the true value if known.
        If no evaluation function is provided this method will assume all states not-yet-terminal to have a value of 0.
        Values for evaluation functions need to be within the range of -1 and 1, true values will be represented by setting the node to proven.
        
        :param depth (int): The depth of the current state in the tree.
        :param index (int): The index of the node for the current state at selected depth.
        :param eval_func (func): Any python object which has a call function which accepts a call with the two parameters observation and player and returns a value between -1 and 1.
        
        :returns (float): A float value representing the value of a specific state.
        '''
        
        env = deepcopy(self.starting)
        
        chain = self.get_parent_chain(depth, index)
        
        queue = []
        
        for i, move in enumerate(chain):
            if not env.terminal:
                env.step(move)
                
                #if the environment is terminal before reaching the end of the move chain, set terminal node to proven and downpropagate
                if env.terminal:
                    value = 1 if env.winner != env.turn else -1 #if current player to move is not winner (meaning the move to this was a winning move), set to 1, else -1
                    
                    #find terminal point as depth or the parent of depth currently at
                    terminal_chain_len = len(chain[i:]) #length is chain after element i
                    #go terminal_chain_len upwards while setting the nodes to visited from depth index
                    idx, d = 0, self.depthmax
                    
                    for _ in range(terminal_chain_len):
#                         self.tree[d][idx].visited = True
                        queue += [(d,idx,'visited',True)]
                        d, idx = d-1, self.tree[d][idx].parent
                    
                    #set the value for the node at terminal
#                     self.tree[d][idx].value = value
                    queue += [(d,idx,'value',value)]
                    queue += [(d,idx,'visited',True)]

                    
                    #do the proven rule for a win leading to the parent being a loss
                    if value == -1:
                        idx = self.tree[d][idx].parent
#                         self.tree[d-1][idx].value = -value
                        queue += [(d-1,idx,'value',-value)]
                        queue += [(d-1,idx,'visited',True)]
                    
                    return queue # we return here to prevent execution of later part.
                
        #after the loop the depth and index are reached and the environment is in the right state
        value = 0
        if eval_func != None:
            value = eval_func(env.observation(), env.turn)
        
#         self.tree[depth][index].value = value
        queue += [(depth,index,'value',value)]
#         self.tree[depth][index].visited = True
        queue += [(depth,index,'visited',True)]
        
        return queue
        
    def add_child_nodes(self, parent_idx : int) -> list:
        '''
        Method for adding all possible children of a parent given an environment.
        This method returns the children nodes, the return value may be used for dictionary entries.
            
        :param parent_idx (int): The index of a parent, used for reference linking from children to parent.
            
        :returns (list): A list of MinimaxNode objects with correct references to parent.
        '''
        
        children = []
        for action in self.action_space:
            children += [MinimaxNode(action,parent_idx)]
            
        return children