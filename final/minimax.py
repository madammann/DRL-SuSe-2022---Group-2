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
        
        proven = False
        depth = 1
        
        #loops until all nodes are visited or until confirmed proven result for depth=1
        while not proven:
            #step 1: Gather all unvisited nodes from current depth
            args = self.gather_next_startpoints(depth)
            
            #step 2: traverse down until last depth from each selected node and update the tree with the results (also handles proven case win -> parent loss)
            results = ThreadPool(len(args)).starmap(self.generate_endpoint_and_evaluate,args)
            queue = [sublst for lst in results for sublst in lst]
            self.process_queue(queue)

            #step 3: tests for proven case all children proven -> parent.val = min(children.val)
            self.validate_proven(depth)
            
            # adjust depth and test proven
            depth += 1
            proven = True if all(node.proven for node in self.tree[1]) else False
            # also make sure to stop once depth would exceed maximum
            if depth > self.depthmax:
                proven = True
        
        #next loop to propagate values up the tree via the actual minmax operation (depth here is used as a subtrahend, minus 1 is applied to prevent accessing the -1th element later for one operation)
        for depth in range(self.depthmax-1):
            #loop over all parents of the current step and for each set its value to the minmax based around uneven or even depth based on starting.turn
            for i, parent in enumerate(self.tree[self.depthmax-depth-1]):
                #only change value if the current node is not proven
                if not parent.proven:
                    if self.depthmax-depth % 2 == 0:
                        #the maximum of the children values is the parent value, since player wants to maximize
                        value = max([self.tree[self.depthmax-depth][idx].value for idx in parent.children]) #no need to negate in this place since already done in value assignment
                        self.tree[self.depthmax-depth-1][i].value = value
                    
                    else:
                        #the minimum of the children values is the parent value, since oponent tries to maximize, ergo minimize player
                        value = min([self.tree[self.depthmax-depth][idx].value for idx in parent.children])
                        self.tree[self.depthmax-depth-1][i].value = value
        
        #return a list of values from depth 1 finally to call argmax on for a move choice (since turn is from perspective of player with first move also make sure to invert for other player)
        if player == self.starting.turn:
            return [node.value for node in self.tree[1]]
        
        else:
            return [-node.value for node in self.tree[1]]
        
    def gather_next_startpoints(self, depth : int) -> list:
        '''
        :returns (list): A list of tuples for arguments to use in a multithread call of form [(depth,index)]
        '''
        
        args = []
        for i, node in enumerate(self.tree[depth]):
            if not node.visited:
                args += [(depth,i)]
        
        #if no nodes where colleted, assume all were visited and go to the next depth unless at max depth
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
    
    def validate_proven(self, depth : int):
        '''
        ADD
        '''
        
        for parent_idx, parent_node in enumerate(self.tree[depth-1]):
            proven = [self.tree[depth][child_idx].proven for child_idx in self.tree[depth-1][parent_idx].children]
            
            #if all children are proven losses make minimum value the value of parent node
            if all(proven) and depth-1 > 0:
                values = [self.tree[depth][child_idx].value for child_idx in self.tree[depth-1][parent_idx].children]
                
                if all([val == 1 for val in values]):
                    self.tree[depth-1][parent_idx].value = 1
                    self.tree[depth-1][parent_idx].proven = True
                    
                elif all([val == -1 for val in values]):
                    self.tree[depth-1][parent_idx].value = -1
                    self.tree[depth-1][parent_idx].proven = True
    
    def get_value_at(self, depth : int, index : int, eval_func=None):
        '''
        Method for getting the value of a state.
        Either uses an evaluation function or the true value if known.
        If no evaluation function is provided this method will assume all states not-yet-terminal to have a value of 0.
        Values for evaluation functions need to be within the range of -1 and 1, true values will be represented by setting the node to proven.
        
        :param depth (int): The depth of the current state in the tree.
        :param index (int): The index of the node for the current state at selected depth.
        :param eval_func (func): Any python object which has a call function which accepts a call with the two parameters observation and player and returns a value between -1 and 1.
        
        :returns (list): ADD
        '''
        
        env = deepcopy(self.starting)
        
        chain = self.get_parent_chain(depth, index)
        terminal_depth = None
        
        queue = []
        
        #traverse down and store terminal index using environment
        for i, move in enumerate(chain):
            if not env.terminal:
                env.step(move)
                
            else:
                terminal_depth = i #calculate the depth of the terminal state
                
        #append items to queue based on cases
        #case 1: there was no terminal index
        if terminal_depth == None:
            #call evalutation function if given on environment after traversal
            value = 0
            if eval_func != None:
                value = eval_func(env.observation(), self.starting.turn)
                
            #make last step in queue which has known index and depth a non-proven visited node with said value
            queue += [(depth, index,'visited',True)]
            queue += [(depth, index,'value',value)]
            
            #make all yet unvisited parents of that node visited
            idx = self.tree[depth][index].parent
            d = depth-1
            while not self.tree[d][idx].visited and d >= 1:
                queue += [(d, idx,'visited',True)]
                idx = self.tree[d][idx].parent
                d = d-1
        
        #case 2: there was a terminal index
        else:
            #find value of terminal node
            value = 1 if env.winner == self.starting.turn else -1 #value is based on root node player perspective
            
            #move up the chain from last element until a visited node is reached, if still before terminal index, make proven terminal (also handle proven case of win -> parent loss)
            idx = index
            d = depth
            while not self.tree[d][idx].visited and d >= 1:
                #make node visited
                queue += [(d, idx,'visited',True)]
                
                #if depth lower or equal to terminal depth make proven and set value
                if d >= terminal_depth-1:
                    queue += [(d, idx,'proven',True)]
                    queue += [(d, idx,'value',value)]
                    
#                 #proven rule for terminal wins we handle here by overwriting the parent of terminal depth here even if already visited
#                 if d == terminal_depth:
#                     queue += [(d-1, self.tree[d][idx].parent,'proven',True)]
#                     queue += [(d-1, self.tree[d][idx].parent,'value',value)] #since we use player perspective parent of terminal shares same value, no inversion to negative value
                    
                #increment depth and index
                idx = self.tree[d][idx].parent
                d = d-1
        
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