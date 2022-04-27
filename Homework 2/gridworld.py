class Tile:
    def __init__(self):
        self.obstructed = False
        self.options = [True, True, True, True]
        self.transition_prob = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        self.reward = 0
        
    def get_reward(self):
        pass
    
    def move(self, action):
        pass

class Gridworld:
    def __init__(self, tiles):
        self.tiles = tiles
        self.agent_location = (0,0)
        
    def initialize(self, data):
        tiles = None
        return Gridworld(tiles)
    
    def reset(self):
        pass
    
    def step(self):
        pass
    
    def visualize(self):
        pass
    
    def get_tile_at(self, coordinates):
        return self.tiles[coordinates[0],coordinates[1]]
    
    def prove_solvable(self):
        pass
    
class Agent:
    def __init__(self):
        pass
    
    def action(self, options):
        pass
    
def SARSA(n):
    pass
