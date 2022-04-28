import numpy as np
import random

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
    agent_starting_location = (0,0)

    def __init__(self):
        self.initialize()
        self.reset()

    def initialize(self, size = 5, seed = 3):

        random.seed(seed)
        np.random.seed(seed)

        #generate Gridworld

        rnd_weights = np.array([0.5, 0.2, 0.1, 0])
        rnd_weights /= rnd_weights.sum() #sum of weights should be 1

        #0:free space, 1:wall, 2:target, 3:toxic, 4:wind
        self.grid = np.asarray(random.choices(population = (0,1,3,4), weights = rnd_weights, k = size**2)).reshape(size, size)
        self.grid[Gridworld.agent_starting_location] = 0

        #set target locarion
        self.grid[np.random.randint(size/2,size), np.random.randint(size/2,size)] = 2


        tiles = np.empty((size, size), dtype = object)
        tiles[:,:] = Tile()


    def reset(self):
        self.agent_location = Gridworld.agent_starting_location

    def step(self):
        pass

    def visualize(self):
        print(self.grid)

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


world = Gridworld()
world.visualize()
