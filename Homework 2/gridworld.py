import numpy as np
import random
from collections import defaultdict
import os

class Tile:
    '''The tile class which contains a move method and necessary attributes for implemented functionalities.'''
    def __init__(self):
        self.obstructed = False
        self.options = [True, True, True, True]
        self.transition_prob = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        self.reward = 0

    def move(self, action):
        '''Calculates a delta for the position of the agent based on choice (0-3 int).'''
        #Todo: check possiblity of delta movement, rework choice system
        if self.options[action]:
            chance = random.random()
            if self.transition_prob[action][0] >= chance:
                return (0,1) # go up
            elif self.transition_prob[action][1] >= chance:
                return (1,0) # go right
            elif self.transition_prob[action][2] >= chance:
                return (0,-1) # go down
            elif self.transition_prob[action][3] >= chance:
                return (-1,0) # go left

    def get_possible(self, choice):
        '''Returns True if a choice (0-3 int) is possible on this tile.'''
        return self.options[choice]

class Gridworld:
    agent_starting_location = (0,0)

    def __init__(self, size = 5, seed = 3):

        self.size = size

        random.seed(seed)
        np.random.seed(seed)

        #generate Gridworld

        rnd_weights = np.array([0.5, 0.2, 0.1, 0])
        rnd_weights /= rnd_weights.sum() #sum of weights should be 1

        #0:free space, 1:wall, 2:target, 3:toxic, 4:wind (, 9:agent)
        self.grid = np.asarray(random.choices(population = (0,1,3,4), weights = rnd_weights, k = size**2)).reshape(size, size)
        self.grid[Gridworld.agent_starting_location] = 0

        #set target location
        self.grid[np.random.randint(size/2,size), np.random.randint(size/2,size)] = 2

        #generating wind direction
        wind = random.randint(0,3)

        '''Creating a transition probability array for wind coming from above, then rotating it to wind direction'''
        wind_prob = np.array([[0.4,0.25,0.1,0.25],[0,0.7,0.3,0],[0,0,1,0],[0,0,0.3,0.7]])
        if wind > 0:
            wind_prob = np.rot90(wind_prob, k=wind)

        #generating tile class objects
        self.tiles = np.empty((size, size), dtype = object)
        for x in range(self.size):
            for y in range(self.size):
                '''Create tile instance for the field'''
                self.tiles[x][y] = Tile()

                '''Case handler for handling moving options due to board edge'''
                options = [True, True, True, True]
                if x+1 >= self.size:
                    options[1] = False
                if x-1 < 0:
                    options[3] = False
                if y+1 >= self.size:
                    options[0] = False
                if y-1 < 0:
                    options[2] = False

                '''Case handler for handling moving options due to obstruction'''
                if options[0]:
                    if self.grid[x][y+1] == 1:
                        options[0] = False
                if options[1]:
                    if self.grid[x+1][y] == 1:
                        options[1] = False
                if options[2]:
                    if self.grid[x][y-1] == 1:
                        options[2] = False
                if options[3]:
                    if self.grid[x-1][y] == 1:
                        options[3] = False

                '''Setting the options for the tile'''
                self.tiles[x][y].options = options

                '''Case handler for all possible types of tiles in the grid'''
                if self.grid[x][y] == 1:
                    self.tiles[x][y].obstructed = True
                elif self.grid[x][y] == 2:
                    self.tiles[x][y].reward = random.randint(1,5)
                elif self.grid[x][y] == 3:
                    self.tiles[x][y].reward = -random.random()
                elif self.grid[x][y] == 4:
                    self.tiles[x][y].transition_prob = wind_prob

    def reset(self):
        pass

    def step(self):
        pass

    def visualize(self, agent):

        #clear screen
        if os.name == 'posix': #Linux, Mac
            os.system('clear')
        else: #Windows
            os.system('cls')

        symbol_map = {0: " ", 1: "\u2588", 2: "t", 3: "\u2622", 9: "a"}

        visualization_grid = np.array(self.grid) #deep copy, as to not change the grid array
        visualization_grid[agent.location[0],agent.location[1]] = 9 #include agent's position

        for x in range(self.size):
            for y in range(self.size):
                print(symbol_map.get(visualization_grid[x, y], "?"), end = "")
            print()
#             print("\n", "-" * self.size * 2, sep = "")

    def get_tile_at(self, coordinates):
        return self.tiles[coordinates[0],coordinates[1]]

    def prove_solvable(self, location=(0,0)):
        #Todo: avoid forever-loop, maybe include tile-checked-flag
        neighbors = [(location[0]+1,location[1]),(location[0]-1,location[1]),(location[0],location[1]+1),(location[0],location[1]-1)]
        if any([self.get_tile_at(coord).reward > 0 for coord in neighbors]):
            return True # returns true if the positive reward is in the neighborhood
        else:
            for neighbor in neighbors:
                if not self.get_tile_at(neighbor).obstructed:
                    self.prove_solvable(location=neighbor)
            return False # returns false if running out of neighbor options

class Agent:
    actions = range(4)

    def __init__(self, world):
        self.location = Gridworld.agent_starting_location
        self.Q_table = np.ones((world.size, world.size, len(Agent.actions))) #only zeros not supported by Agent's action choice yet
        #self.tiles_visited = []
        self.world = world

    def action(self):

        #for now, acion is chosen probabilistically based on Q-values
        current_tile = self.world.get_tile_at(self.location)
        rnd_weights = self.Q_table[self.location[0], self.location[1], :] * current_tile.options
        rnd_weights /= rnd_weights.sum()

        action = random.choices(population = Agent.actions, weights = rnd_weights)[0]

        assert current_tile.get_possible(action) == True

        self.location += np.array(current_tile.move(action))

def SARSA(agent, world, n):
    rewards = []
    for _ in range(n):
        rewards += [agent.action()] #also implement current location
#     rewards += [q_val_at_last_action]
    q_val = np.sum([rewards[i]*np.pow(0.98,i) for i in range(len(rewards))])
    agent.update_q_value()

world = Gridworld()
#world.prove_solvable()
agent = Agent(world)

world.visualize(agent)
#testing agent movement
for timestep in range(20):
    print()
    input("Next action")
    agent.action()
    world.visualize(agent)
