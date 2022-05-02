import numpy as np
import random
from collections import defaultdict
import os
from tqdm import tqdm

class Tile:
    '''The tile class which contains a move method and necessary attributes for implemented functionalities.'''
    def __init__(self):
        self.obstructed = False
        self.options = [True, True, True, True]
        self.transition_prob = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        self.reward = 0

    def move(self, action):
        '''Calculates a delta for the position of the agent based on choice (0-3 int).'''
        if self.options[action]:
            chance = random.random()
            if self.transition_prob[action][0] >= chance:
                return (0,1) # go up
            elif np.sum(self.transition_prob[action][:2]) >= chance:
                return (1,0) # go right
            elif np.sum(self.transition_prob[action][:3]) >= chance:
                return (0,-1) # go down
            elif np.sum(self.transition_prob[action][:4]) >= chance:
                return (-1,0) # go left

class Gridworld:
    def __init__(self, size = 5, seed = 3):

        self.size = size
        self.agent_location = (0,0)
        random.seed(seed)
        np.random.seed(seed)

        '''Generating Gridworld with weights for tile proportions'''
        rnd_weights = np.array([0.5, 0.2, 0.1, 0])
        rnd_weights /= rnd_weights.sum() #sum of weights should be 1

        #0:free space, 1:wall, 2:target, 3:toxic, 4:wind
        self.grid = np.asarray(random.choices(population = (0,1,3,4), weights = rnd_weights, k = size**2)).reshape(size, size)
        self.grid[self.agent_location] = 0

        #set target location
        self.grid[np.random.randint(size/2,size), np.random.randint(size/2,size)] = 2

        '''Creating a transition probability array for wind coming from above, then rotating it randomly to match a specific direction'''
        wind_prob = np.array([[0.4,0.25,0.1,0.25],[0,0.7,0.3,0],[0,0,1,0],[0,0,0.3,0.7]])
        wind_prob = np.rot90(wind_prob, k=random.randint(1,4))

        '''Generating tile class objects'''
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
        self.agent_location = (0,0)

    def step(self, action):
        self.agent_location += np.array(self.get_tile_at(self.agent_location).move(action))
        terminal = self.grid[self.agent_location[0]][self.agent_location[1]] == 2
        return self.agent_location, self.get_tile_at(self.agent_location).reward, terminal

    def visualize(self, agent):
        symbol_map = {0: " ", 1: "\u2588", 2: "t", 3: "\u2622", 9: "a"}

        visualization_grid = np.array(self.grid) #deep copy, as to not change the grid array
        visualization_grid[self.agent_location[0]][self.agent_location[1]] = 9 #include agent's position

        for x in range(self.size):
            for y in range(self.size):
                print(symbol_map.get(visualization_grid[x, y], "?"), end = "")
            print()

    def get_tile_at(self, coordinates):
        return self.tiles[coordinates[0],coordinates[1]]

    def prove_solvable(self, location=(0,0), visited=[]):
        neighbors = [(location[0]+1,location[1]),(location[0]-1,location[1]),(location[0],location[1]+1),(location[0],location[1]-1)]
        to_check = []

        for neighbor in neighbors:
            if neighbor in visited:
                continue
            else:
                if neighbor[0] < 0 or neighbor[0] >= self.size or neighbor[1] < 0 or neighbor[1] >= self.size: #check whether neighbor location is inside gridworld border coordinates
                    continue
            to_check.append(neighbor)

        visited.extend(to_check)

        if any([self.get_tile_at(coord).reward > 0 for coord in to_check]):
            return True, visited # returns true if the positive reward is in the neighborhood
        else:
            for neighbor in to_check:
                if not self.get_tile_at(neighbor).obstructed:
                    solvable, visited = self.prove_solvable(location=neighbor,visited=visited)
                    if solvable: #one True return [0] in recursion is enough to have a solvable gridworld
                        return True, visited
            return False, visited # returns false if running out of neighbor options

class Agent:
    def __init__(self, world):
        self.Q_table = np.ones((world.size, world.size, 4)) #4 possible actions (max)
        self.world = world
        self.reward = 0
        self.terminal = False

    def action(self, epsilon=0.1):
        if not self.terminal:
            '''Action is chosen probabilistically based on Q-values'''
            current_tile = self.world.get_tile_at(self.world.agent_location)
            options = np.multiply(np.array(current_tile.options,dtype='int32'),np.array([1,2,3,4]))
            options = options[options != 0] - 1 #minus 1 for converting to index
            action = None

            if random.random() < epsilon:
                action = random.choice(options)
            else:
                q_values = self.Q_table[self.world.agent_location[0], self.world.agent_location[1], :]
                action = [element for element in np.argsort(q_values)[::-1] if element in options][0]

            state, reward, terminal = self.world.step(action)

            self.reward = reward
            self.location = state
            self.terminal = terminal

            return action

    def reset(self):
        self.world.agent_location = (0,0)
        self.reward = 0
        self.terminal = False
        
    def update_q_value(self, location, value):
        self.Q_table[location] = value
        
def SARSA(agent, world, n=10, gamma=0.98, alpha=0.1):
    if type(n) == int:
        if not n > 0:
            raise ValueError('The value of n must not be 0 or less.')
    '''Creating empty lists for storing episode data'''
    states = [(0,0)] #note that states is longer than actions and rewards with the N-th element having no taken action and reward.
    actions = []
    rewards = []

    while not agent.terminal:
        '''Do an action and update the lists'''
        actions += [agent.action()]
        states += [(agent.location[0],agent.location[1])]
        rewards += [agent.reward]
        
        '''Update the Q-table as soon as n is exceeded by actions or the terminal state is reached'''
        if n > len(actions) or agent.terminal:
            if not agent.terminal:
                '''Retrieving location for the update and calculating new q value'''
                location = (states[0][0],states[0][1],actions[0]) #the location for the update which is the first element of states and actions always
                q_value = agent.Q_table[location] #get current q-value
                
                end_location = (states[-1][0],states[-1][1],actions[-1])
                reward_chain = rewards[0:n-1]+[agent.Q_table[end_location]] #creating the reward chain for n steps with the last step as q-value
                
                q_delta = alpha*(np.sum([reward_chain[i]*np.power(gamma,i) for i in range(len(reward_chain))]) - q_value)
                
                agent.update_q_value(location, q_value+q_delta) #updating the value in the Q-table
                
                '''Deleting the first element since the update has been done'''
                states = states[1:]
                actions = actions[1:]
                rewards = rewards[1:]
            else:
                '''Iterate over remaining non-terminal elements and do all updates'''
                while len(actions) > 0:
                    '''Retrieving location for the update and calculating new q value'''
                    location = (states[0][0],states[0][1],actions[0]) #the location for the update which is the first element of states and actions always
                    q_value = agent.Q_table[location] #get current q-value
                
                    q_delta = alpha*(np.sum([rewards[i]*np.power(gamma,i) for i in range(len(rewards))]))
                
                    agent.update_q_value(location, q_value+q_delta) #updating the value in the Q-table
                
                    '''Deleting the first element since the update has been done'''
                    states = states[1:]
                    actions = actions[1:]
                    rewards = rewards[1:]

'''Create solvable gridworld'''
world = None
for world_seed in range(3, 100):
    world = Gridworld(seed = world_seed)
    if world.prove_solvable()[0] == True:
        break
        
agent = Agent(world)

'''Visualizing the Gridworld once to show '''
print('======================')
print('The Gridworld setup:')
print('======================')
print('\n')
world.visualize(agent)

'''Doing training with SARSA setup'''
print('\n')
print('======================')
print('SARSA:')
print('======================')
print('\n')

value = input('Please select a value for n which is a number greater than 0: ')
while not value.isnumeric():
    print('Value not allowed, please select another value for n!')
    print('\n')
    value = input('Please select a value for n which is a number greater than 0: ')
        
for episode in tqdm(range(1000),desc='Doing SARSA Episodes'):
    SARSA(agent, world, n=int(value))
    agent.reset()
    world.reset()
    
print('\n')
print('======================')
print('Q-Table (x,y,action):')
print('======================')
print('\n')
print(agent.Q_table)

input('Press any key to continue with the solution...')

'''Visualizing the final policy actions'''
print('\n')
print('======================')
print('Gridworld solution:')
print('======================')
print('\n')

while not agent.terminal:
    agent.action(epsilon=0)
    world.visualize(agent)
    print('\n')
    
