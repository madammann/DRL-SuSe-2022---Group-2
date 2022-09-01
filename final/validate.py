import numpy as np
import tensorflow as tf
from model import *
import time
import sys
import environment

#simple script to run and observe the (trained) model in a self-play match of connect-4

#define some parameters (others are parameters of function do_episode)
params = {
    'connect_4_grid_size': (6,7) # vertical x horizontal
}

def do_episode(model, epsilon = 0):
    '''
    Function to run a single episode, defined as starting to terminal state, with the model.

    :param model: The Deep-Q-Network used as model.
    :param epsilon (float [0,1]): exploration probability

    :returns (tuple): return of episode (summed rewards until terminal state is reached) and buffer queue as list of elements to append in buffer.
    '''
    env = connect4_environment
    reward_sum = np.zeros((2)) #returns for each player respectively

    # initialize environment
    observation, terminal = env.reset()
    terminal = False
    first_move = True #a flag, because we can only train after the second move; optimally, we should have two agents

    # while no terminal state is reached we do actions
    while not terminal:

        if np.random.random()<epsilon:
            #choosing exploration: take random action
            action = env.action_space.sample()
        else:
            #choosing greedy action: we input the observation to the model and chose a discrete action by applying the argmax over the output
            policy = model(tf.expand_dims(observation[0],axis=0))
            action = int(tf.argmax(policy,axis=1))

        observation, reward, terminal = env.step(action) #observation (tuple): grid, turn {whose turn the last move was}
        env.render()

        _ = input("Press Enter for next move")
        print("")

        reward_sum += reward
        first_move = False

    return reward_sum


def initialize_model(params):
    # initialize the environment
    connect4_environment = environment.ConnectFourEnv(size=params['connect_4_grid_size'])

    # we initialize the model freshly
    model_q = ConnectFourModel(params['connect_4_grid_size'], 0)

    # create the graphs by passing input once
    model_input_shape = (1, params['connect_4_grid_size'][0], params['connect_4_grid_size'][1], 2)

    model_q(tf.random.normal(model_input_shape))

    # try to load previously started training data (model weights)
    try:
        model_q.load()

    except FileNotFoundError:
        print(f'Error: Unable to load weights!')
        sys.exit(1)

    return connect4_environment, model_q


connect4_environment, model_q = initialize_model(params)

#for _ in range(10):
do_episode(model_q)
