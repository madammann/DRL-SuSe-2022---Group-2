# imports
import tensorflow as tf
import os
import gym
from gym.wrappers import GrayScaleObservation
import pandas as pd
import copy
import numpy as np

from datetime import datetime, timedelta
from model import CarRacingAgent, ValueNetwork
from gradients import *
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

# load environment
car_racing_env = GrayScaleObservation(gym.make("CarRacing-v1"), keep_dim=True)

# define epoch length
epochs = 1000
episodes_per_epoch = 20

environment_steps = 5000 #how long the agent will act in the environment each episode if no terminal state is reached

# define hyperparameters
learning_rate = 0.001

discount_factor = 0.98 # gamma
gae_lambda = 0.97 #lambda for generalized advantage estimation

# define data storage functions

def epoch_results(data : dict) -> str:
    '''
    ADD
    '''
    
    data['TIME'] = str(datetime.now())
    data['LOSS'] = float(-2893.1)
    data['DURATION'] = str(timedelta(hours=5))
    data['BEST'] = float(-2893.1)
    data['WORST'] = float(-2893.1)
    
    data = [[data['TIME'], data['LOSS'], data['DURATION'], data['BEST'], data['WORST']]]
    
    try:
        df = pd.read_csv('./train_data.csv')
        appendix = pd.DataFrame(data, columns=df.columns)
        df = df.append(appendix)
    
    except Exeption as e:
        print('Failed to read train_data.csv')
        
    finally:
        df.to_csv('train_data.csv', index=False)
        return f'Epoch results were stored in train_data.csv.'

# define training functions

# initialization
duration = 10 #TODO: make duration a parameter

# For testing, we should avoid manually inputting parameters
"""
while duration == 0:
    print(f'Please select a training duration in hours after which training shall be stopped at earliest convenience: ')
    duration = int(input())
    
    if type(duration) != int or duration <= 0:
        duration = 0
        print('\n')
        print(f'Invalid duration, please select a positive integer in hours.')
        print('\n')
"""
starttime = datetime.now()
endtime = datetime.now() + timedelta(hours=duration)

print(f'Beginning training from current epoch milestone if present...')

current_epoch = 0
try:
    df = pd.read_csv('./train_data.csv')
    current_epoch = len(df)
    
except FileNotFoundError:
    print(f'Warning: Unable to load train_data.csv, assuming current epoch is initial epoch.')

# model initialization
agent = CarRacingAgent(learning_rate)
value_net = ValueNetwork(learning_rate)
agent(tf.random.normal((10,96,96,1))) # create the graph by passing input once
value_net(tf.random.normal((10,96,96,1)))

try:
    agent.load()
    value_net.load()
    
except FileNotFoundError:
    print(f'Warning: Unable to load weights, assuming model has not been trained before and starting training now.')

# training loop

for epoch in range(current_epoch, epochs):
    # creating data variables
    epoch_start = datetime.now()
    avg_loss = []
    reward_sums = []
    best, worst = None, None

    render = True #Render first episode of each epoch
    # episode loop
    for _ in tqdm(range(episodes_per_epoch), desc=f'Running epoch {epoch}: '):
        #store trajectories in buffer
        buffer = dict(state = [], action = [], action_dist = [], reward = [], ret = [], terminal = [], advantage = [])
        # sample trajectories for 32 multithreaded episodes with each up to a maximum of 100 steps
        #TODO: include multithreading again - I suspect that deepcopying the environment does not work (track attribute not found)
        """
        args = [(copy.deepcopy(car_racing_env),agent,step_len) for _ in range(32)]
        results = ThreadPool(6).starmap(sample_trajectories,args)
        """
        buffer, sum_reward = sample_trajectories(car_racing_env, agent, buffer, environment_steps, render = render)
        reward_sums.append(sum_reward)
        render = False

        #policy_update(agent, buffer, discount_factor) # basic policy gradient
        #policy_update_A2C(agent, value_net, buffer, discount_factor) # A2C
        policy_update_A2C_with_GAE(agent, value_net, buffer, discount_factor, gae_lambda) # A2C with GAE

        # TODO get stats
    
    # generate epoch data
    epoch_end = datetime.now()
    epoch_duration = epoch_end - epoch_start
    
    data = {'TIME' : str(epoch_end),
            #'LOSS' : float(tf.reduce_mean(avg_loss).numpy()),
            'DURATION' : str(epoch_duration),
            #'BEST' : float(best.numpy()),
            #'WORST' : float(worst.numpy())
            }
    
    # store epoch result data
    epoch_results(data)

    print("Average epoch reward:",np.mean(reward_sums))
    
    # store model weights and rename older weights
    try:
        os.rename('./weights.h5', f'./old_weights_{epoch-1}.h5')
        os.rename('./val_weights.h5', f'./old_val_weights_{epoch-1}.h5')
    
    except:
        print(f'Warning: Either an older model weights.h5 did not exist or renaming it was unsuccessful.')
    
    agent.save()
    value_net.save()
    
    # test if continue or break loop
    if (endtime - datetime.now()).seconds <= 0:
        break

print(f'Scheduled a shutdown in 5mins that can be stopped with cmd("shutdown /a") if necessary')
os.system("shutdown -s -t 300") # schedule a shutdown that can be stopped with cmd("shutdown /a") if necessary.