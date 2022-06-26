import numpy as np
import tensorflow as tf
import gym
from model import *
import time

# we create the Lunar Lander environment with specified parameters
lunar_lander_env = gym.make(
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0
)

def do_episode(model):
    '''
    Function to run a single episode, defined as starting to terminal state, with the model.
    
    :param model (LunarLanderModel): The Deep-Q-Network used as model.
    :param buffer (ExperienceReplayBuffer): The Experience Replay Buffer to which we want to add samples.
    
    :returns (int): return of episode (summed rewards until terminal state is reached)
    '''
    
    observation, info = lunar_lander_env.reset(return_info=True)
    terminal = False
    reward = 0

    # while no terminal state is reached we do actions
    while not terminal:
        lunar_lander_env.render()
        
        past_observation = observation
        
        # we input the observation to the model and chose a discrete action by applying the argmax over the output
        policy = model(tf.expand_dims(observation,axis=0))
        action = int(tf.argmax(policy,axis=1))

        observation, reward, terminal, info = lunar_lander_env.step(action)

model = LunarLanderModel()

# we call the model once to built is's graph properly
model(tf.random.normal((1,8)))

model.load()

for _ in range(10):
    do_episode(model)

lunar_lander_env.close()