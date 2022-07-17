# imports
import gym
import tensorflow as tf
from model import *

# load environment
car_racing_env = gym.make("CarRacing-v1")

# load model
model = CarRacingAgent()
model(tf.random.normal((10,96,96,3)))
model.load()

# visualization function for n episodes
def do_episode(model):
    '''
    ADD
    '''
    
    observation, info = car_racing_env.reset(return_info=True)
    terminal = False
    reward = 0

    # while no terminal state is reached we do actions
    while not terminal:
        car_racing_env.render()
        
        # we input the observation to the model and chose a discrete action by applying the argmax over the output
        action = model(tf.expand_dims(observation,axis=0))
        observation, reward, terminal, info = car_racing_env.step(action)
        
for _ in range(10):
    do_episode(model)
    
car_racing_env.close()
    
# training stats functions (plots and stuff)
# TODO