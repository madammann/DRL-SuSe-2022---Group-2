# imports
import gym
from gym.wrappers import GrayScaleObservation
import tensorflow as tf
from model import *
from gradients import process_image

# load environment
env = GrayScaleObservation(gym.make("CarRacing-v1"), keep_dim=True)

# load model
model = CarRacingAgent(learning_rate = 0)
model(tf.random.normal((10,96,96,1)))
model.load()

# visualization function: run episode until terminal state is reached
def do_episode(model):
    '''
    ADD
    '''
    
    observation, info = env.reset(return_info=True)
    observation = process_image(observation)
    terminal = False
    reward = 0

    # while no terminal state is reached we do actions
    while not terminal:
        env.render()
        
        action_dist = model(tf.expand_dims(observation,axis=0))
        action = action_dist.sample()
        observation, reward, terminal, info = env.step(tf.squeeze(action).numpy())
        observation = process_image(observation)
        
for _ in range(1):
    do_episode(model)
    
env.close()
    
# training stats functions (plots and stuff)
# TODO