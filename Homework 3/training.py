import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import tensorflow as tf
import tensorflow_datasets as tfds

import gym

from model import *

# we create the Lunar Lander environment with specified parameters
lunar_lander_env = gym.make(
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0
)

# we initialize the model freshly (since this is a toy task weights will not be stored, only the training statistics for one complete run)
model = LunarLanderModel()

def visualize_progress(epoch_returns):
    '''
    Function to show and also save a graph containing the reward over all episodes.
    :param epoch_returns (list): A list of acerage returns for n epochs (defined as 1000 training steps).
    '''
    
    x = np.arange(0, len(epoch_returns))
    y = epoch_returns
    
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots()
    ax.plot(x,y)
    
    ax.set_title('Model performance over epochs')
    ax.set_ylabel('Average return per epoch')
    ax.set_xlabel('Epoch (1000 episodes)')

    fig.savefig('./results.png')
    
    plt.show()


def do_episode(model, buffer):
    '''
    Function to run a single episode, defined as starting to terminal state, with the model.
    
    :param model (LunarLanderModel): The Deep-Q-Network used as model.
    :param buffer (ExperienceReplayBuffer): The Experience Replay Buffer to which we want to add samples.
    
    :returns (tf.Tensor): ADD
    '''
    
    observation, info = lunar_lander_env.reset(return_info=True)
    terminal = False
    
    # while no terminal state is reached we do actions
    while not terminal:
        lunar_lander_env.render()
        
        past_observation = observation
        
        # we input the observation to the model and chose a discrete action by applying the argmax over the output
        policy = model(observation)
        action = tf.argmax(policy,axis=1)
        
        observation, reward, terminal, info = lunar_lander_env.step(action)
        
        buffer.append([past_observation, action, reward, observation])
    
    return reward

def train_on_buffer(model, samples):
    '''
    ADD
    '''
    
    with tf.GradientTape() as tape:
        target = None # calculate q targets
        policy = model(samples)
        
        loss = model.loss(target, policy) # once we have a target or another way to get the loss directly this is done
        
        gradient = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))

def training(model, episodes=10000, epoch_size=1000, update_frequency = 0.5):
    '''
    ADD
    '''
    
    # we intialize the necessary buffer and environment
    buffer = ExperienceReplayBuffer()
    observation, info = lunar_lander_env.reset(return_info=True)
    
    # we create lists in which we want to store training result data
    epoch_returns = []
    
    print('Starting training for ' + str(int(episodes/epoch_size)) + ' epochs...')
    
    for epoch in range(int(episodes/epoch_size)):
        avg_reward = []
        for episode in tqdm(range(episodes),desc='Progress for epoch ' + str(epoch) + ':'):
            # we do the training episode
            reward = do_episode(model, buffer)
        
            # we store the reward for later statistic analysis
            avg_reward += [reward]
            
            if episode % (1/update_frequency):
                lunar_lander_env.close()
            
                train_on_buffer(model, buffer.sample())
        
        lunar_lander_env.close()
        
        # we take the mean over the epoch and store it
        avg_reward = tf.reduce_mean(avg_reward).numpy()
        print('Epoch ' + str(epoch) + ' finished with an average reward of ' + str(avg_reward) + '.')
        epoch_returns += [avg_reward]
    
    return epoch_returns
    
epoch_returns = training(model)
visualize_progress(epoch_returns)
