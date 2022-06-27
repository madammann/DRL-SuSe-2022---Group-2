import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from multiprocessing.pool import ThreadPool
from copy import deepcopy

import tensorflow as tf

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


def do_episode(model):
    '''
    Function to run a single episode, defined as starting to terminal state, with the model.

    :param model (LunarLanderModel): The Deep-Q-Network used as model.

    :returns (tuple): return of episode (summed rewards until terminal state is reached) and buffer queue as list of elements to append in buffer.
    '''
    env = deepcopy(lunar_lander_env)
    buffer_queue = []
    reward_sum = 0

    # initialize environment
    observation, info = env.reset(return_info=True)
    terminal = False

    # while no terminal state is reached we do actions
    while not terminal:
        past_observation = observation

        # we input the observation to the model and chose a discrete action by applying the argmax over the output
        policy = model(tf.expand_dims(observation,axis=0))
        action = int(tf.argmax(policy,axis=1))

        observation, reward, terminal, info = env.step(action)
        buffer_queue += [[past_observation, action, reward, observation, terminal]]
        reward_sum += reward

    return reward_sum, buffer_queue

def train_on_buffer(model, samples, discount_factor = 0.9):
    '''
    Trains the model via single-step Q-learning

    :param model (LunarLanderModel): The Deep-Q-Network used as model.
    :param samples (list of [past_observation, action, reward, observation, terminal]): samples to used to train model
    '''

    with tf.GradientTape() as tape:

        predictions = []
        targets = []

        for datum in samples:
            observation, action, reward, observation2, terminal = datum
            # retrieve q values of observed state
            q_values_pred = model(tf.expand_dims(observation,axis=0))
            q_values = q_values_pred.numpy()

            #compute targeted q value for taken action in observed state via single step q-learning
            if bool(terminal.numpy()):
                q_values[0][int(action)] = reward

            else:
                q_values[0][int(action)] = reward + tf.multiply(discount_factor, tf.reduce_max(q_values_pred))

            predictions.append(q_values_pred)
            targets.append(q_values)

        loss = model.loss(targets, predictions)

        gradient = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))

def training(model, episodes=100, pool_size=10, epochs=100):
    '''
    ADD
    '''

    # we intialize the necessary buffer and environment
    buffer = ExperienceReplayBuffer()
    observation, info = lunar_lander_env.reset(return_info=True)

    # we create lists in which we want to store training result data
    epoch_returns = []

    print('Starting training for ' + str(epochs) + ' epochs...')

    for epoch in range(epochs):
        avg_reward = []
        new_counter = 0

        for episode in tqdm(range(episodes),desc='Progress for epoch ' + str(epoch) + '/' + str(epochs) + ':'):
            # we do n training episodes in multithreading
            results = ThreadPool(pool_size).map(do_episode,[model for _ in range(pool_size)])

            # we collect results and append them appropriately to the buffer
            for i in range(len(results)):
                for j in range(len(results[i])):
                    buffer.append(results[i][1][j])

                avg_reward += [results[i][0]]

            if new_counter > buffer.batch_size:
                train_on_buffer(model, buffer.sample())
                new_counter = 0
            else:
                new_counter += pool_size

        # we take the mean over the epoch and store it
        avg_reward = tf.reduce_mean(avg_reward).numpy()
        print('Epoch ' + str(epoch) + ' finished with an average reward of ' + str(avg_reward) + '.')
        epoch_returns += [avg_reward]

    return epoch_returns

epoch_returns = training(model)
model.save()
visualize_progress(epoch_returns)
lunar_lander_env.close()
