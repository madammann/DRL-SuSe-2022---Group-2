import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from functools import partial
from multiprocessing.pool import ThreadPool
from copy import deepcopy

import tensorflow as tf

from model import *

import environment


# initialize the environment
connect4_environment = environment.ConnectFourEnv()

# we initialize the model freshly (since this is a toy task weights will not be stored, only the training statistics for one complete run)
model_q = ConnectFourModel()
model_target = ConnectFourModel()


def train_on_buffer(model_q, model_target, samples, discount_factor = 0.99):
    '''
    Trains the model via single-step Delayed target Q-learning

    :param model_q (LunarLanderModel): The Deep-Q-Network used as model.
    :param model_target (LunarLanderModel): The delayed target DQN
    :param samples (list of [past_observation, action, reward, observation, terminal]): samples to used to train model
    :param discount_factor (float [0, 1]): gamma, decrease to set bias for sooner rewards
    '''

    with tf.GradientTape() as tape:

        predictions = []
        targets = []

        for datum in samples:
            observation, action, reward, observation2, terminal = datum
            # retrieve q values of observed state and next state
            q_values_pred = model_q(tf.expand_dims(observation,axis=0))
            q_values_to_train = q_values_pred.numpy()
            q_values_next_state = model_target(tf.expand_dims(observation2,axis=0))

            #compute targeted q value for taken action in observed state via single step q-learning
            if bool(terminal.numpy()):
                q_values_to_train[0][int(action)] = reward

            else:
                q_values_to_train[0][int(action)] = reward + tf.multiply(discount_factor, tf.reduce_max(q_values_next_state))

            predictions.append(q_values_pred)
            targets.append(q_values_to_train)

        loss = model_q.loss(targets, predictions)

        gradient = tape.gradient(loss, model_q.trainable_variables)
        model_q.optimizer.apply_gradients(zip(gradient, model_q.trainable_variables))


def do_episode(model, epsilon = 0.1):
    '''
    Function to run a single episode, defined as starting to terminal state, with the model.

    :param model (LunarLanderModel): The Deep-Q-Network used as model.
    :param epsilon (float [0,1]): exploration probability

    :returns (tuple): return of episode (summed rewards until terminal state is reached) and buffer queue as list of elements to append in buffer.
    '''
    env = deepcopy(connect4_environment)
    buffer_queue = []
    reward_sum = np.zeros((2)) #returns for each player respectively

    # initialize environment
    observation, terminal = env.reset()
    terminal = False
    first_move = True #a flag, because we can only train after the second move; optimally, we should have two agents

    # while no terminal state is reached we do actions
    while not terminal:
        if first_move:
            past_observation = observation[0]
        else:
            past_past_observation = past_observation
            past_observation = observation[0]
            past_action = action

        if np.random.random()<epsilon:
            #choosing exploration: take random action
            action = env.action_space.sample()
        else:
            #choosing greedy action: we input the observation to the model and chose a discrete action by applying the argmax over the output
            policy = model(tf.expand_dims(observation[0],axis=0))
            action = int(tf.argmax(policy,axis=1))

        observation, reward, terminal = env.step(action) #observation (tuple): grid, turn {whose turn the last move was}

        if not first_move:
            #TODO: Didn't check the reward assignment yet, could be that they need to be flipped in the buffer_queue
            if observation[1]==0: #whose turn the last move was
                buffer_queue += [[past_past_observation, past_action, reward[0], observation[0], terminal]]
            else:
                buffer_queue += [[tf.reverse(past_past_observation,[2]), past_action, reward[1], tf.reverse(observation[0],[2]), terminal]]

        reward_sum += reward

        first_move = False


    return reward_sum, buffer_queue

def update_target_network(model_q, model_target):
    #copy weights from model_q to model_target
    for target_variable, source_variable in zip(model_target.trainable_variables, model_q.trainable_variables):
        target_variable.assign(source_variable)

def training(model_q, model_target, episodes=100, pool_size=10, epochs=100, update_target_network_every=20, epsilon=0.9, epsilon_decay=0.02):

    # we initialize the necessary buffer and environment
    buffer = ExperienceReplayBuffer(size=100000, batch_size=64)

    # we create lists in which we want to store training result data
    epoch_returns = []

    print('Starting training for ' + str(epochs) + ' epochs...')

    for epoch in range(epochs):
        avg_reward = []
        new_counter = 0

        for episode in tqdm(range(episodes),desc='Progress for epoch ' + str(epoch) + '/' + str(epochs) + ':'):
            # we do n training episodes in multithreading
            results = ThreadPool(pool_size).map(partial(do_episode, epsilon=epsilon),[model_q for _ in range(pool_size)])#, epsilon)

            # we collect results and append them appropriately to the buffer
            for i in range(len(results)):
                for j in range(len(results[i][1])):
                    buffer.append(results[i][1][j])

                avg_reward += [results[i][0]]

            new_counter += pool_size * 4
            if new_counter > buffer.batch_size:
                train_on_buffer(model_q, model_target, buffer.sample())
                new_counter = 0

            # Delayed update of target network
            if episode*pool_size % update_target_network_every == 0:
                update_target_network(model_q, model_target)


        #Reduce exploration probability:
        epsilon = epsilon * ((1-epsilon_decay)**pool_size)

        # we take the mean over the epoch and store it
        avg_reward = tf.reduce_mean(avg_reward).numpy()
        print('Epoch ' + str(epoch) + ' finished with an average reward of ' + str(avg_reward) + '.')
        epoch_returns += [avg_reward]

        #save weights of DQN after each epoch
        model_q.save()

    return epoch_returns

training(model_q, model_target)
