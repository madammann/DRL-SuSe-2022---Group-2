import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from copy import deepcopy

from model import ConnectFourModel
from environment import ConnectFourEnv

from datetime import datetime

from tf_agents.replay_buffers import tf_uniform_replay_buffer

def append_training_data(name : str, model : str, epoch : int, average_loss : float, average_return : float, path : str):
    '''
    Function to add the data from training to the training data csv file for later analysis.

    :param name (str): The name used for the training instance the training was run in.
    :param model (str): The name of the model which was trained.
    :param epoch (int): The epoch of the training instance the model was trained in.
    :param average_return (float): The acerage return of the model during the training epoch.
    :param path (str): The path in which the weights of the training are stored.
    '''

    date = str(datetime.now())

    appendix = pd.DataFrame([[name, model, date, epoch, average_loss, average_return, path]],columns=['NAME','MODEL','DATE','EPOCH','AVERAGE LOSS','AVERAGE RETURN', 'PATH'])

    df = None
    try:
        df = pd.read_csv('./training_data.csv',index_col=None)
    except:
        raise FileNotFoundError('Unable to load the training data csv file.')

    df = df.append(appendix)
    df.to_csv('./training_data.csv',index=None)

def update_target_network(model_a, model_b):
    '''
    Function to copy variables from model_q to model_target.
    Both models need to have the same architecture and identical trainable variable space.

    :param model_a (tf.keras.Model subclass): A donor model which will provide the weights to be copied.
    :param model_b (tf.keras.Model subclass): An acceptor model which will receive the weight to overwrite itself with.
    '''

    for target_variable, source_variable in zip(model_b.trainable_variables, model_a.trainable_variables):
        target_variable.assign(source_variable)

def initialize_models(grid_size=(6,7), learning_rate=0.001, path=None):
    '''
    Function to help with initialization of environment, models and general utility.

    :grid_size (tuple): A tuple of shape (M,N) for the Connect-4 grid size to be used.
    :learning_rate (float): A learning rate to be used inside the models' Adam optimizers.
    :path (str): A valid file path to the exact weight.h5 file to be loaded if one should be loaded.

    :returns (tuple): The environment instance, DQN, and target network as tuple (env, model_q, model_target).
    '''

    #initialize the environment
    env = ConnectFourEnv(grid_size)

    #we initialize the models freshly
    model_q = ConnectFourModel(grid_size, learning_rate)
    model_target = ConnectFourModel(grid_size, learning_rate)

    #we initialize the buffer with the correct shape
    data_spec =  (tf.TensorSpec([grid_size[0],grid_size[1],3], tf.float32, 'state'),
              (tf.TensorSpec([], tf.float32, 'action'),
              tf.TensorSpec([], tf.float32, 'reward'),
              tf.TensorSpec([grid_size[0],grid_size[1],3], tf.float32, 'successor_state'),
              tf.TensorSpec([], tf.float32, 'terminal')))
    buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec, batch_size=1, max_length=32000) #buffer size set to 32000 which is 1000 32x batches

    #create the graphs by passing input once
    model_q(tf.random.normal((1, grid_size[0], grid_size[1], 3)))
    model_target(tf.random.normal((1, grid_size[0], grid_size[1], 3)))

    if path != None:
        #try to load previously started training data (model weights)
        try:
            model_q.load(path=path)
            update_target_network(model_q, model_target)

        except FileNotFoundError:
            print(f'Warning: Unable to load weights, assuming model has not been trained before and starting training now.')

    return env, buffer, model_q, model_target

def train_on_buffer(model_q, model_target, samples, discount_factor=0.99):
    '''
    Trains the model via single-step Delayed target Q-learning.

    :param model_q (ConnectFourModel): The Deep-Q-Network used as model.
    :param model_target (ConnectFourModel): The delayed target DQN.
    :param samples (tensorflow.python.data.ops.dataset_ops.MapDataset): Samples from the buffer used to train the model of shape=(batch_size, (state, (action, reward, successor_state, terminal))).
    :param discount_factor (float [0, 1]): The discount factor for future rewards to be used, default is .99, decrease to set bias for sooner rewards.

    :returns loss (): ADD
    '''

    with tf.GradientTape() as tape:
        predictions = []
        targets = []

        #we loop over each element in the sampled batch
        for i, item in enumerate(samples):
            #for ease of use we unpack the element into variables
            state, action, reward, successor_state, terminal = item[0][0], item[0][1][0], item[0][1][1], item[0][1][2], item[0][1][3]

            # retrieve q values of observed state and next state
            q_values_pred = model_q(tf.expand_dims(state,axis=0))
            q_values_to_train = q_values_pred.numpy()

            #compute targeted q value for taken action in observed state via single step q-learning
            if bool(terminal.numpy()):
                q_values_to_train[0][int(action.numpy())] = reward

            else:
                q_values_next_state = model_target(tf.expand_dims(successor_state,axis=0))
                q_values_to_train[0][int(action.numpy())] = reward + tf.multiply(discount_factor, tf.reduce_max(q_values_next_state))

            predictions.append(q_values_pred)
            targets.append(q_values_to_train)

            #due to the nature of samples being infinite we stop after collecting the first 320 (ADD)
            if i >= 32:
                break

        loss = model_q.loss(targets, predictions)

        gradient = tape.gradient(loss, model_q.trainable_variables)
        model_q.optimizer.apply_gradients(zip(gradient, model_q.trainable_variables))

        return loss

def do_episode(env, model, epsilon = 0.1):
    '''
    Function to run a single episode, defined as starting to terminal state, with the model.
    Designed for multithreading hence the return values will include lists of appendices to the buffer.

    :param env (ConnectFourEnv): The environment to copy and use for playing the episode.
    :param model (ConnectFourModel): The Deep-Q-Network used as model.
    :param epsilon (float [0,1]): Exploration probability, the percentage ratio from this will determine random actions taken.

    :returns (tuple): Return of episode (summed rewards until terminal state is reached) and buffer queue as list of elements to append in buffer.
    '''

    env = deepcopy(env)
    buffer_queue = []
    reward_sum = np.zeros((2)) #returns for each player respectively

    #initialize environment
    observation, terminal = env.reset()
    terminal = tf.constant(0, dtype='float32')
    first_move = True #a flag, because we can only train after the second move; optimally, we should have two agents

    #while no terminal state is reached we do actions
    while not bool(int(terminal.numpy())):
        if first_move:
            past_observation = observation

        else:
            past_past_observation = past_observation
            past_observation = observation
            past_action = action

        if np.random.random() < epsilon:
            #choosing exploration: take random action
            action = tf.constant(env.action_space.sample(), dtype='float32')

        else:
            #choosing greedy action: we input the observation to the model and chose a discrete action by applying the argmax over the output
            policy = model(tf.expand_dims(observation, axis=0))
            action = tf.cast(tf.squeeze(tf.argmax(policy, axis=1), axis=0), dtype='float32')

        observation, reward, terminal = env.step(int(action)) #observation (tuple): grid, turn {whose turn the last move was}

        if not first_move:
            if env.turn == 0: #whose turn the last move was
                buffer_queue += [(past_past_observation, (past_action, reward[0], observation, terminal))]

            else:
                buffer_queue += [(tf.reverse(past_past_observation,[2]), (past_action, reward[1], tf.reverse(observation,[2]), terminal))]

        reward_sum += reward
        first_move = False

    return reward_sum, buffer_queue

def training(env, buffer, model_q, model_target, name : str, model_name : str, path : str, episodes=100, epochs=100, update_target_network_every=2, epsilon=0.9, epsilon_decay=0.02, min_epsilon = 0.1):
    '''
    Function for training the DQN using the target model, environment, and experience replay buffer.
    Stores weights and training data every epoch.

    ADD more about multithreading
    ADD more about epsilon decay
    Note that the after hot start of the buffer for each sampling process 10 training batches will be trained on.
    This means a delayed update of 1 means updating after 10x32 batches of training which is one episode.

    Parameters for training data structures:
    :param env (ConnectFourEnv): The environment to copy and use for playing the episode.
    :param buffer (TFUniformReplayBuffer): The buffer object used as a buffer for storing episode data.
    :param model_q (ConnectFourModel):
    :param model_target (ConnectFourModel):

    Parameters for training information storage:
    :param name (str): The name of the training, used mainly to know when and how was trainined.
    :param model_name (str): The name of the model.
    :param path (str): The path in which the weights shall be stored.

    Parameters for training length:
    :param episodes (int [1,]): Number of episodes to do per epoch, due to batch size and multithreading the actual number of episodes is episodes*32.
    :param epochs (int [1,]): Total number of epochs to train until stopping.
    :param update_target_network_every (int [1,episodes]): Integer of after how many sampling procedures with following training on buffer the target network should be updated.

    Parameters for exploration:
    :param epsilon (float [0,1]): The starting value for the exploration factor.
    :param epsilon_decay (float [0,1]): The decay factor for the exploration factor which will be substracter every.
    :param min_epsilon (float [0,1]): The lower bound for epsilon decay.
    '''

    batch_size = 32

    print('Starting training for ' + str(epochs) + ' epochs...')

    #we fill the buffer already with 10x32x(min(7) to max(grid_x*grid_y)) elements depending on sample episode length, this ensures at least way more than 320 elements in the buffer
    for _ in range(10):
        args = [(env, model_q, epsilon) for _ in range(batch_size)]
        results = ThreadPool(batch_size).starmap(do_episode, args)

        buffer_queue = [result[1] for result in results]
        buffer_queue = [sublst for lst in buffer_queue for sublst in lst] #we unnest the list of lists into one big list

        #we append the elements in the buffer queue to the buffer
        for values in buffer_queue:
            batch = tf.nest.map_structure(lambda t: tf.stack([t]),values)
            buffer.add_batch(batch)


    for epoch in range(epochs):
        avg_reward, avg_loss = [], []

        for episode in tqdm(range(episodes),desc='Progress for epoch ' + str(epoch) + '/' + str(epochs) + ':'):
            #we do n training episodes in multithreading
            args = [(env, model_q, epsilon) for _ in range(batch_size)]
            results = ThreadPool(batch_size).starmap(do_episode, args)

            #we collect results for rewards and buffer_queue
            rew_list = [results[0] for result in results]
            avg_reward += [sublst for lst in rew_list for sublst in lst]
            buffer_queue = [result[1] for result in results]
            buffer_queue = [sublst for lst in buffer_queue for sublst in lst] #we unnest the list of lists into one big list

            #we append the elements in the buffer queue to the buffer
            for values in buffer_queue:
                batch = tf.nest.map_structure(lambda t: tf.stack([t]),values)
                buffer.add_batch(batch)

            #we do ten batches of 32 training on buffer per sampling
            for _ in range(10):
                avg_loss += [train_on_buffer(model_q, model_target, buffer.as_dataset(single_deterministic_pass=False))]

            #delayed update of target network
            if episode % update_target_network_every == 0:
                update_target_network(model_q, model_target)

        #reduce exploration probability: (after training, exploration should be zero, though)
        if epsilon > min_epsilon:
            epsilon = epsilon * ((1-epsilon_decay)**batch_size)

        else:
            epsilon = min_epsilon

        # we take the mean over the epoch and store it
        avg_reward = 0.0
        avg_loss = 0.0
        try:
            avg_reward = float(tf.reduce_mean(avg_reward).numpy())
            avg_loss = float(tf.reduce_mean(avg_loss).numpy())
        except:
            pass
        print('Epoch ' + str(epoch) + ' finished with an average reward of ' + str(avg_reward) + '.')

        #save weights of DQN after each epoch and append to the training data file
        model_q.save(path)
        append_training_data(name, model_name, epoch+1, avg_loss, avg_reward, path)

    print('Completed training epochs for selected model.')
