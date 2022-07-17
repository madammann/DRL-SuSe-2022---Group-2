import gym
import tensorflow as tf

from model import CarRacingAgent, ValueNetwork

class PolicyGradient():

    def __init__(self, agent, val_network, env):

        self.env = env
        self.num_iterations = None
        self.batch_size = None
        self.observation_space = None
        self.action_space = None
        self.gamma = None
        self.policy_net = agent
        self.value_net = val_network
        

def estimate_step_len():
    return 10 #TODO: replace dummy value

def sample_trajectories(env, model, steps=100):
    '''
    ADD
    '''

    observation, info = env.reset(return_info=True)
    terminal = False
    reward = 0
    trajectories = []

    # while no terminal state is reached we do actions
    for step in range(steps):
        action = model(tf.expand_dims(observation,axis=0))
        observation, reward, terminal, info = env.step(action)

        if terminal:
            break

    return trajectories

def policy_update(trajectories, model, loss, optimizer):
    with tf.GradientTape() as tape:
        loss = loss(targets, trajectories)

        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
