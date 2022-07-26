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

def process_image(img):
    '''
    processes image data from observation - normalization of pixel values
    '''
    #todo: maybe remove bottom bar
    #img = img[:-12,:,:]

    #normalization of values
    img = img/255.0

    return img

def sample_trajectories(env, model, buffer, steps=100):
    '''
    ADD
    '''

    observation, info = env.reset(return_info=True)
    observation = process_image(observation)
    terminal = False
    reward = 0

    # while no terminal state is reached we do actions
    for step in range(steps):
        action_dist = model(tf.expand_dims(observation,axis=0))
        action = action_dist.sample()
        observation, reward, terminal, info = env.step(tf.squeeze(action).numpy())
        observation = process_image(observation)

        #storing what's happened in that step, the action distribution might be just calculated from state during training
        buffer['state'].append(observation)
        buffer['action'].append(action)
        buffer['action_dist'].append(action_dist)
        buffer['reward'].append(reward)
        buffer['ret'].append(0) #calculated later

        if terminal:
            break

    return buffer

def calculate_returns(buffer, discount_factor = 0.98):
    # Calculate cumulative discounted rewards for each timestep in buffer
    # TODO: didn't check this code in detail yet

    buffer_len = len(buffer['reward'])
    for outer_idx in range(buffer_len):
        ret = 0
        for inner_idx in range(outer_idx, buffer_len):
            ret += discount_factor ** (inner_idx-outer_idx) * buffer['reward'][inner_idx]
        buffer['ret'][outer_idx] = ret

    return buffer


def policy_update(model, buffer):
    #Todo: Gradient doesn't work like that
    with tf.GradientTape() as tape:
        buffer = calculate_returns(buffer)

        policy_gradient = 0
        for sample_idx in range(len(buffer['reward'])):
            log_prob = model(tf.expand_dims(buffer['state'][sample_idx], axis=0)).log_prob(buffer['action'][sample_idx])
            policy_gradient += -tf.math.multiply(buffer['ret'][sample_idx], log_prob)

        gradient = tape.gradient(policy_gradient, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))
