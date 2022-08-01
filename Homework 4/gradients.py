import gym
import tensorflow as tf

from model import CarRacingAgent, ValueNetwork


def process_image(img):
    '''
    processes image data from observation - normalization of pixel values
    '''
    #todo: maybe remove bottom bar
    #img = img[:-12,:,:]

    #normalization of values, center around 0
    img = img/127.5-1

    return img

def sample_trajectories(env, model, buffer, steps=5000, render=False):

    observation, info = env.reset(return_info=True)
    observation = process_image(observation)
    terminal = False
    reward = 0
    sum_reward = 0

    # while no terminal state is reached we do actions
    for step in range(steps):
        if render:
            env.render()
        action_dist = model(tf.expand_dims(observation,axis=0))
        action = action_dist.sample()
        observation, reward, terminal, info = env.step(tf.squeeze(action).numpy())
        observation = process_image(observation)
        sum_reward += reward

        #storing what's happened in that step, the action distribution might be just calculated from state during training
        buffer['state'].append(observation)
        buffer['action'].append(action)
        buffer['action_dist'].append(action_dist)
        buffer['reward'].append(reward)
        buffer['ret'].append(None) #return - calculated later
        buffer['advantage'].append(None) #calculated later
        buffer['terminal'].append(terminal)

        if terminal:
            break

    return buffer, sum_reward

def calculate_returns(buffer, discount_factor):
    # Calculate cumulative discounted rewards for each timestep in buffer

    buffer_len = len(buffer['reward'])
    for outer_idx in range(buffer_len):
        ret = 0
        for inner_idx in range(outer_idx, buffer_len):
            ret += discount_factor ** (inner_idx-outer_idx) * buffer['reward'][inner_idx]
        buffer['ret'][outer_idx] = ret

    return buffer

#update method for basic policy gradient
def policy_update(model, buffer, discount_factor):
    with tf.GradientTape() as tape:
        buffer = calculate_returns(buffer, discount_factor)

        loss = 0
        for sample_idx in range(len(buffer['reward'])):
            log_prob = tf.squeeze(model(tf.expand_dims(buffer['state'][sample_idx], axis=0)).log_prob(buffer['action'][sample_idx]))
            loss += -tf.math.multiply(buffer['ret'][sample_idx], log_prob)

        gradient = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))


## The following methods for A2C only

def calculate_advantages(value_net, buffer, discount_factor):

    for idx in range(len(buffer['reward'])):
        if (buffer['terminal'][idx] == True) or (idx >= len(buffer['reward']) - 1):
            advantage = buffer['reward'][idx] - tf.squeeze(value_net(tf.expand_dims(buffer['state'][idx], axis = 0)))
        else:
            advantage = buffer['reward'][idx] * discount_factor * tf.squeeze(value_net(tf.expand_dims(buffer['state'][idx+1], axis = 0))) - tf.squeeze(value_net(tf.expand_dims(buffer['state'][idx], axis = 0)))
        buffer['advantage'][idx] = advantage

    #Normalize advantages and center around zero
    buffer['advantage'] = (buffer['advantage'] - tf.math.reduce_mean(buffer['advantage'])) * 1.0/tf.math.reduce_std(buffer['advantage'])
    return buffer


def calculate_generalized_advantages(value_net, buffer, state_values_old, discount_factor, gae_lambda):
    '''
    state_values_old: state values calculated by critic model before the most recent training
    '''
    buffer_len = len(buffer['reward'])

    #Calculate critic prediction differences over one training iteration
    delta = [None] * buffer_len
    for idx in range(buffer_len):
        if (buffer['terminal'][idx] == True):
            delta[idx] = buffer['reward'][idx] - state_values_old[idx]
        else:
            delta[idx] = buffer['reward'][idx] + discount_factor * tf.squeeze(value_net(tf.expand_dims(buffer['state'][idx], axis = 0))) - state_values_old[idx]

    #calculate generalized advantages
    for idx in reversed(range(buffer_len)):
        if idx >= (buffer_len - 1):
            buffer['advantage'][idx] = delta[idx]
        else:
            buffer['advantage'][idx] = delta[idx] + discount_factor * gae_lambda * buffer['advantage'][idx+1]

    return buffer


def policy_update_A2C_with_GAE(actor_model, critic_model, buffer, discount_factor, gae_lambda):

    #Train Critic
    with tf.GradientTape() as tape:
        buffer = calculate_returns(buffer, discount_factor)

        state_values = []
        loss = 0
        for sample_idx in range(len(buffer['reward'])):
            state_values.append(tf.squeeze(critic_model(tf.expand_dims(buffer['state'][sample_idx], axis = 0))))

        loss = critic_model.loss(buffer['ret'], state_values)
        gradient = tape.gradient(loss, critic_model.trainable_variables)
        critic_model.optimizer.apply_gradients(zip(gradient, critic_model.trainable_variables))

    #Train Actor
    with tf.GradientTape() as tape:
        buffer = calculate_generalized_advantages(critic_model, buffer, state_values, discount_factor, gae_lambda)

        loss = 0
        for sample_idx in range(len(buffer['reward'])):
            log_prob = tf.squeeze(actor_model(tf.expand_dims(buffer['state'][sample_idx], axis=0)).log_prob(buffer['action'][sample_idx]))
            loss += -tf.math.multiply(buffer['advantage'][sample_idx], log_prob)

        gradient = tape.gradient(loss, actor_model.trainable_variables)
        actor_model.optimizer.apply_gradients(zip(gradient, actor_model.trainable_variables))


def policy_update_A2C(actor_model, critic_model, buffer, discount_factor):

    #Train Actor
    with tf.GradientTape() as tape:
        buffer = calculate_advantages(critic_model, buffer, discount_factor)

        loss = 0
        for sample_idx in range(len(buffer['reward'])):
            log_prob = tf.squeeze(actor_model(tf.expand_dims(buffer['state'][sample_idx], axis=0)).log_prob(buffer['action'][sample_idx]))
            loss += -tf.math.multiply(buffer['advantage'][sample_idx], log_prob)

        gradient = tape.gradient(loss, actor_model.trainable_variables)
        actor_model.optimizer.apply_gradients(zip(gradient, actor_model.trainable_variables))

    #Train Critic
    with tf.GradientTape() as tape:

        buffer = calculate_returns(buffer, discount_factor)

        state_values = []
        loss = 0
        for sample_idx in range(len(buffer['reward'])):
            state_values.append(tf.squeeze(critic_model(tf.expand_dims(buffer['state'][sample_idx], axis = 0))))

        loss = critic_model.loss(buffer['ret'], state_values)
        gradient = tape.gradient(loss, critic_model.trainable_variables)
        critic_model.optimizer.apply_gradients(zip(gradient, critic_model.trainable_variables))


