import gym

from model import CarRacingAgent

def estimate_step_len():
    pass

def sample_trajectories(env, model, steps=100):
    '''
    ADD
    '''

    observation, info = car_racing_env.reset(return_info=True)
    terminal = False
    reward = 0
    trajectories = []

    # while no terminal state is reached we do actions
    for step in range(steps):
        action = model(tf.expand_dims(observation,axis=0))
        observation, reward, terminal, info = car_racing_env.step(action)

        if terminal:
            break

    return trajectories

def policy_update(trajectories, model, loss, optimizer):
    with tf.GradientTape() as tape:
        loss = loss(targets, trajectories)

        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
