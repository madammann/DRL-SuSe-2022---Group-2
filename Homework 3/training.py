from model import *
from tqdm import tqdm

import gym

lunar_lander_env = gym.make(
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    #turbulence_power=1.5
)


def run_agent(steps=1000):
    observation, info = lunar_lander_env.reset(return_info=True)

    for step in tqdm(range(steps)):
        lunar_lander_env.render()
        # action = policy(observation)  # User-defined policy function
        # TODO: Use model to get agent action
        observation, reward, terminal, info = lunar_lander_env.step(lunar_lander_env.action_space.sample())

        if terminal:
            observation, info = lunar_lander_env.reset(return_info=True)


def training(model, episodes=1000):
    episode_losses = []
    observation, info = lunar_lander_env.reset(return_info=True)
    terminal = False

    for k in tqdm(range(episodes),desc='Episode progress: '):
        while not terminal:
            lunar_lander_env.render()
            # processing?
            #output = model(observation)
            #action = argmax(output)
            observation, reward, terminal, info = lunar_lander_env.step(action)
            # replay buffer in here?
            # adjust weights
            # store loss somewhere
        # get mean loss and store


run_agent()
