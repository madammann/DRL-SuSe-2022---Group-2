from model import *
from tqdm import tqdm

import gym

lunar_lander_env = gym.make(
    "LunarMander-v2",
    continuous: bool = False,
    gravity: float = -10.0,
    enable_wind: bool = False,
    wind_power: float = 15.0,
    turbulance_power: float = 1.5
)

episode_losses = []

# # PSEUDO FOR NOW!
# def training(model, episodes=1000):
#     for k in tqdm(range(episodes),desc='Episode progress: '):
#         env = None # create env in its starting state
#         while not env.terminal:
#             observation = env.observation
#             # processing?
#             output = model(observation)
#             # processing? (argmax)
#             env.action(output)
#             #replay buffer in here?
#             #adjust weights
#             #store loss somewhere
#         # get mean loss and store