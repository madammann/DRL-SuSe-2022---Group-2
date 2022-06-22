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


def run_random_agent(steps=1000):
    observation, info = lunar_lander_env.reset(return_info=True)

    for step in tqdm(range(steps)):
        lunar_lander_env.render()
        # action = policy(observation)  # User-defined policy function
        observation, reward, terminal, info = lunar_lander_env.step(lunar_lander_env.action_space.sample())

        if terminal:
            observation, info = lunar_lander_env.reset(return_info=True)


def training(model, episodes=1000, update_frequency = 0.1):
    erbuffer = ExperienceReplayBuffer()
    episode_losses = []
    observation, info = lunar_lander_env.reset(return_info=True)
    terminal = False
    step = 0

    for k in tqdm(range(episodes),desc='Episode progress: '):
        while not terminal:
            step += 1
            lunar_lander_env.render()
            # get action via DQN
            output = model.predict(observation)
            action = argmax(output)
            observation_new, reward, terminal, info = lunar_lander_env.step(action)
            #store experience for training
            erbuffer.append(oberservation, action, reward, obersavtion_new, terminal)

            if step % (1/update_frequency):
                #Train model with a batch from Replay Buffer
                model.learn(erbuffer.sampling())
                # store loss somewhere

            observation = observation_new

        # get mean loss and store

#agent = LunarLanderModel()
#training(agent)
run_random_agent()
