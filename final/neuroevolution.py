
import neat
import neat_visualize

import numpy as np
import tensorflow as tf

import environment

'''
Performs neuroevolution to evolve an agent's neural network to play connect 4 using NEAT
training parameters are defined via variable params; NEAT's parameters are to be defined in a separate configuration file
The agent's opponent is playing randomly
'''

#define some parameters
params = {
    'connect_4_grid_size': (6,7), # vertical x horizontal
    'connect_4_rewards': 'default', # reward settings for the environment (it's a little workaround to determine win-rate with), environment's default is 'default'
    'rendering' : False, #Render board during training
    'n_evaluations': 10, #number of games played for each genome
    'training_iterations': 10000, #for how many generations the neuroevolution should run
    'save_progress_every': 500, #how often to save neuroevolution progress
    'continue_from_checkpoint': None, #if None: start neuroevolution anew with config file, otherwise name of file to load progress from
    'neat_configuration': 'neat-config-connect4' #neat-python configuration file with parameters for neuroevolution
}

def evaluate_networks(genomes, config):
    #evaluate population's genomes as ANNs playing against a randomly playing opponent
    global params

    for idx, genome in genomes:
        agent_reward = 0

        #evalute each genome multiple times to get a more consistent evaluation
        for _ in range(params['n_evaluations']):
            #NEAT's internal methods to work with ANNs constructed from genomes
            model = neat.nn.FeedForwardNetwork.create(genome, config)

            # initialize the connect-4 environment
            if params['connect_4_rewards'] == 'default':
                env = environment.ConnectFourEnv(size = params['connect_4_grid_size'], compressed_obs_representation = True)
            else:
                env = environment.ConnectFourEnv(size=params['connect_4_grid_size'], reward_setting=params['connect_4_rewards'], compressed_obs_representation=True)

            reward_sum = np.zeros((2))  # returns for each player respectively
            observation, terminal = env.reset()
            terminal = False

            # while no terminal state is reached we do actions
            while not terminal:

                #play against random opponent
                action = env.get_random_valid_action()
                observation, reward, terminal = env.step(action)  # observation (tuple): grid, turn {whose turn the last move was}
                reward_sum += reward

                if terminal:
                    break

                flat_observation = tf.reshape(observation[0], [-1]).numpy() #Prepare observation for input
                #let network created from genome play a move
                output = model.activate(flat_observation)
                action = np.argmax(output)

                observation, reward, terminal = env.step(action)  # observation (tuple): grid, turn {whose turn the last move was}

                if params['rendering']:
                    #our agent is the 1 in grid, random agent the 2
                    env.render()
                    _ = input("Press Enter for next move")
                    print("")

                reward_sum += reward

            agent_reward += float(reward_sum[0].numpy() / params['n_evaluations'])

        genome.fitness = agent_reward


def run_neuroevolution(config_file, continue_from_checkpoint = None):

    global params

    if continue_from_checkpoint == None:
        #Load configuration from file
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

        #create NEAT population based on configuration
        population = neat.Population(config)

    else:
        #Load configuration from file
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

        #load previously started neuroevolution to continue from there
        population = neat.Checkpointer.restore_checkpoint(continue_from_checkpoint)
        #config = population.config

    # Add a stdout reporter to show progress in the terminal. (Checkpointer doesn't seem to save them)
    stdoutreporter = neat.StdOutReporter(show_species_detail = True)
    population.add_reporter(stdoutreporter)
    statistics = neat.StatisticsReporter()
    population.add_reporter(statistics)

    # helper to save NEAT progress to file (to continue neuroevolution from there)
    checkpointer = neat.Checkpointer(params['save_progress_every'])
    population.add_reporter(checkpointer)

    #Do the neuroevolution
    best_genome = population.run(evaluate_networks, params['training_iterations'])

    #determine win-rate in 100 plays against random agent
    save_params = params
    params['n_evaluations'] = 100
    params['connect_4_rewards'] = {'step': 0, 'win': 1, 'draw': 0, 'loose': 0}
    evaluate_networks([(1, best_genome)], config)
    print(f"Best evolved agent has a winning rate of {best_genome.fitness} against random agent.")
    params = save_params

    #Visualize fitness progress and fittest ANN
    neat_visualize.plot_stats(statistics, view = True)
    neat_visualize.draw_net(config, best_genome, view = True)


run_neuroevolution(params['neat_configuration'], params['continue_from_checkpoint'])

