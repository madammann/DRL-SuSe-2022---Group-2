
import neat
import neat_visualize

import pickle
import numpy as np
import tensorflow as tf

import environment
import agents

'''
Performs neuroevolution to evolve an agent's neural network to play connect 4 using NEAT
training parameters are defined via variable params; NEAT's parameters are to be defined in a separate configuration file
The agent's opponent is playing randomly or is an earlier result of the neuroevolution (kind of self-play light)
'''

#define some parameters
params = {
    'connect_4_grid_size': (6,7), # vertical x horizontal
    'connect_4_rewards': 'default', # reward settings for the environment (it's a little workaround to determine win-rate with), environment's default is 'default'
    'rendering' : False, #Render board during training
    'n_evaluations': 20, #number of games played for each genome
    'training_iterations': 1000, #for how many generations the neuroevolution should run
    'save_progress_every': 100, #how often to save neuroevolution progress
    'continue_from_checkpoint': None, #if None: start neuroevolution anew with config file, otherwise name of file to load progress from
    'statistics_file': 'neat-statistics_reporter.pickle', #save and continue with a separate statistics progress file
    'neat_configuration': 'neat-config-connect4' #neat-python configuration file with parameters for neuroevolution
}

def evaluate_networks(genomes, config):
    #evaluate population's genomes as ANNs playing against a randomly playing opponent
    global params

    #opponents = [agents.NeuroevolutionAgent('Evolved_ANN.pickle'), agents.RandomAgent()]
    opponent = agents.RandomAgent()

    for idx, genome in genomes:
        agent_reward = 0

        #evalute each genome multiple times to get a more consistent evaluation
        for _ in range(params['n_evaluations']):
            #for opponent in opponents:

            #NEAT's internal methods to work with ANNs constructed from genomes
            model = neat.nn.FeedForwardNetwork.create(genome, config)

            # initialize the connect-4 environment
            if params['connect_4_rewards'] == 'default':
                env = environment.ConnectFourEnv(size = params['connect_4_grid_size'])
            else:
                env = environment.ConnectFourEnv(size=params['connect_4_grid_size'], reward_setting=params['connect_4_rewards'])

            reward_sum = np.zeros((2))  # returns for each player respectively
            observation, terminal = env.reset()
            terminal = False

            #randomize who is opening the game
            opening = np.random.randint(0,2)
            #beginner has 2 in grid, second player the 1

            if params['rendering'] == True:
                if opening == 1:
                    print("Neuroevolution agent is opening the game (2 in the grid):")

            # while no terminal state is reached we do actions
            while not terminal:

                if opening == 1:
                    flat_observation = tf.reshape(env.grid2obs_1grid(), [-1]).numpy()  # Prepare observation for input
                    # let network created from genome play a move
                    output = model.activate(flat_observation)
                    action = np.argmax(output)
                else:
                    # play against random opponent
                    action = opponent.select_move(env)

                observation, reward, terminal = env.step(action)  # observation (tuple): grid, turn {whose turn the last move was}
                reward_sum += reward

                if terminal:
                    if params['rendering']:
                        env.render()
                        print("")

                    break

                if opening == 0:
                    flat_observation = tf.reshape(env.grid2obs_1grid(), [-1]).numpy()  # Prepare observation for input
                    # let network created from genome play a move
                    output = model.activate(flat_observation)
                    action = np.argmax(output)
                else:
                    # play against random opponent
                    action = opponent.select_move(env)

                observation, reward, terminal = env.step(action)  # observation (tuple): grid, turn {whose turn the last move was}
                reward_sum += reward

                if params['rendering']:
                    env.render()
                    #_ = input("Press Enter for next move")
                    print("")

            agent_reward += float(reward_sum[1-opening].numpy())

        genome.fitness = agent_reward / params['n_evaluations']


def run_neuroevolution(config_file, continue_from_checkpoint = None):

    global params

    if continue_from_checkpoint == None:
        #Load configuration from file
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

        #create NEAT population based on configuration
        population = neat.Population(config)
        statistics = neat.StatisticsReporter()
    else:
        #load previously started neuroevolution to continue from there
        population = neat.Checkpointer.restore_checkpoint(continue_from_checkpoint)
        config = population.config

        #population.species.species_set_config.compatibility_threshold = 0.8

        try:
            statistics_storage = open(params['statistics_file'], 'rb')
            statistics = pickle.load(statistics_storage)
            print("Continuing statistics")
        except:
            statistics = neat.StatisticsReporter()

    population.add_reporter(statistics)

    # Add a stdout reporter to show progress in the terminal. (Checkpointer doesn't seem to save them)
    stdoutreporter = neat.StdOutReporter(show_species_detail = True)
    population.add_reporter(stdoutreporter)

    # helper to save NEAT progress to file (to later continue neuroevolution from there)
    checkpointer = neat.Checkpointer(params['save_progress_every'])
    population.add_reporter(checkpointer)

    #Do the neuroevolution
    best_genome = population.run(evaluate_networks, params['training_iterations'])

    # save best genome to file via pickle
    genome_storage = open('Evolved_ANN.pickle', 'wb')
    pickle.dump((best_genome, population.config), genome_storage)
    genome_storage.close()

    # save statistics reporter (checkpointer doesn't do that)
    statistics_storage = open(params['statistics_file'], 'wb')
    pickle.dump(statistics, statistics_storage)
    genome_storage.close()

    #determine win-rate of best evolved agent network in 100 games
    save_params = params
    params['n_evaluations'] = 100
    params['connect_4_rewards'] = {'step': 0, 'win': 1, 'draw': 0, 'loose': 0}
    #params['rendering'] = False
    evaluate_networks([(1, best_genome)], config)
    print("")
    print(f" Best evolved agent has a winning rate of {best_genome.fitness}.")
    params = save_params

    #Visualize fitness progress, fittest ANN and speciation
    neat_visualize.plot_stats(statistics, view = True)
    neat_visualize.draw_net(config, best_genome, view = True)
    neat_visualize.plot_species(statistics, view = True)


run_neuroevolution(params['neat_configuration'], params['continue_from_checkpoint'])

