import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime

from model import ConnectFourModel
from environment import ConnectFourEnv
from training import training, initialize_models

from tf_agents.replay_buffers import tf_uniform_replay_buffer

import argparse

parser = argparse.ArgumentParser(description='Script for running the training and evaluation of models for this project.')

#parser arguments to be used
parser.add_argument('--primer', default='False') #whether to run a primer training meaning all other arguments are obsolete
parser.add_argument('--training', default='True') #whether training or evaluation shall be run, if evaluation then statistics will be gathered instead
parser.add_argument('--model') #the model to be used, has to be specified in order to run this script
parser.add_argument('--epochs', default='100') #the number of 100 episode epochs to run
parser.add_argument('--path', default='False') #has to be specified if training mode is run and no primer, will be the weights path
parser.add_argument('--pretrained', default='False') #path if training is to be resumed from weights has to match path

args = parser.parse_args()

def create_training_tag(args):
    '''
    ADD
    '''
    
    date = datetime.now()
    date = f'{date.year}{date.month}{date.day}{date.hour}{args.model}{args.epochs}'
    
    return date
        

def run_primer():
    '''
    Runs if selected in the bash command for this script.
    Will run 10 epochs with 100 episodes each for every model to be trained for training evaluation purposes.
    '''
    
    names = ["primer_6x7","primer_8x9","primer_10x11","primer_12x13"]
    model_names = ["dqn_6x7","dqn_8x9","dqn_10x11","dqn_12x13"]
    grid_sizes = [(6,7),(8,9),(10,11),(12,13)]
    paths = ["./weighs/primers/p6by7.h5","./weighs/primers/p8by9.h5","./weighs/primers/p10by11.h5","./weighs/primers/p12by13.h5"]
    epochs = 10
    episodes = 100
    
    for i in range(len(names)):
        env, buffer, model_q, model_target = initialize_models(grid_sizes[i], path=None)
        training(env, buffer, model_q, model_target, names[i], model_names[i], paths[i], episodes=episodes, epochs=epochs)

if __name__ == "__main__":
    '''
    ADD
    '''
    
    #if not pretrained path is None, also raises error for missing path argument if required
    path = None
    if not args.pretrained == 'True':
        path = args.path
        
    if args.path == None and args.primer == 'True' and args.training == 'True':
        raise ValueError('A path for the saved weights needs to be specified if not running a primer or eval mode.')
    
    instance_name = create_training_tag(args)
    
    #checks selected model params, if selected, also raises error if no model selected and training mode and no primer is selected
    if args.model == None and not args.primer == 'True' and args.training == 'True':
        raise ValueError('When in training mode a model from ("6x7","8x9","10x11","12x13") must be specified')
    
    if args.model == "6x7":
        env, buffer, model_q, model_target = initialize_models((6,7), path=path)
        training(env, buffer, model_q, model_target, instance_name, "dqn_6x7", args.path, epochs=int(args.epochs))
        exit()
    
    elif args.model == "8x9":
        env, buffer, model_q, model_target = initialize_models((8,9), path=path)
        training(env, buffer, model_q, model_target, instance_name, "dqn_8x9", args.path, epochs=int(args.epochs))
        exit()
    
    elif args.model == "10x11":
        env, buffer, model_q, model_target = initialize_models((10,11), path=path)
        training(env, buffer, model_q, model_target, instance_name, "dqn_10x11", args.path, epochs=int(args.epochs))
        exit()
    
    elif args.model == "12x13":
        env, buffer, model_q, model_target = initialize_models((12,13), path=path)
        training(env, buffer, model_q, model_target, instance_name, "dqn_12x13", args.path, epochs=int(args.epochs))
        exit()
    
    elif not args.primer == 'True':
        raise ValueError('When in training mode a model from ("6x7","8x9","10x11","12x13") must be specified')
    
    #if primer is selected this part is run and the rest ignored
    if args.primer == 'True':
        run_primer()
        exit()
    
    #evaluation mode
    else:
        exit() #TODO: handle eval mode later in here