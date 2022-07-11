# imports
import tensorflow as tf
import os
import gym
import pandas as pd

from datetime import datetime, timedelta
from model import CarRacingAgent
from gradients import estimate_step_len, sample_trajectories
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

# load environment
car_racing_env = gym.make("CarRacing-v0")

# define epoch length
epochs = 100
episodes_per_epoch = 1000

# define hyperparameters
learning_rate = 0.01

# define loss and optimizer
loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate)

# define data storage functions

def epoch_results(data : dict) -> str:
    '''
    ADD
    '''
    
    data['TIME'] = str(datetime.now())
    data['LOSS'] = float(-2893.1)
    data['DURATION'] = str(timedelta(hours=5))
    data['BEST'] = float(-2893.1)
    data['WORST'] = float(-2893.1)
    
    data = [[data['TIME'], data['LOSS'], data['DURATION'], data['BEST'], data['WORST']]]
    
    try:
        df = pd.read_csv('./train_data.csv')
        appendix = pd.DataFrame(data, columns=df.columns)
        df = df.append(appendix)
    
    except Exeption as e:
        print('e')
        
    finally:
        df.to_csv('train_data.csv', index=False)
        return f'Epoch results were stored in train_data.csv.'

# define training functions

# initialization
duration = 0

while duration == 0:
    print(f'Please select a training duration in hours after which training shall be stopped at earliest convenience: ')
    duration = int(input())
    
    if type(duration) != int or duration <= 0:
        duration = 0
        print('\n')
        print(f'Invalid duration, please select a positive integer in hours.')
        print('\n')

starttime = datetime.now()
endtime = datatime.now() + timedelta(hours=duration)

print(f'Beginning training from current epoch milestone if present...')

current_epoch = 0
try:
    df = pd.read_csv('./train_data.csv')
    current_epoch = len(df)
    
except FileNotFoundError:
    print(f'Warning: Unable to load train_data.csv, assuming current epoch is initial epoch.')

# model initialization
model = CarRacingAgent()
model(tf.random.normal((10,96,96,3))) # create the graph by passing input once

try:
    model.load('./weights.h5')
    
except FileNotFoundError:
    print(f'Warning: Unable to load weights, assuming model has not been trained before and starting training now.')

# training loop

for epoch in range(current_epoch, epochs):
    # creating data variables
    epoch_start = datetime.now()
    avg_loss = []
    best, worst = None, None
    
    # episode loop
    for _ in tqdm(range(episodes_per_epoch), desc=f'Running epoch {epoch}: '):
        # sample trajectories for 32 multithreaded episodes with each up to a maximum of 100 steps
        step_len = estimate_step_len()
        args = [(model,deepcopy(env),step_len) for _ in range(32)]
        results = ThreadPool(6).starmap(sample_trajectories,args)
        
        # TODO use results
        # ADD
        
        # TODO update model
#         policy_update()
        
        # TODO get stats
    
    # generate epoch data
    epoch_end = datetime.now()
    epoch_duration = epoch_end - epoch_start
    
    data = {'TIME' : str(epoch_end),
            'LOSS' : float(tf.reduce_mean(avg_loss).numpy()),
            'DURATION' : str(epoch_duration),
            'BEST' : float(best.numpy()),
            'WORST' : float(worst.numpy())}
    
    # store epoch result data
    epoch_results(data)
    
    # store model weights and rename older weights
    try:
        os.rename('./weights.h5', f'./old_weights_{epoch-1}.h5')
    
    except:
        print(f'Warning: Either an older model weights.h5 did not exist or renaming it was unsuccessful.')
    
    model.save()
    
    # test if continue or break loop
    if (endtime - datetime.now()).seconds <= 0:
        break

print(f'Scheduled a shutdown in 5mins that can be stopped with cmd("shutdown /a") if necessary')
os.system("shutdown -s -t 300") # schedule a shutdown that can be stopped with cmd("shutdown /a") if necessary.