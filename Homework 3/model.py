import numpy as np

import tensorflow as tf

class ExperienceReplayBuffer:
    '''
    The class for implementing the experience replay buffer.
    
    :att memory: None if empty, else a numpy array in form of a stack of elements for sampling.
    :att size (int): The maximum size of the buffer.
    :att batch_size (int): The size of a return-batch, used later for sampling in batch size.
    '''
    
    def __init__(self, size = 10000, batch_size = 32):
        '''
        Method for initializing the experience replay buffer.
        
        :param size (int): The maximum size of the buffer.
        :param batch_size (int): The size of a return-batch, used later for sampling in batch size.
        '''
        
        self.size = size
        self.batch_size = batch_size
        self.memory = None
    
    def append(self, element):
        '''
        Method for appending an element to the memory of the replay buffer.
        Follows first-in-first-out scheme for appending elements if the memory size is exceeded.
        
        :param element (list): A list of [s,a,r,s_prime,is_terminal] to append to the memory.
        '''
        if len(element) == 5 and type(element) == list:
            if self.memory == None:
                self.memory = [element]
            else:
                self.memory.append(element)

            #remove earliest element from list if it gets bigger than indicated by size
            if len(self.memory) > self.size:
                self.memory.pop(0)
        else:
            raise TypeError('The experience replay buffer can only append tuples of size 5.')

    def sample(self):
        '''
        Method for sampling self.batch_size elements from memory and returning them in dataset batch form.
        
        :returns (list): list of size self.batch_size of samples from memory
        '''
        
        if len(self.memory) > self.batch_size:
            sample_indices = np.random.choice(np.arange(0, len(self.memory)), self.batch_size, replace=False)
            samples = [self.memory[idx] for idx in sample_indices]
            
            return samples
        
        else:
            return self.memory #TODO: shuffle
            #raise AttributeError('A sample was requested but the memory was not yet filled enough to provide one.')

    
class LunarLanderModel(tf.keras.Model):
    '''
    The LunarLanderModel which will act as the agent in LunarLander environment.
    '''
    
    def __init__(self):
        '''
        Method for initializing the model using the subclassing of a tf.keras.Model.
        '''
        
        super(LunarLanderModel, self).__init__()
        self.input_layer = tf.keras.layers.Dense(units=8, activation='relu')
        self.hidden = tf.keras.layers.Dense(units=20, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(units=20, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=4, activation='softmax')
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss = tf.keras.losses.MeanSquaredError()
    
    @tf.function
    def __call__(self, x):
        '''
        Method for the feed-forward of data inside the model.
        
        :param x (tf.Tensor): The input to the model in shape (batch, xyz)
        :returns (tf.Tensor): The final output of the model as tensor of shape (batch, xyz)
        '''
        if len(x.get_shape().as_list()) < 2:
            x = tf.expand_dims(x, axis = 0)

        x = self.input_layer(x)
        x = self.hidden(x)
        x = self.hidden2(x)
        x = self.output_layer(x)

        return x