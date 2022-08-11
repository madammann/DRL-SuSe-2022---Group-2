import numpy as np

import tensorflow as tf
#import tensorflow_datasets as tfds



class ExperienceReplayBuffer:

    """ 
    Implementing the replay buffer.
    :att memory: None if empty, else a numpy array in form of a stack of elements for sampling.
    :att size (int): The maximum size of the buffer.
    :att batch_size (int): The size of a return-batch, used later for sampling in batch size.
    """


    def __init__(self, size = 10000, batch_size =32):

        """ Initialized the replay buffer. """

        self.size = size
        self.batch_size = batch_size
        self.memory = None

    def append(self, element):

        """ 
        Adding a new element to the memory of the replay buffer. 
        First-in-first-out method is used for appending, therefore
        the earlier elements will be erased when the size is exceeded 
        to make room for new elements.

        element (list): A list of [s,a,r,s_prime,is_terminal] to append to the memory.
        """

        if len(element) == 5 and type(element) == list:
            appendix = [e for e in element[0]]+[element[1]]+[element[2]]+[e for e in element[3]]+[int(element[4])]
            
            if type(self.memory) == np.ndarray:
                appendix = np.array(appendix,dtype='float32')
                self.memory = np.vstack([self.memory,appendix])
                
            else:
                self.memory = np.array(appendix,dtype='float32')
                self.memory = self.memory[-self.size:]

        else:
            raise TypeError('The experience replay buffer can only append tuples of size 5.')

    def sample(self):
        """
        Method for sampling self.batch_size elements from memory and returning them in dataset batch form.
        
        :returns (tf.data.Dataset): A tensorflow dataset slice with shape (batch_size,5).
        """
        
        if len(self.memory) > self.batch_size:
            sample_indices = np.random.choice(np.arange(0, len(self.memory)), self.batch_size, replace=False)
            samples = [self.memory[idx] for idx in sample_indices]
            
            data = self.preprocess(samples)
            
            return data
        
        else:
            raise AttributeError('A sample was requested but the memory was not yet filled enough to provide one.')
            

    def preprocess(self, dataset):
                
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        
        # convert data to tuple
        dataset = dataset.map(lambda data: (data[0:8],data[8],data[9],data[10:18],data[18]))
        
        return dataset

class ConnectFourModel(tf.keras.Model):
    """ 
    A Connect-4 Model to play in the Connect-4 environment.     
    """

    def __init__ (self):

        """"
        Method to initialize the model using tf.keras.Model.
        """

        super (ConnectFourModel, self).__init__()

        self.input_layer = tf.keras.layers.Dense(activation='relu')
        self.hidden = tf.keras.layers.Dense(units=20, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(units=20, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=7, activation='sigmoid')

    
    
    @tf.function
    def __call__(self, x):

        """
        Method for the feed-forward of data inside the model.
        
        :param x (tf.Tensor): The input to the model in shape (batch, xyz)
        :returns (tf.Tensor): The final output of the model as tensor of shape (batch, xyz)
        """

        x = self.input_layer(x)
        x = self.hidden(x)
        x = self.hidden2(x)
        x = self.output_layer(x)

        return x

    def save(self, path='./weights.h5'):
        self.save_weights(path)
    
    def load(self, path='./weights.h5'):
        self.built = True
        self.load_weights(path)
