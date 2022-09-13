import numpy as np

import tensorflow as tf

class ConnectFourModel(tf.keras.Model):
    '''
    A Connect-4 Model to play in the Connect-4 environment.
    '''

    def __init__ (self, grid_size, learning_rate):
        '''
        Method to initialize the model inheriting from keras.Model with layers, loss function and optimizer.
        
        :param grid_size (tuple): The grid size of the Connect4 environment (M,N) the model is supposed to train and perform on.
        :param learning_rate (float): The learning rate to use in the Adam optimizer.
        '''

        super(ConnectFourModel, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')
        
        self.glbavg = tf.keras.layers.GlobalAveragePooling2D()
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.output_layer = tf.keras.layers.Dense(units=grid_size[1], activation='sigmoid')

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def __call__(self, x):
        '''
        Method for the feed-forward of input data inside the model.
        
        :param x (tf.Tensor): The input to the model in shape (batch, M, N, 2) with MxN being the game grid size.
        
        :returns (tf.Tensor): The final output of the model as tensor of shape (batch, N).
        '''
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.glbavg(x)
        x = self.flatten(x)
        x = self.output_layer(x)

        return x

    def save(self, path='./weights.h5'):
        '''
        Method for storing the model's current weights at a given path.
        
        :param path (str): A legal file path for the weights.h5 file.
        '''
        
        self.save_weights(path)
    
    def load(self, path='./weights.h5'):
        '''
        Method for loading, assumed to be compatible, model weights.
        
        :param path (str): A legal file path to the weights.h5 file to load.
        '''
        
        self.built = True
        self.load_weights(path)
