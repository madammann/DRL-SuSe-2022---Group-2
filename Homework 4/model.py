import tensorflow as tf
import tensorflow_probability as tfp

class CarRacingAgent(tf.keras.Model):
    '''
    Car Racing Agent in the Car Racing environment
    '''

    def __init__(self, learning_rate, dev=0.05):
        '''
        Initialization function with model class super call.
        '''
        super(CarRacingAgent, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(96, 96, 1))
        self.conv_2 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
        self.maxpool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.conv_3 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
        self.conv_4 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
        self.maxpool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.glbavg = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.steering = tf.keras.layers.Dense(units=1, activation='tanh')
        self.accel = tf.keras.layers.Dense(units=1, activation='sigmoid')
        self.breaking = tf.keras.layers.Dense(units=1, activation='sigmoid')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        #fixed sigma to determine an action together with mu as policy network outputs
        self.sigma = dev


    @tf.function
    def __call__(self, x):
        '''
        The Model call function, receives a (batch,96,96,1) input tensor.
        
        :x (tf.Tensor): Tensor of shape (batch,96,96,1).
        
        :returns (distribution): A Multivariate Normal distribution parameterised by a tuple of three tensors as each action-dimension's mu and fixed sigma.
        '''
        
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.maxpool_1(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.maxpool_2(x)
        x = self.glbavg(x)
        x = self.flatten(x)
        
        mu_steering = self.steering(x)
        mu_accel = self.accel(x)
        mu_breaking = self.breaking(x)

        action = tf.concat((mu_steering, mu_accel, mu_breaking), axis=-1)
        #Create multidimensional sigma for Multivariate Normal distribution
        sigmas = tf.stack((self.sigma,self.sigma,self.sigma))

        #Using Multivariate Normal distribution due to multidimensional action space
        return tfp.distributions.MultivariateNormalDiag(action, scale_diag = sigmas)

    def save(self, path='./weights.h5'):

        self.save_weights(path)
    
    def load(self, path='./weights.h5'):

        self.built = True
        self.load_weights(path)


class ValueNetwork(tf.keras.Model):
    '''
    Critic Model for A2C
    '''

    def __init__(self, learning_rate):
        '''
        Initialization function with model class super call.
        '''
        super(ValueNetwork, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(96, 96, 1))
        self.conv_2 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")
        self.maxpool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv_3 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")
        self.conv_4 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")
        self.maxpool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.glbavg = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()

        self.out_value = tf.keras.layers.Dense(units=1, activation=None)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()


    @tf.function
    def __call__(self, x):
        '''
        The Model call function, receives an observed state as a (batch,96,96,1) input tensor, outputs the value of that state.

        :x (tf.Tensor): Tensor of shape (batch,96,96,1).

        :returns (tensor): The value of the state (which is input)
        '''

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.maxpool_1(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.maxpool_2(x)
        x = self.glbavg(x)
        x = self.flatten(x)

        output = self.out_value(x)

        return output

    def save(self, path='./val_weights.h5'):
        
        self.save_weights(path)
    
    def load(self, path='./val_weights.h5'):
        
        self.built = True
        self.load_weights(path)