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
        self.loss = tf.keras.losses.MeanSquaredError()

        #fixed sigma to determine an action together with mu as policy network outputs
        self.sigma = dev


    @tf.function
    def __call__(self, x):
        '''
        The Model call function, receives a (batch,96,96,3) input tensor.
        
        :x (tf.Tensor): Tensor of shape (batch,96,96,3).
        
        :returns (tuple): A tuple of two tensors of shape (batch,4) as action and log prob of action.
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
        return tfp.distributions.MultivariateNormalDiag(action, sigmas)

    def save(self, path='./weights.h5'):
        '''
        ADD
        '''
        
        self.save_weights(path)
    
    def load(self, path='./weights.h5'):
        '''
        ADD
        '''
        
        self.built = True
        self.load_weights(path)
        
class ValueNetwork(tf.keras.Model):
    '''
    ADD
    '''

#     def __init__(self, dev=0.05):
#         '''
#         Initialization function with model class super call.
#         '''
#         super(CarRacingAgent, self).__init__()

#         self.conv_1 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
#         self.conv_2 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
#         self.maxpool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
#         self.conv_3 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
#         self.conv_4 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
#         self.maxpool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
#         self.glbavg = tf.keras.layers.GlobalAveragePooling2D()
#         self.flatten = tf.keras.layers.Flatten()
#         self.steering = tf.keras.layers.Dense(units=1, activation='tanh')
#         self.accel = tf.keras.layers.Dense(units=2, activation='sigmoid')

#     @tf.function
#     def __call__(self, x):
#         '''
#         The Model call function, receives a (batch,96,96,3) input tensor.
        
#         :x (tf.Tensor): Tensor of shape (batch,96,96,3).
        
#         :returns (tf.Tensor): A tensor of shape (batch,4) as policy.
#         '''
        
#         x = self.conv_1(x)
#         x = self.conv_2(x)
#         x = self.maxpool_1(x)
#         x = self.conv_3(x)
#         x = self.conv_4(x)
#         x = self.maxpool_2(x)
#         x = self.glbavg(x)
#         x = self.flatten(x)
        
#         steering = self.steering(x)
#         accel = self.accel(x)
        
#         x = tf.concat([steering,accel],axis=1)
        
#         return x
    
    def save(self, path='./val_weights.h5'):
        '''
        ADD
        '''
        
        self.save_weights(path)
    
    def load(self, path='./val_weights.h5'):
        '''
        ADD
        '''
        
        self.built = True
        self.load_weights(path)