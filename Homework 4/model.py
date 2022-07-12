import tensorflow as tf
import tensorflow_probabilities as tfp

class CarRacingAgent(tf.keras.Model):
    '''
    Car Racing Agent in the Car Racing environment
    '''

    def __init__(self, dev=0.05):
        '''
        Initialization function with model class super call.
        '''
        super(CarRacingAgent, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
        self.conv_2 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
        self.maxpool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.conv_3 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
        self.conv_4 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
        self.maxpool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.glbavg = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.steering = tf.keras.layers.Dense(units=1, activation='tanh')
        self.accel = tf.keras.layers.Dense(units=2, activation='sigmoid')
        
        self.dist = tfp.

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
        
        steering = self.steering(x)
        accel = self.accel(x)
        
        action = tf.concat([steering,accel],axis=1)
        log_prob = None
        
        return (x, log_prob)
    
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