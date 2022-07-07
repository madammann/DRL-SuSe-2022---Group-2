import tensorflow as tf

class CarRacingAgent(tf.keras.Model):
    '''
    Car Racing Agent in the Car Racing environment
    '''

    def __init__(self):
        '''
        ADD
        '''
        super(CarRacingAgent, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
        self.conv_2 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
        self.maxpool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.conv_3 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
        self.conv_4 = tf.keras.layers.Conv2D(16, (3,3), activation="relu")
        self.maxpool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.glbavg = tf.keras.layers.GlobalAveragePooling2D()
        self.pol_out = tf.keras.layers.Dense(units=4, activation='softmax')

        # 2dconv with 16 filters, 3x3, relu 2x
        # maxpool2d
        # 2dconv with 16 filters, 3x3, relu 2x
        # maxpool2d
        # globalavg pooling up to 9 values
        # flatten
        # last layer dense(4,softmax)


    @tf.function
    def __call___(self, x):
        '''
        ADD
        '''

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.maxpool_1(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.maxpool_2(x)
        x = self.glbavg(x)
        x = self.pol_out(x)

        return x
