import tensorflow as tf

class PolicyNet(tf.keras.Model):
    def __init__(self):

        super(PolicyNet, self).__init__()

        # Policy Network layers

        self.Dense_1 = tf.keras.layers.Dense(16, activation="relu")

        # Other layers and the output layer



    @tf.function
    def __call__(self, x):

        x = self.Dense_1(x)
        x = # Policy Network layer .. (x)
        x = # Policy Network layer .. (x)

        # ...



    return x

class ValueNet(tf.keras.Model):  # aka Baseline



    def __init__(self):

        super(ValueNet, self).__init__()

        # Value Network layers
        self.Dense_1 = tf.keras.layers.Dense(16, activation="relu")

        # Other layers and the output layer



    @tf.function
    def __call__(self, x):

        x = self.Dense_1(x)
        x = # Value Network Layer .. (x)
        x = # Value Network Layer .. (x)

        # ...



    return x


class PolicyGradient():

    def __init__(self, env, num_iterations, batch_size):

        self.env = env
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.observation_space =
        self.action_space =
        self.gamma =
        self.policy_net = PolicyNet()
        self.value_net = ValueNet()
