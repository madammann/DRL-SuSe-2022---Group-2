import tensorflow as tf


class ExperienceReplayBuffer:
    def __init__(self):
        # parameters like length
        pass
    
    def append(self, elem):
        # appending
        pass
    
    def dequeue(self):
        # first out
        pass
    
    def pop(self):
        pass # return some elements and call dequeue
    
    # add functions for getter returning as tfds
    
class LunarLanderModel(tf.keras.Model):
    def __init__(self):
        # make super call on model superclass
        self.input_layer = None
        self.hidden = None
        self.hidden2 = None
        self.output_layer = None
        #TODO tweaks
    
    
    @tf.function
    def __call__(self, x):
        x = self.input_layer(x)
        x = self.hidden(x)
        x = self.hidden2(x)
        x = self.output_layer(x)
        # TODO: Replay Buffer and tweaks
        return x

