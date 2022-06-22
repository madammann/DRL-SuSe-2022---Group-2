import tensorflow as tf
import collections #for deque
import random
from tqdm import tqdm 


class ExperienceReplayBuffer:
    def __init__(self, action_space, size_replaybuffer = 100000, size_batch = 50):

        # parameters like length
        self.size_replaybuffer = size_replaybuffer
        self.action_space = action_space
        self.size_batch = size_batch
        #self.experience = ()
        self.memory = deque(maxlen=size_replaybuffer) 
        pass
    
    def append(self, state, action, reward, next_state, terminal):
        # appending
        new_experience = (state, action, reward, next_state, terminal)
        self.memory.append(new_experience)

        pass

    # deque() uses fifo. when the list reaches the maximum length, it will get rid of the first element. 

    def sampling(self):
        
        sample_experience = random.sample(self.memory, k=self.size_batch)
        print(sample_experience)

        return sample_experience

    def fill_buffer(self):
        
        for i in tqdm(range(self.size_replaybuffer), desc = "Buffer Progress"):
          buff = []
          buff.append(i)
        pass
        
    #def dequeue(self):
        # first out
                   

     #   pass
    
    #def pop(self):
     #   pass # return some elements and call dequeue
    
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

