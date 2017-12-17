import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras import optimizers
import numpy as np
from collections import deque
np.random.seed(1)

class DQN:
    def __init__(self):
        # dqn - model; can use nadam for convergence
        self.brain = self.__createLayers__()
        self.brain.compile(optimizer='nadam', loss='mse')
        
        # hyperparameters
        self.action_space = 2
        self.alpha = 0.95
        self.gamma = 0.9

        # memory
        self.memory = deque(list(), maxlen=1000)

        #keras.utils.plot_model(self.brain, to_file='model.png', show_shapes=True)

    def __createLayers__(self, action_space=2):
        # create the actual model for the brain
        brain = Sequential()
        brain.add(Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)))
        brain.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        brain.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        brain.add(Flatten())
        brain.add(Dense(512, activation='relu'))
        brain.add(Dense(action_space, activation='linear'))
        return brain

    def makeFragment(self, state, action, reward, next_state):
        # creates a memory-fragment <s,a,r,s'>
        return tuple([state, action, reward, next_state])

    def remember(self, fragment, insert_rate=2.0):
        # save the memory-fragment <s,a,r,s'> state in replay-memory
        while True:
            value = np.random.random()
            if value <= (insert_rate/1.414):
                self.memory.appendleft(fragment)
                break
            elif value <= (insert_rate):
                self.memory.append(fragment)
                break
            else:
                self.memory.rotate(1)

    def sampleMiniBatch(self, batch_size):
        # sample a mimibatch for recall process
        mini_batch = list()
        for _ in range(batch_size):
            mini_batch.append(self.memory.pop())
        for fragment in mini_batch:
            self.remember(fragment, 0.8)
        return mini_batch
    
    def forwardPass(self, state):
        # does Q(s) -> [r_1, r_2, ..., r_i] for each action [a_1, a_2, ..., a_i]
        return self.brain.predict(np.expand_dims(state, axis=0))[0]

    def batchForwardPass(self, states):
        # same as forwardPass but evaluate of multiple states at once
        return self.brain.predict(states)
    
    def trainOnFragment(self, fragment):
        # train on a single memory-fragment <s,a,r,s'>
        state, action, reward, next_state = fragment
        predictQ = self.forwardPass(state)
        predictNQ = self.forwardPass(next_state)
        targetAQ = predictQ.copy()
        targetAQ[action] = max(predictNQ)*self.gamma + reward
        #print predictQ, action, reward, targetAQ #status-line
        self.brain.fit(np.expand_dims(state, axis=0), np.expand_dims(targetAQ, axis=0), verbose=0, epochs=1)
        #print (targetAQ-self.forwardPass(state))**2 #print-loss

    def batchTrainOnFrament(self, fragments):
        # train on multiple memory-fragments <s,a,r,s'>
        batch_size = len(fragments)
        states, actions, rewards, next_states = zip(*fragments)
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)
        assert states.shape == next_states.shape
        assert batch_size == states.shape[0] and batch_size == len(actions) and batch_size == len(rewards)

        predictQs = self.batchForwardPass(states)
        predictNQs = self.batchForwardPass(next_states)
        targetAQs = predictQs.copy()
        targetAQs[np.arange(batch_size), actions] = np.amax(predictNQs, 1) * self.gamma + rewards

        #print '\n\npredictQs =', predictQs, '\nactions =', actions, '\nrewards =', rewards, '\ntargetAQs =', targetAQs #status-line
        self.brain.fit(states, targetAQs, verbose=0, epochs=1)
        #print '\n\npredictQs =', (targetAQs-self.batchForwardPass(states))**2 #print-loss

if __name__ == '__main__':
    dqn = DQN()
    #print dqn.forwardPass(np.random.randint(256, size=(84, 84, 4)))
    #print dqn.batchForwardPass(np.random.randint(256, size=(8, 84, 84, 4)))
    tests = 64
    batch = list()
    while tests > 0:
        s = np.random.randint(256, size=(84, 84, 4))
        a = np.random.randint(2)
        r = np.random.randint(5)
        s1 = np.random.randint(256, size=(84, 84, 4))
        fragment = dqn.makeFragment(s,a,r,s1)
        batch.append(fragment)
        dqn.trainOnFragment(fragment)
        tests -= 1
    dqn.batchTrainOnFrament(batch)
