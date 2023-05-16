import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import random
import copy
import signal
import gc
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU
from tensorflow.keras.callbacks import Callback
import sys
from collections import deque
import os
from pympler import asizeof

from gymnasium.wrappers import AtariPreprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#tf.logging.set_verbosity(tf.logging.INFO)
# export TF_CPP_MIN_LOG_LEVEL="3"

class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


class DQNAgent:
    def __init__(self, state_shape, action_space, replay_memory_size=50000, frame_stack_size = 4, epsilon_lin=False):
        self.state_shape = state_shape
        self.action_space = action_space
        self.memory = deque(maxlen=replay_memory_size)
        self.replay_memory_size = replay_memory_size
        self.frame_stack_size = frame_stack_size
        self.state_buffer = np.array([])
        self.epsilon_lin = epsilon_lin
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay_linear = 0.000005
        self.epsilon_decay = 0.9999
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same', 
                        activation='relu', input_shape=[*self.state_shape, self.frame_stack_size]))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', 
                        activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', 
                        activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_space))
        model.compile(loss='mse', optimizer=RMSprop(), run_eagerly=False)
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):

        #state: a list of frames
        if len(self.memory) > self.replay_memory_size:
            self.memory.popleft()
        stacked_state = np.array(state)
        

        self.memory.append((stacked_state, action, reward, next_state, done))
        #print(f'Size of memory entry: {asizeof.asizeof(self.memory.pop())}')
        #print(f'type of next_state: {type(next_state[0][0][0])}')
        #print(f'type of stacked_state: {type(stacked_state[0][0][0])}')

    def act(self, state):
        #print(f'state buffer: {self.state_buffer.shape}')
        #self.state_buffer = state
        if (np.random.rand() <= self.epsilon):
            return random.randrange(self.action_space)
        act_values = self.model.predict(np.expand_dims(state, axis=0),verbose=0)

        return np.argmax(act_values[0])  # returns action
    


    def replay(self, batch_size):
        '''if len(self.state_buffer) < self.frame_stack_size:
            #print(f'state_buffer: {len(self.state_buffer)}')
            return'''
        minibatch = random.sample(self.memory, batch_size)
        actual_batch_size = len(minibatch)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        #next_states_unstacked = np.array([i[3] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])


        targets = rewards + self.gamma*(np.amax(self.target_model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(actual_batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0, callbacks=ClearMemory())
        if self.epsilon > self.epsilon_min:
            if self.epsilon_lin:
                self.epsilon -= self.epsilon_decay_linear
            else:
                self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = tf.keras.models.load_model(name)

    def handle_interrupt(self):
        print("Interrupted, saving...")
        self.save('interrupted_model.h5')
        sys.exit(0)



# 1 for train, 0 for play
train_play = 1

def sigint_handler(signal, frame):
        agent.handle_interrupt()

if train_play:
    # Initialize gym environment and the agent
    env = gym.make('BreakoutNoFrameskip-v4')
    
    env = AtariPreprocessing(env, noop_max=10 )
    print(env.observation_space)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)

    signal.signal(signal.SIGINT, sigint_handler)
    
    # Iterate the game
    prev_frame = None
    num_eps = 10000
    save_rate = 100
    total_score: float
    total_steps = 0
    for e in range(num_eps):
        # reset state in the beginning of each game
        state, info = env.reset()
        state = np.stack([state]*agent.frame_stack_size, axis=-1)  # initialize the state with the same frame repeated
        
        if (e>0)and(e%save_rate == 0):
            agent.save(f'benchmark_{e}.h5')

        total_score = 0.0
        # time ticks
        for time in range(5000):
        

            # Decide action
            action = agent.act(state)

            # Advance the game to the next frame based on the action.
            next_state, reward, done, trunc, info  = env.step(action)
            
            total_score += reward

            next_state = np.concatenate( (state[..., 1:], np.expand_dims(next_state, -1) ), axis=-1)


            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}, num_steps: {}"
                        .format(e, num_eps, total_score, total_steps+time))
                break

            # train the agent with the experience of the episode
            if len(agent.memory) > 128:
                agent.replay(128)

        # update target model weights every episode
        agent.update_target_model()
        total_steps += time

    # Save the model after training
    agent.save("dqn_model_bigly.h5")

else:
    # Initialize gym environment and the agent
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
    env = AtariPreprocessing(env, noop_max=10, scale_obs=True )
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    
    agent.load('interrupted_model.h5')
    agent.frame_stack_size = 4
    agent.epsilon = 0.2

    #print(f'LOOK OBERSVATION SPACE: {env.observation_space.shape}')
    # Iterate the game

    num_eps = 10
    for e in range(num_eps):
        # reset state in the beginning of each game
        state, info = env.reset()
        state = np.stack([state]*agent.frame_stack_size, axis=-1)
        #print(f'THIS IS THE STATE: {state}')
        #state = np.reshape(state, [1, *state.shape])

        # time ticks
        for time in range(5000):

            # turn this on if you want to render
            env.render()

            # Decide action
            action = agent.act(state)

            # Advance the game to the next frame based on the action.
            #print(f'THIS IS THE ACTION: {env.step(action)}')
            next_state, reward, done, trunc, info  = env.step(action)
            #print(f'LOOK NEXT STATE: {next_state.shape}')

            next_state = np.concatenate( (state[..., 1:], np.expand_dims(next_state, -1) ), axis=-1)

            # Remember the previous state, action, reward, and done
            #next_state = np.reshape(next_state, [1, *next_state.shape])
            #print(f'LOOK NEXT STATE RESHAPED: {next_state.shape}')
            #agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                    .format(e, num_eps, time))
                break
