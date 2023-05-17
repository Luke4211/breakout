import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten

# from tensorflow.keras.optimizers import Adam
from keras.optimizers import RMSprop
import numpy as np
import random
import signal
import gc
from keras import backend as k
from keras.layers import Conv2D
import sys
from collections import deque
import os

import pickle
import psutil


class DQNAgent:
    def __init__(
        self,
        state_shape,
        action_space,
        replay_memory_size=50000,
        frame_stack_size=4,
        epsilon_lin=False,
        gamma=0.98,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.999,
        batch_size=32,
        decay_func=lambda x, y: x * y,
    ):
        self.state_shape = state_shape
        self.action_space = action_space
        self.memory = deque(maxlen=replay_memory_size)
        self.replay_memory_size = replay_memory_size
        self.frame_stack_size = frame_stack_size
        self.epsilon_lin = epsilon_lin
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.decay_func = decay_func
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()

        model.add(
            Conv2D(
                32,
                (8, 8),
                strides=(4, 4),
                padding="same",
                activation="relu",
                input_shape=[*self.state_shape, self.frame_stack_size],
            )
        )
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation="relu"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu"))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(self.action_space))
        model.compile(loss="mse", optimizer=RMSprop(), run_eagerly=False)
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.replay_memory_size:
            self.memory.popleft()
        stacked_state = np.array(state)

        self.memory.append((stacked_state, action, reward, next_state, done))
        # print(f'Size of memory entry: {asizeof.asizeof(self.memory.pop())}')
        # print(f'type of next_state: {type(next_state[0][0][0])}')
        # print(f'type of stacked_state: {type(stacked_state[0][0][0])}')

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)

        return np.argmax(act_values[0])  # returns action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        actual_batch_size = len(minibatch)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        targets = rewards + self.gamma * (
            np.amax(self.target_model.predict_on_batch(next_states), axis=1)
        ) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(actual_batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)

        self.epsilon = self.decay_func(self.epsilon, self.epsilon_decay)

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = tf.keras.models.load_model(name)

    def handle_interrupt(self):
        print("Interrupted, saving...")
        self.save("interrupted_model.h5")
        sys.exit(0)


class TrainAgent:
    def __init__(self, env, agent, num_eps=1500, max_ep_steps=5000, save_rate=100):
        self.env = env
        self.agent = agent
        self.num_eps = num_eps
        self.max_ep_steps = max_ep_steps
        self.save_rate = save_rate
        self.process = psutil.Process()
        self.mem_data = []

        def handler(signal, frame):
            self.sigint_handler(signal, frame)

        signal.signal(signal.SIGINT, handler)

    def get_memory(self):
        memory_info = self.process.memory_info()
        memory_usage_mb = memory_info.rss / (1024**2)
        return memory_usage_mb

    def sigint_handler(self, signal, frame):
        with open("mem_usage.pkl", "wb") as f:
            pickle.dump(self.mem_data, f)
        self.agent.handle_interrupt()

    def train(self):
        total_score: float
        total_steps = 0
        for e in range(self.max_ep_steps):
            # reset state in the beginning of each game
            state, info = self.env.reset()
            state = np.stack(
                [state] * self.agent.frame_stack_size, axis=-1
            )  # initialize the state with the same frame repeated

            if (e > 0) and (e % self.save_rate == 0):
                self.agent.save(f"benchmark_{e}.h5")
                with open(f"{e}_rewards.pkl", "wb") as f:
                    pickle.dump(list(self.env.return_queue), f)

            total_score = 0.0
            # time ticks
            for time in range(5000):
                # Decide action
                action = self.agent.act(state)

                # Advance the game to the next frame based on the action.
                next_state, reward, done, trunc, info = self.env.step(action)

                total_score += reward

                next_state = np.concatenate(
                    (state[..., 1:], np.expand_dims(next_state, -1)), axis=-1
                )

                # Remember the previous state, action, reward, and done
                self.agent.remember(state, action, reward, next_state, done)

                # make next_state the new current state for the next frame.
                state = next_state

                if done:
                    # print the score and break out of the loop
                    print(
                        "episode: {}/{}, score: {}, num_steps: {}".format(
                            e, self.num_eps, total_score, total_steps + time
                        )
                    )
                    break

                # train the agent with the experience of the episode
                self.agent.replay()

            # update target model weights every episode
            self.agent.update_target_model()
            total_steps += time
            # print(f'Size of memory entry: {asizeof.asizeof(agent.memory)}')
            print(f"mem: {self.get_memory(self.process)}")
            self.mem_data.append((total_steps, self.get_memory(self.process)))
            k.clear_session()
            gc.collect()
        # Save the model after training
        self.agent.save("dqn_model_bigly.h5")
        with open("mem_usage.pkl", "wb") as f:
            pickle.dump(self.mem_data, f)

    def play_model(self):
        for e in range(self.num_eps):
            state, info = self.env.reset()
            state = np.stack([state] * self.agent.frame_stack_size, axis=-1)

            for time in range(self.max_ep_steps):
                self.env.render()

                action = self.agent.act(state)

                next_state, _, done, _, _ = self.env.step(action)

                next_state = np.concatenate(
                    (state[..., 1:], np.expand_dims(next_state, -1)), axis=-1
                )

                state = next_state

                if done:
                    break
