import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import random
import signal
import gc
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Conv2D
import sys
from collections import deque
import os

# from pympler import asizeof
import pickle
from gymnasium.wrappers import AtariPreprocessing, RecordEpisodeStatistics
import psutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# tf.logging.set_verbosity(tf.logging.INFO)
# export TF_CPP_MIN_LOG_LEVEL="3"


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
        # state: a list of frames
        if len(self.memory) > self.replay_memory_size:
            self.memory.popleft()
        stacked_state = np.array(state)

        self.memory.append((stacked_state, action, reward, next_state, done))
        # print(f'Size of memory entry: {asizeof.asizeof(self.memory.pop())}')
        # print(f'type of next_state: {type(next_state[0][0][0])}')
        # print(f'type of stacked_state: {type(stacked_state[0][0][0])}')

    def act(self, state):
        # print(f'state buffer: {self.state_buffer.shape}')
        # self.state_buffer = state
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        actual_batch_size = len(minibatch)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        # next_states_unstacked = np.array([i[3] for i in minibatch])
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


process = psutil.Process()


def get_memory(proc):
    memory_info = proc.memory_info()
    memory_usage_mb = memory_info.rss / (1024**2)
    return memory_usage_mb


mem_data = []

# 1 for train, 0 for play
train_play = 1


def sigint_handler(signal, frame):
    with open(f"mem_usage.pkl", "wb") as f:
        pickle.dump(mem_data, f)
    agent.handle_interrupt()


if train_play:
    # Initialize gym environment and the agent
    env = gym.make("BreakoutNoFrameskip-v4")

    env = AtariPreprocessing(env, noop_max=10)
    env = RecordEpisodeStatistics(env)
    print(env.observation_space)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)

    signal.signal(signal.SIGINT, sigint_handler)

    # Iterate the game
    prev_frame = None
    num_eps = 1500
    save_rate = 100
    total_score: float
    total_steps = 0
    for e in range(num_eps):
        # reset state in the beginning of each game
        state, info = env.reset()
        state = np.stack(
            [state] * agent.frame_stack_size, axis=-1
        )  # initialize the state with the same frame repeated

        if (e > 0) and (e % save_rate == 0):
            agent.save(f"benchmark_{e}.h5")
            with open(f"{e}_rewards.pkl", "wb") as f:
                pickle.dump(list(env.return_queue), f)

        total_score = 0.0
        # time ticks
        for time in range(5000):
            # Decide action
            action = agent.act(state)

            # Advance the game to the next frame based on the action.
            next_state, reward, done, trunc, info = env.step(action)

            total_score += reward

            next_state = np.concatenate(
                (state[..., 1:], np.expand_dims(next_state, -1)), axis=-1
            )

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            if done:
                # print the score and break out of the loop
                print(
                    "episode: {}/{}, score: {}, num_steps: {}".format(
                        e, num_eps, total_score, total_steps + time
                    )
                )
                break

            # train the agent with the experience of the episode
            agent.replay(32)

        # update target model weights every episode
        agent.update_target_model()
        total_steps += time
        # print(f'Size of memory entry: {asizeof.asizeof(agent.memory)}')
        print(f"mem: {get_memory(process)}")
        mem_data.append((total_steps, get_memory(process)))
        k.clear_session()
        gc.collect()
    # Save the model after training
    agent.save("dqn_model_bigly.h5")
    with open(f"mem_usage.pkl", "wb") as f:
        pickle.dump(mem_data, f)

else:
    # Initialize gym environment and the agent
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
    env = AtariPreprocessing(env, noop_max=10)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)

    agent.load("benchmark_100.h5")
    agent.frame_stack_size = 4
    agent.epsilon = 0.1

    # print(f'LOOK OBERSVATION SPACE: {env.observation_space.shape}')
    # Iterate the game

    num_eps = 10
    for e in range(num_eps):
        # reset state in the beginning of each game
        state, info = env.reset()
        state = np.stack([state] * agent.frame_stack_size, axis=-1)
        # print(f'THIS IS THE STATE: {state}')
        # state = np.reshape(state, [1, *state.shape])

        # time ticks
        for time in range(5000):
            # turn this on if you want to render
            env.render()

            # Decide action
            action = agent.act(state)

            # Advance the game to the next frame based on the action.
            # print(f'THIS IS THE ACTION: {env.step(action)}')
            next_state, reward, done, trunc, info = env.step(action)
            # print(f'LOOK NEXT STATE: {next_state.shape}')

            next_state = np.concatenate(
                (state[..., 1:], np.expand_dims(next_state, -1)), axis=-1
            )

            # Remember the previous state, action, reward, and done
            # next_state = np.reshape(next_state, [1, *next_state.shape])
            # print(f'LOOK NEXT STATE RESHAPED: {next_state.shape}')
            # agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}".format(e, num_eps, time))
                break
