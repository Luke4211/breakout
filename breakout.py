import tensorflow as tf

# from tensorflow import keras
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
        gamma=0.98,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.999,
        batch_size=32,
        model_name="beta",
        decay_func=lambda x, y: x * y,
        learning_rate=0.00001,
    ):
        """_summary_

        Args:
            state_shape: Shape of the env states
            action_space: number of legal actions in the environment
            replay_memory_size (int, optional): Size of the experience buffer
            frame_stack_size (int, optional): Number of frames to stack together for temporal info
            gamma (float, optional): Discount factor for future rewards
            epsilon (float, optional): Probability to select random action
            epsilon_min (float, optional): Minimum epsilon value
            epsilon_decay (float, optional): Value to decay epsilon by per step
            batch_size (int, optional): Number of experiences to train on
            model_name (str, optional): Name of the directory to store files in
            decay_func (function: float): Function describing how epsilon decays
        """
        self.state_shape = state_shape
        self.action_space = action_space
        self.memory = deque(maxlen=replay_memory_size)
        self.replay_memory_size = replay_memory_size
        self.frame_stack_size = frame_stack_size
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.model_name = model_name
        self.decay_func = decay_func
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        os.makedirs(f"models/{self.model_name}", exist_ok=True)

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
        model.compile(
            loss="mse",
            optimizer=RMSprop(learning_rate=self.learning_rate),
            run_eagerly=False,
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.replay_memory_size:
            self.memory.popleft()
        stacked_state = np.array(state)

        self.memory.append((stacked_state, action, reward, next_state, done))

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.decay_func(self.epsilon, self.epsilon_decay)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = state.astype("float32") / 255.0
        act_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)

        return np.argmax(act_values[0])  # returns action

    def replay(self, num_env):
        if len(self.memory) < self.batch_size * num_env:
            return
        minibatch = random.sample(self.memory, self.batch_size * num_env)
        actual_batch_size = len(minibatch)

        states = np.array([i[0] for i in minibatch]).astype("float32") / 255.0
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch]).astype("float32") / 255.0
        dones = np.array([i[4] for i in minibatch])

        targets = rewards + self.gamma * (
            np.amax(self.target_model.predict_on_batch(next_states), axis=1)
        ) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(actual_batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)

    def save(self, name):
        self.model.save(f"models/{self.model_name}/{name}")

    def load(self, name):
        self.model = tf.keras.models.load_model(f"models/{self.model_name}/{name}")

    def load_other(self, name):
        self.model = tf.keras.models.load_model(f"models/{name}")
        print("loaded")

    def handle_interrupt(self):
        print("Interrupted, saving...")
        self.save("interrupted_model.h5")


class TrainAgent:
    def __init__(
        self,
        env,
        agent,
        num_envs,
        num_eps=1500,
        max_steps=5000000,
        save_rate=100,
        update_freq=3,
        neg_reward=-0.5,
        prog_freq=5000,
    ):
        self.env = env
        self.agent = agent
        self.num_envs = num_envs
        self.num_eps = num_eps
        self.max_steps = max_steps
        self.save_rate = save_rate
        self.update_freq = update_freq
        self.prog_freq = prog_freq
        self.mem_data = []
        self.neg_reward = neg_reward

        def handler(signal, frame):
            self.sigint_handler(signal, frame)

        signal.signal(signal.SIGINT, handler)

    def sigint_handler(self, signal, frame):
        with open(f"models/{self.agent.model_name}/mem_usage.pkl", "wb") as f:
            pickle.dump(self.mem_data, f)
        self.agent.handle_interrupt()
        sys.exit(0)

    def train(self):
        total_score: float
        reward_queue = []
        # for e in range(self.num_eps):
        # reset state in the beginning of each game
        states, infos = self.env.reset()
        states = [
            np.stack([state] * self.agent.frame_stack_size, axis=-1) for state in states
        ]  # initialize the state with the same frame repeated

        """if (e > 0) and (e % self.save_rate == 0):
            self.agent.save(f"benchmark_{e}.h5")
            with open(f"models/{self.agent.model_name}/{e}_rewards.pkl", "wb") as f:
                pickle.dump(reward_queue, f)
            reward_queue = []
            """

        total_score = 0.0
        lives = infos["lives"]
        # time ticks
        for steps in range(self.max_steps):
            # Decide action
            actions = [self.agent.act(state) for state in states]

            # Advance the game to the next frame based on the action.
            next_states, rewards, dones, truncs, infos = self.env.step(actions)

            total_score += np.sum(rewards) / self.num_envs

            next_states = [
                np.concatenate(
                    (state[..., 1:], np.expand_dims(next_state, -1)), axis=-1
                )
                for (state, next_state) in zip(states, next_states)
            ]

            """rewards_tmp = []
            lives_tmp = []
            for next_lives, reward, live in zip(infos["lives"], rewards, lives):
                if next_lives < live:
                    rewards_tmp.append(self.neg_reward)
                    lives_tmp.append(next_lives)
            rewards, lives = rewards_tmp, lives_tmp"""

            # Remember the previous state, action, reward, and done
            for state, action, reward, next_state, done in zip(
                states,
                actions,
                rewards,
                next_states,
                dones,
            ):
                self.agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            states = next_states

            if steps % self.prog_freq == 0:
                # print the score and break out of the loop
                print(
                    "steps: {}/{}, avg score: {}".format(
                        steps * self.num_envs,
                        self.max_steps * self.num_envs,
                        total_score / self.num_envs,
                    )
                )
                reward_queue.append((steps, total_score / self.num_envs))
                total_score = 0.0

            # train the agent with the experience of the episode
            self.agent.replay(self.num_envs)

            # update target model weights every episode
            if self.num_envs % self.update_freq == 0:
                self.agent.update_target_model()
                k.clear_session()
                gc.collect()

            if steps % self.save_rate == 0:
                if os.path.exists(f"models/{self.agent.model_name}/avg_rewards.pkl"):
                    with open(
                        f"models/{self.agent.model_name}/avg_rewards.pkl", "rb"
                    ) as file:
                        existing_data = pickle.load(file)
                else:
                    existing_data = []
                existing_data.extend(reward_queue)
                with open(
                    f"models/{self.agent.model_name}/avg_rewards.pkl", "wb"
                ) as file:
                    pickle.dump(existing_data, file)
                reward_queue = []

        # Save the model after training
        self.agent.save(f"models/{self.agent.model_name}/final_model.h5")

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


class CircularBuffer:
    def __init__(
        self, max_size, frame_shape=(84, 84), stack_size=4, action_space=4, num_envs=12
    ):
        self.max_size = max_size
        self.frame_shape = frame_shape
        self.stack_size = stack_size
        self.action_space = action_space
        self.num_envs = num_envs
        self.states = np.empty((0, frame_shape, stack_size))
        self.actions = np.empty((0, action_space))
        self.rewards = np.empty(0)
        self.next_states = np.empty((0, frame_shape, stack_size))
        self.dones = np.empty(0, dtype=bool)
        self.index_pointer = 0
        self.current_size = 0

    def add_memory(self, states, actions, rewards, next_states, dones):
        new_size = self.current_size + self.num_envs
        num_over = 0
        if new_size > self.max_size:
            num_over = new_size - self.max_size
        else:
            self.current_size += self.num_envs
        self.states = np.vstack(self.states[num_over:], states)
        self.actions = np.vstack(self.actions[num_over:], actions)
        self.rewards = np.vstack(self.rewards[num_over:], rewards)
        self.next_states = np.vstack(self.next_states[num_over:], next_states)
        self.dones = np.vstack(self.dones[num_over:], dones)

    def sample_memory(self, batch_size):
        indices = np.random.choice(self.current_size, batch_size, replace=False)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        return states, actions, rewards, next_states, dones

    def get_size(self):
        return self.current_size
