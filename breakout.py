import tensorflow as tf

# from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
import line_profiler

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
        num_envs=8,
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
        self.memory = CircularBuffer(self.replay_memory_size, num_envs=num_envs)
        self.writer = tf.summary.create_file_writer(
            f"models/{self.model_name}/logs/weights"
        )
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
                input_shape=[*self.state_shape],
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
            jit_compile=True,
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, states, actions, rewards, next_states, dones):
        self.memory.add_memory(states, actions, rewards, next_states, dones)

        """stacked_state = np.array(state)

        self.memory.append((stacked_state, action, reward, next_state, done))"""

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = np.array(state).astype("float32") / 255.0
        act_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)

        return np.argmax(act_values[0])  # returns action

    @profile
    def act_on_batch(self, states, batch_size):
        states = np.array(states).astype("float32") / 255.0
        rand_values = -np.ones(batch_size)
        random_positions = np.random.rand(batch_size) <= self.epsilon
        rand_values[random_positions] = np.random.randint(
            self.action_space, size=np.sum(random_positions)
        )
        pred_values = np.argmax(self.model.predict_on_batch(states), axis=1)
        pred_values[random_positions] = rand_values[random_positions]

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.decay_func(self.epsilon, self.epsilon_decay)

        return np.array(pred_values, dtype=np.uint8)

    @tf.function(jit_compile=True)
    def replay(self, num_env):
        # if self.memory.get_size() < self.batch_size * num_env:
        #    return None, None

        states, actions, rewards, next_states, dones = self.memory.sample_memory(
            self.batch_size
        )
        # minibatch = random.sample(self.memory, self.batch_size * num_env)
        actual_batch_size = len(states)

        states = tf.convert_to_tensor(states, dtype=tf.float32) / 255.0
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32) / 255.0
        targets = rewards + self.gamma * (
            tf.reduce_max(self.target_model(next_states), axis=1)
        ) * (1 - dones)
        targets_full = self.model(states)

        # ind = np.array([i for i in range(actual_batch_size)])
        ind = tf.range(actual_batch_size, dtype=tf.int64)
        # targets_full[[ind], [actions]] = targets
        indices = tf.stack([ind, actions], axis=-1)
        targets_full = tf.tensor_scatter_nd_update(targets_full, indices, targets)
        # x_train = tf.convert_to_tensor(states)
        # y_train = tf.convert_to_tensor(targets_full)
        # self.model.fit(x_train, y_train, epochs=1, verbose=0)
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            loss = tf.keras.losses.MSE(targets_full, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        return predictions, targets_full

    def record_weights(self, steps):
        with self.writer.as_default():
            for i, layer in enumerate(self.model.layers):
                for j, weight in enumerate(layer.weights):
                    tf.summary.histogram(f"Layer_{i}_Weight_{j}", weight, step=steps)

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
        self.agent: DQNAgent = agent
        self.num_envs = num_envs
        self.num_eps = num_eps
        self.max_steps = max_steps
        self.save_rate = save_rate
        self.update_freq = update_freq
        self.prog_freq = prog_freq
        self.mem_data = []
        self.neg_reward = neg_reward
        self.score_writer = tf.summary.create_file_writer(
            f"models/{self.agent.model_name}/logs/score"
        )
        self.q_writer = tf.summary.create_file_writer(
            f"models/{self.agent.model_name}/logs/q"
        )

        def handler(signal, frame):
            self.sigint_handler(signal, frame)

        signal.signal(signal.SIGINT, handler)

    def sigint_handler(self, signal, frame):
        with open(f"models/{self.agent.model_name}/mem_usage.pkl", "wb") as f:
            pickle.dump(self.mem_data, f)
        self.agent.handle_interrupt()
        sys.exit(0)

    @profile
    def train(self):
        total_score: float
        # reward_queue = []
        states, infos = self.env.reset()

        total_score = 0.0
        lives = np.array(infos["lives"])
        # time ticks
        for steps in range(self.max_steps):
            # Decide action
            actions = self.agent.act_on_batch(states, self.num_envs)

            # Advance the game to the next frame based on the action.
            next_states, rewards, dones, truncs, infos = self.env.step(actions)
            total_score += np.sum(rewards) / self.num_envs

            new_lives = np.array(infos["lives"])
            death_indices = np.where(lives > new_lives)
            lives = new_lives
            dones[death_indices] = True
            rewards[death_indices] = self.neg_reward
            self.agent.remember(states, actions, rewards, next_states, dones)
            # make next_state the new current state for the next frame.
            states = next_states

            if self.agent.memory.get_size() >= self.agent.batch_size * self.num_envs:
                predictions, targets = self.agent.replay(self.num_envs)
            else:
                predictions, targets = None, None

            # update target model weights every episode
            if steps % self.update_freq == 0:
                self.agent.update_target_model()
                # k.clear_session()
                # gc.collect()

            if steps % self.prog_freq == 0:
                # print the score and break out of the loop
                print(
                    "steps: {}/{}, avg score: {}".format(
                        steps * self.num_envs,
                        self.max_steps * self.num_envs,
                        total_score,
                    )
                )
                # reward_queue.append((steps, total_score))

                with self.score_writer.as_default():
                    tf.summary.scalar(
                        f"Avg Score per env over {self.prog_freq} steps",
                        total_score,
                        step=steps,
                    )
                    self.score_writer.flush()
                total_score = 0.0

                if predictions is not None:
                    with self.q_writer.as_default():
                        tf.summary.scalar(
                            "mean_target_q_value", tf.reduce_mean(targets), step=steps
                        )
                        tf.summary.scalar(
                            "std_target_q_value",
                            tf.math.reduce_std(targets),
                            step=steps,
                        )
                        tf.summary.scalar(
                            "mean_predicted_q_value",
                            tf.reduce_mean(predictions),
                            step=steps,
                        )
                        tf.summary.scalar(
                            "std_predicted_q_value",
                            tf.math.reduce_std(predictions),
                            step=steps,
                        )
                        self.q_writer.flush()
            # self.agent.record_weights(steps)
            """
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
                """

        # Save the model after training
        self.agent.save(f"models/{self.agent.model_name}/final_model.h5")

    def play_model(self):
        for e in range(self.num_eps):
            state, info = self.env.reset()
            # state = np.stack([state] * self.agent.frame_stack_size, axis=-1)

            for time in range(self.max_steps):
                self.env.render()

                action = self.agent.act(state)

                next_state, _, done, _, _ = self.env.step(action)
                """
                next_state = np.concatenate(
                    (state[..., 1:], np.expand_dims(next_state, -1)), axis=-1
                )
                """

                state = next_state

                if done:
                    break


class CircularBuffer:
    def __init__(
        self,
        max_size,
        frame_shape=(4, 84, 84),
        stack_size=4,
        action_space=4,
        num_envs=8,
    ):
        self.max_size = max_size
        self.frame_shape = frame_shape
        self.stack_size = stack_size
        self.action_space = action_space
        self.num_envs = num_envs
        self.states = np.empty((self.max_size, *frame_shape), dtype=np.uint8)
        self.actions = np.empty((self.max_size), dtype=np.float32)
        self.rewards = np.zeros((self.max_size), dtype=np.uint8)
        self.next_states = np.empty((self.max_size, *frame_shape), dtype=np.uint8)
        self.dones = np.empty((self.max_size), dtype=bool)
        self.position = 0
        self.current_size = 0

    @profile
    def add_memory(self, states, actions, rewards, next_states, dones):
        size = states.shape[0]  # assuming the first dimension is the batch size
        if self.position + size <= self.max_size:
            # there's enough space at the end of the buffer
            self.states[self.position : self.position + size] = states
            self.actions[self.position : self.position + size] = actions
            self.rewards[self.position : self.position + size] = rewards
            self.next_states[self.position : self.position + size] = next_states
            self.dones[self.position : self.position + size] = dones
        else:
            # the batch spans the end and the beginning of the buffer
            # split the batch and add it in two parts
            split = self.max_size - self.position
            self.states[self.position :] = states[:split]
            self.actions[self.position :] = actions[:split]
            self.rewards[self.position :] = rewards[:split]
            self.next_states[self.position :] = next_states[:split]
            self.dones[self.position :] = dones[:split]
            self.states[: size - split] = states[split:]
            self.actions[: size - split] = actions[split:]
            self.rewards[: size - split] = rewards[split:]
            self.next_states[: size - split] = next_states[split:]
            self.dones[: size - split] = dones[split:]
        self.position = (self.position + size) % self.max_size
        self.current_size = min(self.current_size + size, self.max_size)

    @profile
    def sample_memory(self, batch_size):
        indices = np.random.choice(
            self.current_size, batch_size * self.num_envs, replace=True
        )
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        return states, actions, rewards, next_states, dones

    def get_size(self):
        return self.current_size
