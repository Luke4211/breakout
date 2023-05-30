from breakout import DQNAgent, TrainAgent
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, RecordEpisodeStatistics
from config import (
    MODEL_NAME,
    SAVE_RATE,
    UPDATE_FREQ,
    BATCH_SIZE,
    EPSILON_START,
    EPSILON_DECAY,
    GAMMA,
    EPSILON_MIN,
    NUM_EPS,
    MEMORY_SIZE,
    DECAY_FUNC,
    NEGATIVE_REWARD,
    LOAD_MODEL,
    LOAD_MODEL_NAME,
    LEARNING_RATE,
)

env = gym.make("BreakoutNoFrameskip-v4")
env = AtariPreprocessing(env, noop_max=10)
env = RecordEpisodeStatistics(env)

print(env.observation_space)
print(f"Training model {MODEL_NAME}")
agent = DQNAgent(
    env.observation_space.shape,
    env.action_space.n,
    replay_memory_size=MEMORY_SIZE,
    gamma=GAMMA,
    epsilon=EPSILON_START,
    epsilon_min=EPSILON_MIN,
    epsilon_decay=EPSILON_DECAY,
    batch_size=BATCH_SIZE,
    model_name=MODEL_NAME,
    decay_func=DECAY_FUNC,
    learning_rate=LEARNING_RATE,
)

if LOAD_MODEL:
    agent.load_other(LOAD_MODEL_NAME)

classroom = TrainAgent(
    env,
    agent,
    num_eps=NUM_EPS,
    save_rate=SAVE_RATE,
    update_freq=UPDATE_FREQ,
    neg_reward=NEGATIVE_REWARD,
)

classroom.train()
