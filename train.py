from breakout import DQNAgent, TrainAgent
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, RecordEpisodeStatistics
from config import MODEL_NAME, SAVE_RATE, UPDATE_FREQ, BATCH_SIZE, EPSILON_DECAY

env = gym.make("BreakoutNoFrameskip-v4")
env = AtariPreprocessing(env, noop_max=10)
env = RecordEpisodeStatistics(env)

print(env.observation_space)

agent = DQNAgent(
    env.observation_space.shape,
    env.action_space.n,
    epsilon_decay=EPSILON_DECAY,
    batch_size=BATCH_SIZE,
    model_name=MODEL_NAME,
)

classroom = TrainAgent(env, agent, save_rate=SAVE_RATE, update_freq=UPDATE_FREQ)

classroom.train()