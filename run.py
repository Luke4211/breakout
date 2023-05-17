from breakout import DQNAgent, TrainAgent
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, RecordEpisodeStatistics

env = gym.make("BreakoutNoFrameskip-v4")

env = AtariPreprocessing(env, noop_max=10)
env = RecordEpisodeStatistics(env)
print(env.observation_space)
agent = DQNAgent(env.observation_space.shape, env.action_space.n, batch_size=64)

classroom = TrainAgent(env, agent)

# classroom.train()

agent.load("benchmark_300.h5")
agent.epsilon = 0.5
agent.update_target_model()

classroom.play_model()
