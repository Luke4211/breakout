from breakout import DQNAgent, TrainAgent
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, RecordEpisodeStatistics
from config import MODEL_NAME

env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
env = AtariPreprocessing(env, noop_max=2)
env = RecordEpisodeStatistics(env)

print(env.observation_space)

agent = DQNAgent(
    env.observation_space.shape,
    env.action_space.n,
    batch_size=64,
    model_name=MODEL_NAME,
)

classroom = TrainAgent(env, agent)

# classroom.train()

agent.load("benchmark_500.h5")
agent.epsilon = 0.2
agent.update_target_model()

classroom.play_model()
