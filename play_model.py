from breakout import DQNAgent, TrainAgent
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from config import MODEL_NAME

env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
env = AtariPreprocessing(env, noop_max=20)
env = FrameStack(env, 4)

print(env.observation_space)

agent = DQNAgent(
    env.observation_space.shape,
    env.action_space.n,
    batch_size=32,
    model_name=MODEL_NAME,
)

classroom = TrainAgent(env, agent, num_envs=1)

# classroom.train()

agent.load("interrupted_model.h5")
agent.epsilon = 0.01
agent.update_target_model()

classroom.play_model()
