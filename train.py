from breakout import DQNAgent, TrainAgent
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from line_profiler import LineProfiler
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
    NUM_ENVS,
    PROGRESS_FREQ,
    MAX_STEPS,
)


def wrap_env(env):
    env = AtariPreprocessing(env, noop_max=20)
    env = FrameStack(env, 4)
    # env = RecordEpisodeStatistics(env)
    return env


"""
env_list = []
for _ in range(NUM_ENVS):
    tmp = gym.make("BreakoutNoFrameskip-v4")
    tmp = wrap_env(tmp)
    env_list.append(lambda: tmp)
env = gym.make("BreakoutNoFrameskip-v4")
envs = gym.vector.AsyncVectorEnv(
    env_list,
    shared_memory=False,
)
"""
envs = gym.vector.make("BreakoutNoFrameskip-v4", NUM_ENVS, wrappers=wrap_env)

print(envs.single_observation_space.shape)
print(f"Training model {MODEL_NAME}")
agent = DQNAgent(
    envs.single_observation_space.shape,
    envs.single_action_space.n,
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
    envs,
    agent,
    NUM_ENVS,
    num_eps=NUM_EPS,
    save_rate=SAVE_RATE,
    update_freq=UPDATE_FREQ,
    neg_reward=NEGATIVE_REWARD,
    prog_freq=PROGRESS_FREQ,
    max_steps=MAX_STEPS,
)

lp = LineProfiler()
lp_wrapper = lp(classroom.train)
lp_wrapper()
# classroom.train()
lp.print_stats()
