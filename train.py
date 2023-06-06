from breakout import DQNAgent, TrainAgent
import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStack,
    RecordEpisodeStatistics,
    VectorListInfo,
    RecordVideo,
)
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
    PROFILE,
    MIN_SAMPLE_SIZE,
    VIDEO_FREQ,
    STICKY_ACTION_PROBABILITY,
)


def wrap_env(env):
    env = AtariPreprocessing(env, noop_max=10)
    env = FrameStack(env, 4)
    """
    env = RecordVideo(
        env, f"models/{MODEL_NAME}/videos", step_trigger=lambda x: x % VIDEO_FREQ == 0
    )
    """

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
envs = gym.vector.make(
    "BreakoutNoFrameskip-v4",
    NUM_ENVS,
    wrappers=wrap_env,
    repeat_action_probability=STICKY_ACTION_PROBABILITY,
)
envs = RecordEpisodeStatistics(envs, deque_size=PROGRESS_FREQ)

video = gym.make(
    "BreakoutNoFrameskip-v4",
    render_mode="rgb_array",
    repeat_action_probability=STICKY_ACTION_PROBABILITY,
)
video = AtariPreprocessing(video, noop_max=0)
video = FrameStack(video, 4)
video = RecordVideo(video, f"models/{MODEL_NAME}/videos", step_trigger=lambda x: True)
# envs = VectorListInfo(envs)

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
    min_sample_size=MIN_SAMPLE_SIZE,
    video_env=video,
    video_freq=VIDEO_FREQ,
)
if PROFILE:
    lp = LineProfiler()
    lp_wrapper = lp(classroom.train)
    lp_wrapper()

    lp.print_stats()
else:
    classroom.train()
