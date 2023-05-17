import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, RecordEpisodeStatistics
import psutil
import pickle

env = gym.make("BreakoutNoFrameskip-v4")


env = AtariPreprocessing(env, noop_max=10)
env = RecordEpisodeStatistics(env)

mem_data = []
process = psutil.Process()


def get_memory(proc):
    memory_info = proc.memory_info()
    memory_usage_mb = memory_info.rss / (1024**2)
    return memory_usage_mb


for step in range(10000):
    print(f"Memory: {get_memory(process)}")
    mem_data.append((step, get_memory(process)))
    env.reset()

with open("mem_usage_env_only.pkl", "wb") as f:
    pickle.dump(mem_data, f)
