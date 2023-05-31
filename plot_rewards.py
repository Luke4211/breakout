import pickle
import numpy as np
import matplotlib.pyplot as plt
from config import MODEL_NAME, PROGRESS_FREQ, NUM_ENVS

window_size = 50

data_list = []

# Load and unpickle the array


with open(f"models/{MODEL_NAME}/avg_rewards.pkl", "rb") as f:
    data_list = pickle.load(f)
data_list = np.array(data_list)
x, y = zip(*data_list)
x = [i * NUM_ENVS for i in x]


plt.figure()

# plt.plot(np.ravel(data_list), "b-", alpha=0.5)


smoothed = np.convolve(
    np.ravel(data_list), np.ones(window_size) / window_size, mode="valid"
)
plt.plot(x, y, "r-")

plt.title(f"Avg. Reward per {PROGRESS_FREQ} steps.")
plt.xlabel("Steps")
plt.ylabel("Avg. Reward")
plt.savefig(f"models/{MODEL_NAME}/reward_chart.png")
