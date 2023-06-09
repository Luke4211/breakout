import pickle
import numpy as np
import matplotlib.pyplot as plt

window_size = 50
model_name = "pfizer"

data_list = []

# Load and unpickle the array


with open(f"models/{model_name}/avg_rewards.pkl", "rb") as f:
    data_list = pickle.load(f)
    print(data_list)
data_list = np.array(data_list)
x, y = zip(*data_list)


plt.figure()

# plt.plot(np.ravel(data_list), "b-", alpha=0.5)


smoothed = np.convolve(
    np.ravel(data_list), np.ones(window_size) / window_size, mode="valid"
)
plt.plot(x, y, "r-")

plt.title("Cumulative Rewards")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.savefig(f"models/{model_name}/reward_chart.png")
