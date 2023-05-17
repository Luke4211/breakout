import pickle
import numpy as np
import matplotlib.pyplot as plt

num_files = 5
window_size = 15
model_name = "alpha"

data_list = []

file_names = [f"models/{model_name}/{i*100}_rewards.pkl" for i in range(1, num_files)]
# Load and unpickle the array

for file in file_names:
    with open(file, "rb") as f:
        data = pickle.load(f)
        data_list.append(data)
data_list = np.array(data_list)

data_list = data_list.reshape((len(file_names) * 100, 1))

plt.figure()

plt.plot(np.ravel(data_list), "b-", alpha=0.5)


smoothed = np.convolve(
    np.ravel(data_list), np.ones(window_size) / window_size, mode="valid"
)
plt.plot(smoothed, "r-")

plt.title("Cumulative Rewards")
plt.xlabel("Episode")
plt.ylabel("Value")
plt.show()
