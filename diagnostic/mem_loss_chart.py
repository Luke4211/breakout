import pickle
import numpy as np
import matplotlib.pyplot as plt

MODEL_NAME = "maydar"
STEP_CUTOFF = 0
# Load and unpickle the array
with open(f"models/{MODEL_NAME}/mem_usage.pkl", "rb") as f:
    loaded_array = pickle.load(f)

# Convert the loaded array to a NumPy array
loaded_array = np.array(loaded_array)

new_arr = np.array([i for i in loaded_array if i[0] > STEP_CUTOFF])
x_1 = new_arr[0]
x_2 = new_arr[-1]

elapsed_steps = x_2[0] - x_1[0]
mem_increase = x_2[1] - x_1[1]

print(
    f"Memory usage per step after {STEP_CUTOFF} steps: {(mem_increase/elapsed_steps)*1024}kb "
)

print(len(new_arr))
# Plot memory usage vs steps
plt.plot(new_arr[:, 0], new_arr[:, 1])
# plt.plot(loaded_array[:, 0], loaded_array[:, 1])
plt.xlabel("Steps")
plt.ylabel("Memory Usage (MB)")
plt.title(f"Memory Usage Over Steps {(mem_increase/elapsed_steps)*1024}kb/step")
plt.grid(True)
plt.show()
