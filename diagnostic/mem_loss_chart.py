import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load and unpickle the array
with open('mem_usage.pkl', 'rb') as f:
    loaded_array = pickle.load(f)

# Convert the loaded array to a NumPy array
loaded_array = np.array(loaded_array)

# Plot memory usage vs steps
plt.plot(loaded_array[:, 0], loaded_array[:, 1])
plt.xlabel('Steps')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Over Steps')
plt.grid(True)
plt.show()