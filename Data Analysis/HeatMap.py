import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Test1.csv")
plt.colorbar(plt.imshow(data["Age"], cmap='viridis', interpolation='nearest'))
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Heatmap Example")
plt.show()