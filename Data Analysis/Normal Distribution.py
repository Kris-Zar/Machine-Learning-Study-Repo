import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

mean= 50
std_dev=9
size=5000
data=np.random.normal(loc=mean, scale=std_dev, size=size)
sb.histplot(data, kde=True, color='green', bins=50, linewidth=1.0)
plt.title(f"Normal Distribution of mean {mean} and standard deviation {std_dev}")
plt.xlabel("Values")
plt.ylabel("Density")
plt.show()