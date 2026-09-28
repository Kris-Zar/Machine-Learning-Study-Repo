import numpy as np
from scipy import stats

before=np.array([120, 122, 118, 130, 125, 128, 115, 121, 123, 119])
after=np.array([115, 120, 112, 128, 122, 125, 110, 117, 119, 114])

alpha = 0.05

t_stat, p_value = stats.ttest_rel(before, after) 
m = np.mean(after - before)
s = np.std(after - before, ddof=1)
n = len(before)
t_manual = m / (s / np.sqrt(n))

decision = "Reject" if p_value <= alpha else "Fail reject"
concl = "Significant difference." if decision == "Reject" else "No significant difference."

print("T:", t_stat)
print("P:", p_value)
print("T manual:", t_manual)
print(f"Decision: {decision} H0 at α={alpha}")
print("Conclusion:", concl)