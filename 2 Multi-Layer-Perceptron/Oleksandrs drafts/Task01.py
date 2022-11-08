import numpy as np
import matplotlib.pyplot as plt 

x = np.random.randint(100, size = (100))
print(x)
t = [i**3 - i**2 for i in x]
print(t)
plt.plot(sorted(x), sorted(t))
plt.show()