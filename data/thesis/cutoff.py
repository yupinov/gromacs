import numpy as np
import matplotlib.pyplot as plt
import math

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0.1, 5, 0.01)
y = 1 / x
beta = 1
ydir = [math.erf(beta * i) / i for i in x]
yrec = [math.erfc(beta * i) / i for i in x]

# Plot the points using matplotlib
plt.plot(x, y, label='Total')
plt.plot(x, ydir, label='Direct')
plt.plot(x, yrec, label='Reciprocal')
plt.xlabel('r (distance)')
plt.title('Ewald decomposition')
plt.legend()
#plt.show()  # You must call plt.show() to make graphics appear.

plt.savefig('pics/cutoff'+str(beta)+'.png')

