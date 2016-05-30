import numpy as np
import matplotlib.pyplot as plt
import math

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0.01, 5, 0.01)
y = 1 / x
beta = 2
ydir = [math.erfc(beta * i) / i for i in x]
yrec = [math.erf(beta * i) / i for i in x]

plt.rcParams.update({'font.size': 18})

# Plot the points using matplotlib
plt.plot(x, y, label='Total (1 / r)')
plt.plot(x, ydir, label='Direct (erfc('+str(beta)+' * r) / r)')
plt.plot(x, yrec, label='Reciprocal (erf('+str(beta)+' * r) / r)')
plt.xlabel('r (distance)')
plt.xlim(xmin=0, xmax=4)
plt.ylim(ymin=0, ymax=4)
#plt.ylabel('r (distance)')
# plt.title('Dependency')
plt.legend()
#plt.show()  # You must call plt.show() to make graphics appear.

plt.savefig('pics/cutoff'+str(beta)+'.png')

