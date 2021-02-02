import numpy as np

a = np.random.uniform(0.0, 100.0, (1000, 1000))
b = np.random.normal(size = (1000,1000))
np.savetxt('a.dat', a, fmt = "%f")
np.savetxt('b.dat', b, fmt = "%f")
