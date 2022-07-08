import numpy as np

with open('mol0.aux') as f:
    lines = f.readlines()

energies = []

for i in range(8,len(lines),10):
        energies.append(float(lines[i].split()[0]))

np.savetxt('text.txt',np.array(energies))
