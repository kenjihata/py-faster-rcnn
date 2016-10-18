import sys
import matplotlib
import matplotlib.pyplot as plt

filename = sys.argv[1]

with open(filename) as f:
    lines = f.read().splitlines()
loss = []
for line in lines:
    idx = line.find(" loss = ") 
    if idx > -1:
        loss.append(float(line[idx+8:]))
print len(loss)
plt.plot(loss)
plt.show()
