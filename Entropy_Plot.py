import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# create raw x,y grid
N = 300
x = np.linspace(0, 1.0, N)
y = np.linspace(0, 1.0, N)
X, Y = np.meshgrid(x, y)

P1 = X
P2 = Y

# Entropy
eps = 1e-12
H = -(P1 * np.log(np.clip(P1, eps, 1.0)) + 
      P2 * np.log(np.clip(P2, eps, 1.0)))

# 3D surface
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(P1, P2, H, cmap='viridis', edgecolor='none')

ax.set_xlabel("p1")
ax.set_ylabel("p2")
ax.set_zlabel("Entropy H (nats)")
ax.set_title("Entropy Surface for 2-D Probability Distribution")
plt.show()
