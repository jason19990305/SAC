import numpy as np
import matplotlib.pyplot as plt

# Define KL divergence for Bernoulli distributions
def kl_divergence(p, q):
    # Avoid log(0) by clipping values
    p = np.clip(p, 1e-10, 1 - 1e-10)
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

# Create grid for p and q values
p_vals = np.linspace(0.01, 0.99, 100)
q_vals = np.linspace(0.01, 0.99, 100)
P, Q = np.meshgrid(p_vals, q_vals)

# Compute KL(p||q)
KL = kl_divergence(P, Q)

# Plot 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(P, Q, KL, cmap="viridis")

ax.set_xlabel("p")
ax.set_ylabel("q")
ax.set_zlabel("KL(p||q)")
ax.set_title("KL Divergence between Bernoulli(p) and Bernoulli(q)")

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
