import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])

x_grid, y_grid = np.meshgrid(a, b, indexing='ij')

print(x_grid[2, 3], y_grid[2, 3])

u = np.linspace(0, 1, 11)
print(u)

fig = plt.figure(1, figsize=(16, 8))

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(x1_grid, x2_grid, f(x1_grid, x2_grid), cmap=cm.coolwarm, alpha=0.5)
ax.plot(X1, X2, f(X1, X2), marker='.', color='k')
ax.text(x1, x2, x3, "texto que queremos que aparezca en este punto en el gráfico", color='k')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.contour(x1_grid, x2_grid, f(x1_grid, x2_grid), 100, cmap=cm.coolwarm)
ax.plot(X1, X2, f(X1, X2), marker='.', color='k')

