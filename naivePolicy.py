import torch
import copy
import numpy as np
import matplotlib.pyplot as plt

from LibMyPaint.LibMyPaint import Environment


canvas_width = 28
grid_width = 14
use_color = False
brushes_basedir = '/home/albert/spiral/third_party/mypaint-brushes-1.3.0'
brush_type = 'classic/calligraphy'


environment = Environment(canvas_width, grid_width, brush_type, use_color, brush_sizes=torch.linspace(1, 3, 20),
                          use_pressure=True, use_alpha=False, background="white", brushes_basedir=brushes_basedir)

action = copy.copy(environment.action_spec)

num_steps = 3
for step in range(num_steps):
    for key in action:
        action[key] = np.random.randint(0, environment.action_spec[key].maximum+1, dtype=np.int32)
    environment.step(action)

output = environment.observation()["canvas"]
plt.imshow(output, cmap="gray", vmin=0, vmax=1)
plt.show()

print((output == 0).sum() + (output == 1).sum() == 28*28)