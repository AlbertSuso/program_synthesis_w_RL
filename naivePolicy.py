import torch
import copy
import numpy as np
import matplotlib.pyplot as plt

from LibMyPaint.LibMyPaint import Environment


canvas_width = 64
grid_width = 32
use_color = False
brushes_basedir = '/home/albert/spiral/third_party/mypaint-brushes-1.3.0'
brush_type = 'classic/calligraphy'


environment = Environment(canvas_width, grid_width, brush_type, use_color, brush_sizes=torch.linspace(1.2, 3, 10),
                          use_pressure=False, use_alpha=False, background="transparent", brushes_basedir=brushes_basedir)

action = copy.copy(environment.action_spec)

num_steps = 1

for step in range(num_steps):
    for key in action:
        action[key] = np.random.randint(0, environment.action_spec[key].maximum+1, dtype=np.int32)
    action['flag'] = np.random.randint(1, 2, dtype=np.int32)
    action['size'] = np.random.randint(9, 10, dtype=np.int32)
    print(action)
    print(torch.linspace(1.2, 1.5, 10))
    environment.step(action)

output = environment.observation()["canvas"]
plt.imshow(output, cmap="gray", vmin=0, vmax=1)
plt.show()

print((output == 0).sum())
print((output == 1).sum())

if (output == 0).sum() + (output == 1).sum() != 28*28:
    print("Se acabo")
