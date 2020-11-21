import torch
import time
from torch.optim import Adam

t0 = time.time()
a = torch.zeros((10000, 10000))
for i in range(10000):
    b = torch.rand(10000)
    a[i] = b
print("Tiempo por filas:", time.time()-t0)

t0 = time.time()
a = torch.zeros((10000, 10000))
for i in range(10000):
    b = torch.rand(10000)
    a[:, i] = b
print("Tiempo por columnas:", time.time()-t0)



print("FIN DE PRUEBA")


print("Empezando !!!!!!!!!!")

t0 = time.time()

a = torch.nn.Sequential(torch.nn.Linear(50, 30), torch.nn.ReLU(), torch.nn.Linear(30, 10),
                        torch.nn.ReLU(), torch.nn.Linear(10, 1)).cuda(0)
optimizer_a = Adam(a.parameters(), lr=0.001)

b = torch.nn.Sequential(torch.nn.Linear(50, 30), torch.nn.ReLU(), torch.nn.Linear(30, 10),
                        torch.nn.ReLU(), torch.nn.Linear(10, 1)).cuda(1)
optimizer_b = Adam(b.parameters(), lr=0.001)

print("El tiempo de inicializacion es de", time.time()-t0)
print("El numero de GPU's disponibles es", torch.cuda.device_count())

K = 1000000
k = 0
loss_1 = float('inf')
loss_2 = float('inf')

t0 = time.time()
while loss_1+loss_2>0.01 and k < K:
    k += 1

    if k%10000 == 9999:
        print(k,loss_a+loss_b)

    a.zero_grad()
    b.zero_grad()

    inputs = torch.rand((64, 50))
    input_a = inputs.cuda(0)
    input_b = inputs.cuda(1)

    out_a = a(input_a)
    out_b = b(input_b)

    target = torch.sin(torch.sum(inputs, dim=1)).view(-1, 1)

    detached_out_a = out_a.cpu().cuda(1).detach()
    loss_b = torch.nn.functional.mse_loss(out_b, detached_out_a)
    loss_b.backward()
    optimizer_b.step()


    loss_a = torch.nn.functional.mse_loss(out_a, target.cuda(0))
    loss_a.backward()
    optimizer_a.step()

    loss_1 = loss_a.cpu()
    loss_2 = loss_b.cpu()

print("DONE!", k, loss_a, loss_b, loss_a+loss_b)
print("TIME=", time.time()-t0)
