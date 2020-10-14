a = float(1558)
b = float(312)
c = float(311)
print(a%b==c)


i = 1558
batch_size = 32


print(i % int((50000//batch_size)//5) == int((50000//batch_size)//5 - 1))