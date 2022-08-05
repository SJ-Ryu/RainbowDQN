import random
import torch

potter_indexies = []
for i in range(5):
    random.seed(i)
    potter_indexies.append(random.sample(range(10),10))

print(potter_indexies)