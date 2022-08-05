import torch
import random
from time import time
import torch.backends.cudnn as cudnn

custom_seed = 0

potter_indexies = []
for i in range(custom_seed,custom_seed+5):
    random.seed(i)
    potter_indexies.append(random.sample(range(10),10))
potter_indexies = torch.tensor(potter_indexies, device= 'cuda')

def potter(input, seed_tensor):
    input_len = len(input)
    randomized_input = torch.empty(size=seed_tensor.shape)
    for i, index in zip(range(seed_tensor.shape[0]),seed_tensor):
        randomized_input[i,:] = input[index]

    return randomized_input

A = torch.randperm(10, device= 'cuda')

# @torch.jit.script
# def potter_jit(input, stack_size : int):
#     input_len = len(input)
#     indexies = torch.randperm(input_len,device= 'cuda').unsqueeze(0)
#     randomized_input = torch.tensor([],device= 'cuda')
#     for _ in range(stack_size):
#         randomized_input = torch.cat((randomized_input, input[indexies]),1)

#     return randomized_input

start = time()
B = potter(A, potter_indexies)
print('potter cal time : ', time() - start)
print(B)

# start = time()
# C = potter(A,2)
# print('potter_jit cal time : ', time() - start)
# print(C)