import torch
import numpy as np
import matrix_utils

num = 10
original_matrix = torch.rand(num, num).to('cuda')
print(original_matrix.dtype)

indexs = torch.tensor([0,3,5,7,8], device='cuda')
target_matrix1 = original_matrix[indexs]
print(target_matrix1)

indexs = torch.tensor([0,3,5,7,8], device='cuda')
# target_matrix2 = torch.zeros((5,num), device='cuda')
print(indexs.dtype)
target_matrix2 = matrix_utils.rowwise_deepcopy(original_matrix, indexs)
print(target_matrix2)


num = 5
original_matrix = torch.rand(num, num).to('cuda')
print(original_matrix)
vector = torch.tensor([1,2,3,4,5], dtype=torch.float32).to('cuda')
target_matrix = matrix_utils.rowwise_multiply(original_matrix, vector)
print(target_matrix)