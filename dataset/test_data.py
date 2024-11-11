import torch
import numpy as np
from torch_geometric.utils import to_dense_adj

a = np.load('speed_0.1/circle_low_res_test/circle_low_res_in_domain_2e-3_50.npy', allow_pickle=True)
edge_attr_A = a[0]['edge_attr'][:, 1]
A = to_dense_adj(a[0]['edge_index'], edge_attr=edge_attr_A)
A = A.squeeze(0)

print(A)

lhs = a[0]['lhs']
rhs = a[0]['rhs']

print(lhs.shape)
print(A @ lhs)
print(rhs)
print('no dirichlet max',  torch.abs(A @ lhs - rhs).max())
print('no dirichlet argmax',  torch.abs(A @ lhs - rhs).argmax())
print('no dirichlet mean',  torch.abs(A @ lhs - rhs).mean())


dirichlet_mask = a[0]['x'][:, 5].long()
dirichlet_node = torch.nonzero(dirichlet_mask)
A[dirichlet_node] = 0
A[:,dirichlet_node] = 0

lhs[dirichlet_mask] = 0
rhs[dirichlet_mask] = 0


print('dirichlet max',  torch.abs(A @ lhs - rhs).max())
print('dirichlet argmax',  torch.abs(A @ lhs - rhs).argmax())
print('dirichlet mean',  torch.abs(A @ lhs - rhs).mean())

