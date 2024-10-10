import torch
import geodesic_distance
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

num = 10
graph = np.random.rand(num, num).astype(np.float32)
for i in range(num):
    graph[i][i] = 0
graph = (graph+graph.T)/2
adjacency_matrix = graph
graph = csr_matrix(graph)
dist_matrix1 = floyd_warshall(csgraph=graph, directed=False).astype(np.float32)
# print(dist_matrix1)


adjacency_matrix = torch.from_numpy(adjacency_matrix).to(torch.device('cuda'))
dist_matrix2 = geodesic_distance.bellman_ford(20, adjacency_matrix)
dist_matrix2 = dist_matrix2.to(torch.device('cpu')).numpy()
# print(dist_matrix2)
print(np.sum(dist_matrix1-dist_matrix2))