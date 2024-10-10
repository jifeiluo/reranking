import random
import torch
import time
import numpy as np
import warnings
import torch.nn.functional as F
import torch
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from reranker.distance import compute_distance_matrix, euclidean_squared_distance, cosine_distance
from torch.cuda.amp import autocast


def all_pairs_shortest_paths(adjacency_matrix: torch.Tensor, max_threshold=50, device=torch.device('cuda')) -> torch.Tensor:
    vertex_num = adjacency_matrix.shape[0]
    assert adjacency_matrix.shape[0]==adjacency_matrix.shape[1]

    if device == torch.device('cuda'):
        adjacency_matrix = adjacency_matrix.to(device)
        adjacency_matrix[torch.arange(vertex_num), torch.arange(vertex_num)] = 0
        adjacency_matrix = torch.minimum(adjacency_matrix, adjacency_matrix.t())
        adjacency_matrix = torch.clamp(adjacency_matrix, min=0.0, max=max_threshold)
        distance_matrix = torch.clone(adjacency_matrix)
        try:
            from reranker.geodesic_distance import geodesic_distance
        except ImportError:
            for k in range(vertex_num):
                k_distance = distance_matrix[k].repeat(vertex_num,1) + distance_matrix[:,k].repeat(vertex_num, 1).transpose(0,1)
                distance_matrix = torch.minimum(distance_matrix, k_distance)
        else:
            distance_matrix = geodesic_distance.square_bellman_ford(5, adjacency_matrix)
            
        distance_matrix = torch.clamp(distance_matrix, min=0.0, max=max_threshold)
        return distance_matrix

    elif device == torch.device('cpu'):
        adjacency_matrix[torch.arange(vertex_num), torch.arange(vertex_num)] = 0
        adjacency_matrix = torch.minimum(adjacency_matrix, adjacency_matrix.t())
        adjacency_matrix = torch.clamp(adjacency_matrix, min=0.0, max=max_threshold)
        distance_matrix = torch.clone(adjacency_matrix)
        for k in range(vertex_num):
            k_distance = distance_matrix[k].repeat(vertex_num,1) + distance_matrix[:,k].repeat(vertex_num, 1).transpose(0,1)
            distance_matrix = torch.minimum(distance_matrix, k_distance)
        distance_matrix = torch.clamp(distance_matrix, min=0.0, max=max_threshold)
        return distance_matrix


def transport_cost(all_num, original_dist, initial_rank, topk=4, max_threshold=50, confident_matrix=None, device=torch.device('cuda')):
    original_dist = original_dist.cpu()

    adjacency_connection = torch.zeros((all_num, all_num))
    adjacency_matrix = torch.zeros((all_num, all_num))
    if confident_matrix is not None:
        adjacency_connection = adjacency_connection + confident_matrix
        adjacency_connection[torch.repeat_interleave(torch.arange(all_num), topk), initial_rank[:,:topk].reshape(-1)] = 1
    else:
        adjacency_connection[torch.repeat_interleave(torch.arange(all_num), topk), initial_rank[:,:topk].reshape(-1)] = 1
    # adjacency_matrix = adjacency_connection*original_dist + (1-adjacency_connection)*max_threshold # implementation 1
    mutual_adjacency_connection = adjacency_connection * adjacency_connection.t()
    single_adjacency_connection = adjacency_connection + adjacency_connection.t() - 2*mutual_adjacency_connection
    adjacency_matrix = (mutual_adjacency_connection+single_adjacency_connection)*original_dist + (1-mutual_adjacency_connection-single_adjacency_connection)*max_threshold 

    print('Computing All Pairs of Shortest Paths')
    start_time = time.time()
    cost_matrix = all_pairs_shortest_paths(adjacency_matrix, max_threshold=max_threshold, device=device)
    end_time = time.time()
    print(f'===> Time cost {end_time-start_time}')

    return cost_matrix



def similarity_diffusion(all_num, confident_matrix, similarity_matrix, initial_rank, 
                  knn=10, mu=0.23, max_iter=10, lambda_value=2, threshold=1e-2, device=torch.device('cuda')):
    alpha = 1/(1+mu)

    # Initialize Weight Matrix
    W_matrix = torch.zeros((all_num, all_num), device=device)
    W_matrix[torch.repeat_interleave(torch.arange(all_num), knn-1), initial_rank[:,1:knn].reshape(-1)] = \
        similarity_matrix[torch.repeat_interleave(torch.arange(all_num), knn-1), initial_rank[:,1:knn].reshape(-1)]
    # W_matrix[torch.repeat_interleave(torch.arange(all_num), knn), initial_rank[:,:knn].reshape(-1)] = \
    #     similarity_matrix[torch.repeat_interleave(torch.arange(all_num), knn), initial_rank[:,:knn].reshape(-1)]
    # Enhance the weights of confident candidates
    W_matrix.add_(torch.multiply(confident_matrix, W_matrix), alpha=lambda_value-1)
    # Symmetry Operation
    W_matrix = (W_matrix + W_matrix.t()) / 2

    d_vector = torch.sqrt(torch.sum(W_matrix, dim=1, keepdim=True)).reciprocal_()
    D_matrix = torch.mul(d_vector, d_vector.t())
    S_matrix = torch.multiply(W_matrix, D_matrix)
    S_matrix = (S_matrix + S_matrix.t()) / 2

    E_matrix = torch.eye(all_num, device=device)# + S_matrix
    F_initial = torch.clone(W_matrix)
    F_matrix = torch.clone(F_initial)

    # Basic Iteration
    for iter in range(max_iter):
        # F_matrix = alpha*torch.mm(torch.mm(S_matrix, F_matrix), S_matrix.t()) + (1-alpha)*E_matrix
        F_matrix = 0.5*alpha*(torch.mm(F_matrix, S_matrix)+torch.mm(S_matrix, F_matrix)) + (1-alpha)*E_matrix

    return F_matrix

def bidirectional_similarity_diffusion(all_num, confident_matrix, similarity_matrix, initial_rank, 
                  knn_list=[10,15,20,25], mu=0.1, max_iter=10, lambda_value=2, lv=1000, threshold=1e-2, device=torch.device('cuda')):
    W_matrix_list = []
    S_matrix_list = []
    for knn in knn_list:
        W_matrix = torch.zeros((all_num, all_num)).to(device)
        W_matrix[torch.repeat_interleave(torch.arange(all_num), knn), initial_rank[:,:knn].reshape(-1)] = \
            similarity_matrix[torch.repeat_interleave(torch.arange(all_num), knn), initial_rank[:,:knn].reshape(-1)]
        W_matrix = (W_matrix + W_matrix.t()) / 2
        W_matrix_list.append(W_matrix)

        d_vector = torch.sqrt(torch.sum(W_matrix, dim=1, keepdim=True)).reciprocal()
        D_matrix = torch.mul(d_vector, d_vector.t()).to(device)
        S_matrix = torch.multiply(W_matrix, D_matrix)
        S_matrix = (S_matrix + S_matrix.t()) / 2
        S_matrix_list.append(S_matrix)
        
    E_matrix = torch.eye(all_num).to(device)
    # for i in range(len(knn_list)):
    #     E_matrix += S_matrix_list[i]/len(knn_list)
    F_matrix = torch.eye(all_num).to(device)

    beta_list = torch.ones(len(knn_list))/ len(knn_list)
    tmp_beta_list = torch.ones(len(knn_list)) / len(knn_list)
    objective_value_list = torch.ones(len(knn_list)).to(device)
    for multi_scale_iter in range(3):
        alpha_list = beta_list / (mu/len(knn_list)+torch.sum(beta_list))
        # alpha_list = alpha_list.to(device)
        for iter in range(max_iter):
            tmp =  torch.zeros((all_num, all_num)).to(device)
            for i in range(len(knn_list)):
                tmp += alpha_list[i]*(0.5*torch.mm(S_matrix_list[i], F_matrix) + 0.5*torch.mm(F_matrix, S_matrix_list[i].t()) - E_matrix)
            F_matrix = tmp + E_matrix

        for i in range(len(knn_list)):
            objective_value_list[i] = torch.mul(F_matrix, F_matrix).sum() - \
                    0.5*torch.mul(F_matrix, torch.mm(F_matrix, S_matrix_list[i].t())+torch.mm(S_matrix_list[i], F_matrix)).sum()
            
        print(objective_value_list)
        for i in range(200):
            idx1, idx2 = random.randint(0, len(knn_list)-1), random.randint(0, len(knn_list)-1)
            tmp_beta_list[idx1] = (lv*(beta_list[idx1]+beta_list[idx2])+objective_value_list[idx2]-objective_value_list[idx1])/(2*lv)
            tmp_beta_list[idx1] = torch.clamp(tmp_beta_list[idx1], min=0, max=beta_list[idx1]+beta_list[idx2])
            tmp_beta_list[idx2] = beta_list[idx1] + beta_list[idx2] - tmp_beta_list[idx1]
            beta_list = tmp_beta_list.clone()
            
    return F_matrix


def markov_expansion(all_num, sparse_feature, initial_rank, 
                     k1=6, k2=20, iterations=1, lambda_value=2, confident_matrix=None, device=torch.device('cuda')):
    sparse_feature = sparse_feature.to(device=device)
    mixed_feature = torch.zeros((all_num, all_num), device=device)

    print('Applying Markov Sparse Feature Expansion')
    start_time = time.time()
    if confident_matrix is not None:
        # confident matrix can be used to enhance the weight of the most confident neighbors when applying query expansion
        # otherwise, average the features according to the initial ranking list
        confident_matrix = confident_matrix.to(device=device)
        for i in range(k1):
            mixed_feature.add_(sparse_feature[initial_rank[:,i],:])
        mixed_feature = mixed_feature / k1
        mixed_feature = (mixed_feature + lambda_value*torch.mm(F.normalize(confident_matrix, p=1, dim=1), sparse_feature)) / (1 + lambda_value)
    else:
        for i in range(k1):
            mixed_feature.add_(sparse_feature[initial_rank[:,i],:])
        mmixed_feature = mixed_feature / k1
    mixed_feature = F.normalize(mixed_feature, p=1, dim=1)

    # expand_sparse_feature = mixed_feature
    for iter in range(iterations):
        markov_feature = torch.mm(mixed_feature, mixed_feature)
        markov_rank = torch.argsort(markov_feature, descending=True)
        truncate_markov_feature = torch.zeros((all_num, all_num), device=device)
        truncate_markov_feature[torch.repeat_interleave(torch.arange(all_num), k2), markov_rank[:,:k2].reshape(-1)] = \
            markov_feature[torch.repeat_interleave(torch.arange(all_num), k2), markov_rank[:,:k2].reshape(-1)]
        truncate_markov_feature = F.normalize(truncate_markov_feature, p=1, dim=1)
        mixed_feature = mixed_feature + lambda_value*truncate_markov_feature

    expand_sparse_feature = torch.mm(truncate_markov_feature, truncate_markov_feature)
    expand_sparse_feature = F.normalize(expand_sparse_feature, p=1, dim=1)

    end_time = time.time()
    print(f'===> Time cost {end_time-start_time}')
    # expand_sparse_feature = expand_sparse_feature.to(torch.device('cpu')).numpy()

    return expand_sparse_feature


def confident_candidate_expansion(all_num, initial_rank, k=5, device=torch.device('cuda')):
    '''
    Constructing the Confident Candidate Matrix
    '''
    confident_matrix = torch.zeros((all_num, all_num), device=device)
    confident_matrix[torch.repeat_interleave(torch.arange(all_num), k), initial_rank[:,:k].reshape(-1)] = 1
    confident_matrix = torch.multiply(confident_matrix, torch.transpose(confident_matrix, dim0=0, dim1=1))
    
    return confident_matrix


def candidate_expansion(all_num, initial_rank, k=20, neighbor_threshold=0, device=torch.device('cuda')):
    '''
    Constructing the Candidate Matrix
    '''
    candidate_matrix = torch.zeros((all_num, all_num), device=device)
    expansion_matrix = torch.zeros((all_num, all_num), device=device)
    print('Applying Candidate Expansion')
    start_time = time.time()
    # Initialize method 1
    # for i in range(all_num):
    #     candidate_matrix[i, initial_rank[i,:k+1]] = 1
    #     expansion_matrix[i, initial_rank[i,:int(np.around(k/2.))+1]] = 1
    # Initialize method 2
    # candidate_matrix[np.repeat(np.arange(all_num), k+1), initial_rank[:,:k+1].reshape(-1)] = 1
    # expansion_matrix[np.repeat(np.arange(all_num), int(np.around(k/2.0))+1), initial_rank[:,:int(np.around(k/2.0))+1].reshape(-1)] = 1
    candidate_matrix[torch.repeat_interleave(torch.arange(all_num), k+1), initial_rank[:,:k+1].reshape(-1)] = 1
    expansion_matrix[torch.repeat_interleave(torch.arange(all_num), k//2+1), initial_rank[:,:k//2+1].reshape(-1)] = 1
    candidate_matrix = torch.multiply(candidate_matrix, torch.transpose(candidate_matrix, dim0=0, dim1=1))
    expansion_matrix = torch.multiply(expansion_matrix, torch.transpose(expansion_matrix, dim0=0, dim1=1))
    init_candidate_matrix = torch.clone(candidate_matrix)

    for i in range(k+1):
        mask_expansion_matrix = expansion_matrix[initial_rank[:,i],:]
        mask_expansion_sum = torch.sum(mask_expansion_matrix, dim=1)
        intersect_matrix = torch.multiply(mask_expansion_matrix, init_candidate_matrix)
        intersect_sum = torch.sum(intersect_matrix, dim=1)
        intersect_sum = torch.multiply(intersect_sum, init_candidate_matrix[torch.arange(all_num), initial_rank[:,i]]) # set the values not exist in the initial candidate matrix to 0
        mask = torch.where(intersect_sum > 2/3*mask_expansion_sum)[0]
        candidate_matrix[mask] = candidate_matrix[mask] + mask_expansion_matrix[mask] - torch.multiply(candidate_matrix[mask], mask_expansion_matrix[mask])
    # Deal with the situations where there are only a few candidate neighbors
    candidate_num = torch.sum(candidate_matrix, dim=1)
    mask = torch.where(candidate_num<=neighbor_threshold)[0]
    initial_rank = initial_rank.to(device=device)
    candidate_matrix[torch.repeat_interleave(mask, k), initial_rank[mask, :k].reshape(-1)] = 1

    end_time = time.time()
    print(f'===> Time cost {end_time-start_time}')
    # candidate_matrix = candidate_matrix.to(torch.device('cpu')).numpy()
    
    return candidate_matrix


def sparse_feature_construction(all_num, original_dist, initial_rank, 
                                confident_k=5, candidate_k=20, trans_k=20, sigma=0.4, beta=0, lambda_value=2, mu=0.23,
                                device=torch.device('cuda')):
    similarity_matrix = torch.exp(-original_dist/sigma**2)
    sparse_feature = torch.zeros((all_num, all_num), device=device)
    confident_matrix = confident_candidate_expansion(all_num, initial_rank, k=confident_k)
    candidate_matrix = candidate_expansion(all_num, initial_rank, k=candidate_k)
    semi_candidate_matrix = candidate_matrix - confident_matrix

    print('Applying Similarity Diffusion')
    start_time = time.time()
    # F_matrix = similarity_matrix
    # F_matrix = similarity_diffusion(all_num, confident_matrix, similarity_matrix, initial_rank, knn=2*confident_k) # MAC, R-MAC
    F_matrix = similarity_diffusion(all_num, confident_matrix, similarity_matrix, initial_rank, knn=confident_k, lambda_value=lambda_value, mu=mu) # R-GeM
    end_time = time.time()
    print(f'===> Time cost {end_time-start_time}')
    sparse_feature = torch.multiply(candidate_matrix, F_matrix)
    sparse_feature = F.normalize(sparse_feature, p=1, dim=1)
    # sparse_feature = sparse_feature.to(torch.device('cpu'))

    print('Applying Feature Augmentation or Smoothing')
    start_time = time.time()
    trans_matrix = torch.zeros((all_num, all_num), device=device)
    trans_matrix[torch.repeat_interleave(torch.arange(all_num), trans_k-1), initial_rank[:,1:trans_k].reshape(-1)] =\
            similarity_matrix[torch.repeat_interleave(torch.arange(all_num), trans_k-1), initial_rank[:,1:trans_k].reshape(-1)]
    trans_matrix = F.normalize(trans_matrix, p=1, dim=1)
    trans_matrix = torch.mm(trans_matrix, trans_matrix) # self-enhancement

    if device==torch.device('cpu'):
        transition_weight_matrix = torch.zeros((all_num, all_num), device=device)
        for i in range(all_num):
            confident_non_zero_num, confident_non_zero_index = torch.count_nonzero(confident_matrix[i]), torch.nonzero(confident_matrix[i], as_tuple=True)[0]
            candidate_non_zero_num, candidate_non_zero_index = torch.count_nonzero(candidate_matrix[i]), torch.nonzero(candidate_matrix[i], as_tuple=True)[0]
            semi_candidate_non_zero_num, semi_candidate_non_zero_index = torch.count_nonzero(semi_candidate_matrix[i]), torch.nonzero(semi_candidate_matrix[i], as_tuple=True)[0]
            
            if confident_non_zero_num>1 and semi_candidate_non_zero_num>1:
                anchor = torch.mean(trans_matrix[confident_non_zero_index, confident_non_zero_index])
                # anchor = torch.mean(trans_matrix[confident_non_zero_index][:,confident_non_zero_index])
                transition_weight_matrix[i, confident_non_zero_index] = anchor
                transition_weight_matrix[i, semi_candidate_non_zero_index] = torch.clamp_max(torch.mean(trans_matrix[semi_candidate_non_zero_index][:,confident_non_zero_index], dim=1), max=anchor)        
                transition_weight = transition_weight_matrix[i, candidate_non_zero_index]
                tmp_value = (anchor**2)*torch.mean(F_matrix[i, candidate_non_zero_index]) - anchor*torch.mean(F_matrix[i, candidate_non_zero_index]*transition_weight)
                sparse_feature[i, candidate_non_zero_index] = (F_matrix[i, candidate_non_zero_index]*(anchor*transition_weight+2*beta) + tmp_value) / (anchor**2+2*beta)
            else:
                sparse_feature[i, candidate_non_zero_index] = F_matrix[i, candidate_non_zero_index]

    elif device==torch.device('cuda'):
        from reranker.feature_operations import feature_operations
        confident_non_zero_all_index = torch.nonzero(confident_matrix, as_tuple=True)[1]
        confident_non_zero_num = torch.count_nonzero(confident_matrix, dim=1)
        candidate_non_zero_num = torch.count_nonzero(candidate_matrix, dim=1)
        semi_candidate_non_zero_num = torch.count_nonzero(semi_candidate_matrix, dim=1)
        confident_non_zero_start_num = torch.cumsum(confident_non_zero_num, dim=0) - confident_non_zero_num
        confident_non_zero_end_num = confident_non_zero_start_num + confident_non_zero_num
        semi_candidate_non_zero_start_num = torch.cumsum(semi_candidate_non_zero_num, dim=0) - semi_candidate_non_zero_num
        semi_candidate_non_zero_end_num = semi_candidate_non_zero_start_num + semi_candidate_non_zero_num

        anchor_vector = feature_operations.compute_anchor_vector(trans_matrix, 
                                                                confident_non_zero_start_num, 
                                                                confident_non_zero_end_num, 
                                                                confident_non_zero_all_index,
                                                                1)
        transition_weight_matrix = feature_operations.compute_transition_weight_matrix(trans_matrix, 
                                                                                        confident_non_zero_start_num,
                                                                                        confident_non_zero_end_num,
                                                                                        confident_non_zero_all_index,
                                                                                        semi_candidate_non_zero_start_num,
                                                                                        semi_candidate_non_zero_end_num,
                                                                                        confident_matrix,
                                                                                        semi_candidate_matrix,
                                                                                        anchor_vector)
        sparse_feature = torch.mul(candidate_matrix, F_matrix)
        tmp_value_vector = (torch.pow(anchor_vector, 2)*torch.sum(sparse_feature, dim=1) - anchor_vector*torch.sum(torch.mul(sparse_feature, transition_weight_matrix), dim=1)) / candidate_non_zero_num
        sparse_feature = torch.mul(candidate_matrix, (sparse_feature*(torch.mul(transition_weight_matrix, anchor_vector.unsqueeze(1))+2*beta) + tmp_value_vector.unsqueeze(1)) / (torch.pow(anchor_vector, 2).unsqueeze(1)+2*beta))
    
    sparse_feature = F.normalize(sparse_feature, p=1, dim=1)
    end_time = time.time()
    print(f'===> Time cost {end_time-start_time}')

    momentum = sparse_feature - F.normalize(torch.mul(candidate_matrix, F_matrix), p=1, dim=1)

    return sparse_feature, confident_matrix, candidate_matrix, F_matrix, momentum


def cas_reranking(query_features, 
                    gallery_features, 
                    metric = 'euclidean', 
                    mode = 'normal', 
                    k1=5, 
                    k2=20, 
                    k3=25, 
                    k4=6, 
                    k5=25, 
                    lambda_value=0.3, 
                    lv=2, 
                    sigma=0.3, 
                    mu=0.23,
                    beta=0,
                    baryweight=0.1,
                    device=torch.device('cuda'),
                    mask=None):
    """Computes the reranking distance.

    Args:
        query_features (torch.Tensor): 2-D feature matrix.
        gallery_features (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: reranking distance matrix.
    """
    # Statistical information 
    query_num = query_features.shape[0]
    gallery_num = gallery_features.shape[0]
    all_num = query_num + gallery_num
    features = np.concatenate((query_features, gallery_features), axis=0).astype(np.float32)
    features = torch.from_numpy(features).to(device)

    original_dist = compute_distance_matrix(features, features, metric)
    original_dist = torch.div(original_dist, torch.max(original_dist, dim=1)[0].unsqueeze(1))
    initial_rank = torch.argsort(original_dist)

    sparse_feature, confident_matrix, candidate_matrix, F_matrix, momentum = sparse_feature_construction(all_num, original_dist, initial_rank, confident_k=k1, candidate_k=k2, trans_k=k3, beta=beta, sigma=sigma, mu=mu)
    
    refine_cost = torch.ones(all_num, all_num)
    refine_cost[torch.arange(all_num), torch.arange(all_num)] = 0
    sparse_feature = markov_expansion(all_num, sparse_feature, initial_rank, k1=k4, k2=k5, confident_matrix=confident_matrix, lambda_value=lv)

    ## method 1
    # modified_dist = compute_distance_matrix(sparse_feature, sparse_feature, 'jensen shannon', device=torch.device('cuda'))
    ## method 2
    adjacency_matrix = 50*torch.ones((all_num, all_num)).to('cuda')
    modified_dist = compute_distance_matrix(sparse_feature, sparse_feature, 'jensen shannon', device=torch.device('cuda'))
    adjacency_matrix[torch.repeat_interleave(torch.arange(all_num), 100), initial_rank[:,:100].reshape(-1)] = modified_dist[torch.repeat_interleave(torch.arange(all_num), 100), initial_rank[:,:100].reshape(-1)]
    modified_dist = all_pairs_shortest_paths(adjacency_matrix, max_threshold=50, device=torch.device('cuda'))
    
    # modified_dist = compute_distance_matrix(sparse_feature, sparse_feature, 'cosine').to(torch.device('cuda'))
    # modified_dist = compute_distance_matrix(sparse_feature, sparse_feature, 'euclidean').to(torch.device('cuda'))

    final_dist = modified_dist*(1-lambda_value) + original_dist*lambda_value
    final_dist = final_dist[:query_num,query_num:]
    final_dist = final_dist.to(torch.device('cpu')).numpy()

    return final_dist
