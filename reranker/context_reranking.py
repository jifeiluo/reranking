import torch
import numpy as np
import torch.nn.functional as F
from reranker.distance import compute_distance_matrix


def kreciprocal_reranking(query_features: np.ndarray, gallery_features: np.ndarray, metric='euclidean', k1=20, k2=6, lambda_value=0.3) -> np.ndarray:
    """Computes modified distance matrix with k-reciprocal reranking method.

    Args:
        query_features (numpy.ndarray): 2-D feature matrix.
        gallery_features (numpy.ndarray): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine". Default is "euclidean".
        k1 (int): A hyperparameter. Default is 20.
        k1 (int): A hyperparameter. Default is 6.
        lambda_value (float): A hyperparameter. Default is 0.3.

    Returns:
        numpy.ndarray: modified distance matrix.  
    """
    # print('Applying k-reciprocal Re-ranking')
    query_num  = query_features.shape[0]
    gallery_num = gallery_features.shape[0]
    all_num = query_num + gallery_num
    features = np.concatenate((query_features.astype(np.float32), gallery_features.astype(np.float32)), axis=0)
    original_dist = compute_distance_matrix(features, features, metric).numpy()

    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist

    final_dist = final_dist[:query_num,query_num:]
    return final_dist


def gnn_reranking(query_features: np.ndarray, gallery_features: np.ndarray, metric='euclidean', k1=40, k2=10, alpha=2.0) -> np.ndarray:
    """Computes modified score matrix with gnn reranking method.

    Args:
        query_features (numpy.ndarray): 2-D feature matrix.
        gallery_features (numpy.ndarray): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine". Default is "euclidean".
        k1 (int): A hyperparameter. Default is 40.
        k1 (int): A hyperparameter. Default is 10.
        alpha (float): A hyperparameter. Default is 2.0.

    Returns:
        numpy.ndarray: modified score matrix.  
    """
    # print('Applying GNN Re-ranking')
    query_num = query_features.shape[0]
    gallery_num = gallery_features.shape[0]
    all_num = query_num + gallery_num

    query_features = torch.from_numpy(query_features).float()
    gallery_features = torch.from_numpy(gallery_features).float()
    all_features = torch.cat((query_features, gallery_features), dim=0)
    norm_features = F.normalize(all_features, p=2, dim=1)
    original_score = torch.mm(norm_features, norm_features.t())
    original_cosine_similarity = torch.mm(query_features, gallery_features.T)
    
    # initial ranking list
    similarity_matrix, initial_rank = original_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    initial_rank = initial_rank.numpy()
    # Stage 1
    similarity_matrix = torch.pow(similarity_matrix, alpha)
    A_matrix = torch.zeros((all_num, all_num))
    for i in range(all_num):
        A_matrix[i, initial_rank[i]] = 1.0

    # Stage 2
    if k2 != 1:
        for i in range(2):
            A_matrix = (A_matrix + A_matrix.T)/2
            tmp_A_matrix = A_matrix.clone()

            for j in range(all_num):
                A_matrix[j] = tmp_A_matrix[j] + torch.mm(similarity_matrix[j,:k2].reshape(1,-1), tmp_A_matrix[initial_rank[j,:k2]])
            A_matrix = F.normalize(A_matrix, p=2, dim=1)
            # A_norm = torch.norm(A_matrix, p=2, dim=1, keepdim=True)
            # A_matrix = A_matrix.div(A_norm.expand_as(A_matrix)) 
    cosine_similarity = torch.mm(A_matrix[:query_num,], A_matrix[query_num:, ].t())
    scores = cosine_similarity.numpy()

    return scores


def stml_reranking(query_features: np.ndarray, gallery_features: np.ndarray, metric='euclidean', k=20) -> np.ndarray:
    """Computes modified score matrix with STML reranking method.

    Args:
        query_features (numpy.ndarray): 2-D feature matrix.
        gallery_features (numpy.ndarray): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine". Default is "euclidean".
        k (int): A hyperparameter. Default is 20.

    Returns:
        numpy.ndarray: modified score matrix.  
    """
    # print('Applying STML Re-ranking')
    query_num  = query_features.shape[0]
    gallery_num = gallery_features.shape[0]
    all_num = query_num + gallery_num
    features = np.concatenate((query_features.astype(np.float32), gallery_features.astype(np.float32)), axis=0)
    original_dist = compute_distance_matrix(features, features, metric).numpy()

    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    sigma = 0.1
    weight_P = np.exp(-original_dist/sigma)
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        V[i,k_reciprocal_index] = 1

    weight_C = np.dot(V, V.T) / np.sum(V, axis=1, keepdims=True)

    weight_C_2 = np.zeros_like(V,dtype=np.float32)
    for i in range(all_num):
        weight_C_2[i,:] = np.mean(weight_C[initial_rank[i,:k//2],:],axis=0)
    weight = (weight_C_2 + weight_C_2.T)/2
    
    final_weight = (weight_P+weight)/2
    final_weight = final_weight[:query_num,query_num:]
    scores = final_weight

    return scores


def knn_reranking(query_features: np.ndarray, gallery_features: np.ndarray, k=50) -> np.ndarray:
    """Computes modified distance matrix with kNN reranking method.

    Args:
        query_features (numpy.ndarray): 2-D feature matrix.
        gallery_features (numpy.ndarray): 2-D feature matrix.
        k (int): A hyperparameter. Default is 50.

    Returns:
        numpy.ndarray: modified distance matrix.  
    """
    # print('Applying KNN Re-ranking')
    query_num  = query_features.shape[0]
    gallery_num = gallery_features.shape[0]
    all_num = query_num + gallery_num

    features = np.concatenate((query_features.astype(np.float32), gallery_features.astype(np.float32)), axis=0)
    original_dist = compute_distance_matrix(features, features, 'euclidean').numpy()
    initial_rank = np.argsort(original_dist).astype(np.int32)
    rank_matrix = np.argsort(initial_rank).astype(np.float32) + 1

    knn_dist = np.zeros((query_num, all_num))
    for i in range(query_num):
        for j in range(k+1):
            knn_dist[i] = knn_dist[i] + 1/((j + 1 + rank_matrix[initial_rank[i,j],i])*rank_matrix[initial_rank[i,j]])
    final_dist = -knn_dist[:query_num,query_num:]

    return final_dist