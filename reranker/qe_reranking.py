import torch
import torch.nn as nn
import numpy as np
from reranker.distance import compute_distance_matrix
from sklearn.preprocessing import normalize
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


def adba(gallery_features, kdba=4):
    similarity = np.dot(gallery_features, gallery_features.T)
    ranks = np.argsort(-similarity, axis=1)
    update_gallery_features = np.zeros(gallery_features.shape, dtype=np.float32)
    for i in range(gallery_features.shape[0]):
        sum_feature = np.sum(gallery_features[ranks[i,:kdba]], axis=0)
        update_gallery_features[i] = normalize(sum_feature[:, np.newaxis], axis=0).ravel()

    return update_gallery_features

def aqe_reranking(query_features: np.ndarray, gallery_features: np.ndarray, kqe=4, DBA=False, kdba=4) -> np.ndarray:
    """Computes modified score matrix with AQE reranking method.

    Args:
        query_features (numpy.ndarray): 2-D feature matrix.
        gallery_features (numpy.ndarray): 2-D feature matrix.
        kqe (int): A hyperparameter. Default is 4.
        DBA (bool): Use database augmentation. Default is False.
        kdba (int): A hyperparameter. Default is 4.

    Returns:
        numpy.ndarray: modified score matrix.

    Examples:
        >>> kqe = 2
        >>> kqe = 10
        >>> kqe, kdba = 4, 4
    """
    similarity = np.dot(query_features, gallery_features.T)
    ranks = np.argsort(-similarity, axis=1)
    update_query_features = np.zeros(query_features.shape, dtype=np.float32)

    for i in range(query_features.shape[0]):
        all_feature = np.concatenate((query_features[i].reshape(1, -1), gallery_features[ranks[i,:kqe]]), axis=0)
        sum_feature = np.sum(all_feature, axis=0)
        update_query_features[i] = normalize(sum_feature[:, np.newaxis], axis=0).ravel()

    if DBA==True:
        update_gallery_features = adba(gallery_features, kdba)
    else:
        update_gallery_features = gallery_features
    scores = np.dot(update_query_features, update_gallery_features.T)

    return scores


def adba_wd(gallery_features, kdba):
    similarity = np.dot(gallery_features, gallery_features.T)
    ranks = np.argsort(-similarity, axis=1)
    update_gallery_features = np.zeros(gallery_features.shape, dtype=np.float32)
    for i in range(gallery_features.shape[0]):
        weight = ((kdba - np.arange(kdba)) / (kdba)).reshape(-1, 1)
        sum_feature = np.sum(gallery_features[ranks[i,:kdba]]*weight, axis=0)
        update_gallery_features[i] = normalize(sum_feature[:, np.newaxis], axis=0).ravel()

    return update_gallery_features

def aqewd_reranking(query_features: np.ndarray, gallery_features: np.ndarray, kqe=8, DBA=False, kdba=4) -> np.ndarray:
    """Computes modified score matrix with AQEwD reranking method.

    Args:
        query_features (numpy.ndarray): 2-D feature matrix.
        gallery_features (numpy.ndarray): 2-D feature matrix.
        kqe (int): A hyperparameter. Default is 48.
        alpha (float): A hyperparameter. Default is 3.0.
        DBA (bool): Use database augmentation. Default is False.
        kdba (int): A hyperparameter. Default is 32.

    Returns:
        numpy.ndarray: modified score matrix.

    Examples:
        >>> kqe = 4
        >>> kqe = 6
        >>> kqe, kdba = 6, 4
    """
    similarity = np.dot(query_features, gallery_features.T)
    ranks = np.argsort(-similarity, axis=1)
    update_query_features = np.zeros(query_features.shape, dtype=np.float32)

    for i in range(query_features.shape[0]):
        all_feature = np.concatenate((query_features[i].reshape(1, -1), gallery_features[ranks[i,:kqe]]), axis=0)
        weight = ((kqe+1 - np.arange(kqe+1)) / (kqe+1)).reshape(-1, 1)
        sum_feature = np.sum(all_feature*weight, axis=0)
        update_query_features[i] = normalize(sum_feature[:, np.newaxis], axis=0).ravel()

    if DBA==True:
        update_gallery_features = adba_wd(gallery_features, kdba)
    else:
        update_gallery_features = gallery_features
    scores = np.dot(update_query_features, update_gallery_features.T)

    return scores


def alpha_dba(gallery_features, alpha=3, kdba=32):
    similarity = np.dot(gallery_features, gallery_features.T)
    ranks = np.argsort(-similarity, axis=1)
    update_gallery_features = np.zeros(gallery_features.shape, dtype=np.float32)
    for i in range(gallery_features.shape[0]):
        weight = (np.dot(gallery_features[i], gallery_features[ranks[i,:kdba]].T) ** alpha).reshape(-1, 1)
        sum_feature = np.sum(gallery_features[ranks[i,:kdba]]*weight, axis=0)
        update_gallery_features[i] = normalize(sum_feature[:, np.newaxis], axis=0).ravel()

    return update_gallery_features

def alpha_qe_reranking(query_features: np.ndarray, gallery_features:np.ndarray, kqe=48, alpha=3.0, DBA=False, kdba=32) -> np.ndarray:
    """Computes modified score matrix with alphaQE reranking method.

    Args:
        query_features (numpy.ndarray): 2-D feature matrix.
        gallery_features (numpy.ndarray): 2-D feature matrix.
        kqe (int): A hyperparameter. Default is 48.
        alpha (float): A hyperparameter. Default is 3.0.
        DBA (bool): Use database augmentation. Default is False.
        kdba (int): A hyperparameter. Default is 32.

    Returns:
        numpy.ndarray: modified score matrix.

    Examples:
        >>> kqe = 48
        >>> kqe, kdba = 48, 10
        >>> kqe, kdba = 32, 32
    """
    similarity = np.dot(query_features, gallery_features.T)
    ranks = np.argsort(-similarity, axis=1)
    update_query_features = np.zeros(query_features.shape, dtype=np.float32)

    for i in range(query_features.shape[0]):
        all_feature = np.concatenate((query_features[i].reshape(1, -1), gallery_features[ranks[i,:kqe]]), axis=0)
        weight = (np.dot(query_features[i], all_feature.T) ** alpha).reshape(-1, 1)
        sum_feature = np.sum(all_feature*weight, axis=0)
        update_query_features[i] = normalize(sum_feature[:, np.newaxis], axis=0).ravel()

    if DBA==True:
        update_gallery_features = alpha_dba(gallery_features, alpha, kdba)
    else:
        update_gallery_features = gallery_features
    scores = np.dot(update_query_features, update_gallery_features.T)

    return scores


def dual_solve(X, y):
    #Initializing values and computing H. Note the 1. to force to float type
    m,n = X.shape
    y = y.reshape(-1,1) * 1.
    X_dash = y * X
    H = np.dot(X_dash , X_dash.T) * 1.

    #Converting into cvxopt format
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    #Setting solver parameters (change default to decrease tolerance)
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10

    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas

def ddba(gallery_features, kdba=10, dbneg=10):
    similarity = np.dot(gallery_features, gallery_features.T)
    ranks = np.argsort(-similarity, axis=1)
    update_gallery_features = np.zeros(gallery_features.shape, dtype=np.float32)
    y = np.array([1.] * kdba + [-1.] * dbneg)
    for i in range(gallery_features.shape[0]):
        all_feature = np.concatenate((gallery_features[ranks[i,:kdba]], gallery_features[ranks[i, -dbneg:]]), axis=0)
        weight = (dual_solve(all_feature, y).flatten() * y).reshape(-1, 1)
        sum_feature = np.sum(all_feature*weight, axis=0)
        update_gallery_features[i] = normalize(sum_feature[:, np.newaxis], axis=0).ravel()

    return update_gallery_features

def dqe_reranking(query_features: np.ndarray, gallery_features: np.ndarray, kqe=20, neg=20, DBA=False, kdba=4, dbneg=4) -> np.ndarray:
    """Computes modified score matrix with DQE reranking method.

    Args:
        query_features (numpy.ndarray): 2-D feature matrix.
        gallery_features (numpy.ndarray): 2-D feature matrix.
        kqe (int): A hyperparameter. Default is 20.
        neg (int): A hyperparameter. Default is 20.
        DBA (bool): Use database augmentation. Default is False.
        kdba (int): A hyperparameter. Default is 4.
        dbneg (int): A hyperparameter. Default is 4.
    
    Returns:
        numpy.ndarray: modified score matrix.

    Examples:
        >>> kqe, neg = 2, 4
        >>> kqe, neg, kdba, dbneg = 3, 10, 18, 18
        >>> kqe, neg, kdba, dbneg = 2, 4, 4, 4
        >>> kqe, neg = 20, 20
        >>> kqe, neg, kdba, dbneg = 20, 20, 18, 18
    """
    similarity = np.dot(query_features, gallery_features.T)
    ranks = np.argsort(-similarity, axis=1)
    update_query_features = np.zeros(query_features.shape, dtype=np.float32)
    np.argsort
    y = np.array([1.] * (kqe+1) + [-1.] * neg)
    for i in range(query_features.shape[0]):
        all_feature = np.concatenate((query_features[i].reshape(1, -1), gallery_features[ranks[i,:kqe]], gallery_features[ranks[i, -neg:]]), axis=0)
        weight = (dual_solve(all_feature, y).flatten() * y).reshape(-1, 1)
        sum_feature = np.sum(all_feature*weight, axis=0)
        update_query_features[i] = normalize(sum_feature[:, np.newaxis], axis=0).ravel()

    if DBA==True:
        update_gallery_features = ddba(gallery_features, kdba, dbneg)
    else:
        update_gallery_features = gallery_features
    scores = np.dot(update_query_features, update_gallery_features.T)

    return scores


class MDescAug(nn.Module):
    """ Top-M Descriptor Augmentation"""
    def __init__(self, M=400, K=9, beta=0.15):
        super(MDescAug, self).__init__()
        self.M = M
        self.K = K + 1 # including oneself
        self.beta = beta

    def forward(self, X, Q, ranks):

        #ranks = torch.argsort(-sim, axis=0) # 6322 70
        
        ranks_trans_1000 = torch.transpose(ranks,1,0)[:,:self.M] # 70 400 

        X_tensor1 = torch.tensor(X[ranks_trans_1000]).cuda()
        
        res_ie = torch.einsum('abc,adc->abd',
                X_tensor1,X_tensor1) # 70 400 400

        res_ie_ranks = torch.unsqueeze(torch.argsort(-res_ie.clone(), axis=-1)[:,:,:self.K],-1) # 70 400 10 1
        res_ie_ranks_value = torch.unsqueeze(-torch.sort(-res_ie.clone(), axis=-1)[0][:,:,:self.K],-1) # 70 400 10 1
        res_ie_ranks_value = res_ie_ranks_value
        res_ie_ranks_value[:,:,1:,:] *= self.beta
        res_ie_ranks_value[:,:,0:1,:] = 1.
        res_ie_ranks = torch.squeeze(res_ie_ranks,-1) # 70 400 10
        x_dba = X[ranks_trans_1000] # 70 1 400 2048
        
        x_dba_list = []
        for i,j in zip(res_ie_ranks,x_dba):
            # we should avoid for-loop in python, 
            # thus even make the numbers in paper look nicer, 
            # but i just want to go to bed.
            # i 400 10 j # 400 2048
            x_dba_list.append(j[i])
        
        x_dba = torch.stack(x_dba_list,0) # 70 400 10 2048
        
        x_dba = torch.sum(x_dba * res_ie_ranks_value, 2) / torch.sum(res_ie_ranks_value,2) # 70 400 2048
        res_top1000_dba = torch.einsum('ac,adc->ad', Q, x_dba) # 70 400 
 
        ranks_trans_1000_pre = torch.argsort(-res_top1000_dba,-1) # 70 400
        rerank_dba_final = []
        for i in range(ranks_trans_1000_pre.shape[0]):
            temp_concat = ranks_trans_1000[i][ranks_trans_1000_pre[i]]
            rerank_dba_final.append(temp_concat) # 400

        return rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba

class RerankwMDA(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, M=400, K=9, beta=0.15):
        super(RerankwMDA, self).__init__()
        self.M = M 
        self.K = K + 1 # including oneself
        self.beta = beta

    def forward(self, ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba):

        ranks_trans_1000 = torch.stack(rerank_dba_final,0) # 70 400
        ranks_value_trans_1000 = -torch.sort(-res_top1000_dba,-1)[0] # 70 400

        ranks_trans = torch.unsqueeze(ranks_trans_1000_pre[:,:self.K],-1) # 70 10 1
        ranks_value_trans = torch.unsqueeze(ranks_value_trans_1000[:,:self.K].clone(),-1) # 70 10 1
        ranks_value_trans[:,:,:] *=self.beta
        
        X1 =torch.take_along_dim(x_dba, ranks_trans,1) # 70 10 2048
        X2 =torch.take_along_dim(x_dba, torch.unsqueeze(ranks_trans_1000_pre,-1),1) # 70 400 2048
        X1 = torch.max(X1, 1, True)[0] # 70 1 2048
        res_rerank = torch.sum(torch.einsum(
            'abc,adc->abd',X1,X2),1) # 70 400
        
        res_rerank = (ranks_value_trans_1000 + res_rerank) / 2. # 70 400
        res_rerank_ranks = torch.argsort(-res_rerank, axis=-1) # 70 400
        
        rerank_qe_final = []
        ranks_transpose = torch.transpose(ranks,1,0)[:,self.M:] # 70 6322-400
        for i in range(res_rerank_ranks.shape[0]):
            temp_concat = torch.concat([ranks_trans_1000[i][res_rerank_ranks[i]],ranks_transpose[i]],0)
            rerank_qe_final.append(temp_concat) # 6322
        ranks = torch.transpose(torch.stack(rerank_qe_final,0),1,0) # 70 6322
        
        return ranks
    
def super_global_reranking(query_features: np.ndarray, gallery_features: np.ndarray, M=400, K=9, beta=0.15) -> np.ndarray:
    """Computes modified sorting matrix with super global reranking method.

    Args:
        query_features (numpy.ndarray): 2-D feature matrix.
        gallery_features (numpy.ndarray): 2-D feature matrix.
        M (int): A hyperparameter. Default is 400.
        K (int): A hyperparameter. Default is 9.
        beta (float): A hyperparameter. Default is 0.15.

    Returns:
        numpy.ndarray: modified sorting matrix, can be sorted again.
    """
    MDescAug_obj = MDescAug(M, K, beta)
    RerankwMDA_obj = RerankwMDA(M, K, beta)
    Q = torch.from_numpy(query_features).cuda()
    X = torch.from_numpy(gallery_features).cuda()
    sim = torch.matmul(X, Q.T) # 6322 70
    ranks = torch.argsort(-sim, axis=0) # 6322 70
    rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = MDescAug_obj(X, Q, ranks)
    ranks = RerankwMDA_obj(ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba)
    ranks = ranks.data.cpu().numpy()

    return ranks