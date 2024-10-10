import torch
import faiss
import scipy
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from tqdm import tqdm
from reranker.distance import compute_distance_matrix

class BaseKNN(object):
    """KNN base class"""
    def __init__(self, database, method):
        if database.dtype != np.float32:
            database = database.astype(np.float32)
        self.N = len(database)
        self.D = database[0].shape[-1]
        self.database = database if database.flags['C_CONTIGUOUS'] \
                               else np.ascontiguousarray(database)

    def add(self, batch_size=10000):
        """Add data into index"""
        if self.N <= batch_size:
            self.index.add(self.database)
        else:
            [self.index.add(self.database[i:i+batch_size])
                    for i in tqdm(range(0, len(self.database), batch_size),
                                  desc='[index] add')]

    def search(self, queries, k):
        """Search
        Args:
            queries: query vectors
            k: get top-k results
        Returns:
            sims: similarities of k-NN
            ids: indexes of k-NN
        """
        if not queries.flags['C_CONTIGUOUS']:
            queries = np.ascontiguousarray(queries)
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        sims, ids = self.index.search(queries, k)
        return sims, ids


class KNN(BaseKNN):
    """KNN class
    Args:
        database: feature vectors in database
        method: distance metric
    """
    def __init__(self, database, method):
        super().__init__(database, method)
        self.index = {'cosine': faiss.IndexFlatIP,
                      'euclidean': faiss.IndexFlatL2}[method](self.D)
        self.add()

def get_affinity(sims, ids, gamma=3):
    """Create affinity matrix for the mutual kNN graph of the whole dataset
    Args:
        sims: similarities of kNN
        ids: indexes of kNN
    Returns:
        affinity: affinity matrix
    """
    num = sims.shape[0]
    sims[sims < 0] = 0  # similarity should be non-negative
    sims = sims ** gamma
    # vec_ids: feature vectors' ids
    # mut_ids: mutual (reciprocal) nearest neighbors' ids
    # mut_sims: similarites between feature vectors and their mutual nearest neighbors
    vec_ids, mut_ids, mut_sims = [], [], []
    for i in range(num):
        # check reciprocity: i is in j's kNN and j is in i's kNN when i != j
        ismutual = np.isin(ids[ids[i]], i).any(axis=1)
        if np.isin(ids[i],i).any():
            ismutual[np.where(ids[i]==i)[0]] = False
        if ismutual.any():
            vec_ids.append(i * np.ones(ismutual.sum(), dtype=int))
            mut_ids.append(ids[i, ismutual])
            mut_sims.append(sims[i, ismutual])
    vec_ids, mut_ids, mut_sims = map(np.concatenate, [vec_ids, mut_ids, mut_sims])
    affinity = sparse.csc_matrix((mut_sims, (vec_ids, mut_ids)),
                                    shape=(num, num), dtype=np.float32)
    return affinity

def get_laplacian(sims, ids, alpha=0.99, gamma=3):
    """Get Laplacian_alpha matrix
    """
    affinity = get_affinity(sims, ids, gamma=gamma)
    num = affinity.shape[0]
    degrees = affinity @ np.ones(num) + 1e-12
    # mat: degree matrix ^ (-1/2)
    mat = sparse.dia_matrix(
        (degrees ** (-0.5), [0]), shape=(num, num), dtype=np.float32)
    stochastic = mat @ affinity @ mat
    sparse_eye = sparse.dia_matrix(
        (np.ones(num), [0]), shape=(num, num), dtype=np.float32)
    lap_alpha = sparse_eye - alpha * stochastic
    return lap_alpha


def dfs_reranking(query_features: np.ndarray, gallery_features: np.ndarray, k=50, kq=4, alpha=0.99, gamma=3.0, tol=1e-6, maxiter=20) -> np.ndarray:
    """Computes modified score matrix with DFS reranking method.

    Args:
        query_features (numpy.ndarray): 2-D feature matrix.
        gallery_features (numpy.ndarray): 2-D feature matrix.
        k (int): : A hyperparameter. Default is 50.
        kq (int): A hyperparameter. Default is 10.
        alpha (float): A hyperparameter. Default is 0.99.
        gamma (float): A hyperparameter. Default is 3.0.
        tol (float): A hyperparameter for conjugate gradient method. Default is 20.
        maxiter (int): A hyperparameter for conjugate gradient method. Default is 20.

    Returns:
        numpy.ndarray: modified score matrix.
    
    Examples:
        >>> k, kq = 50, 4
        >>> k, kq = 50, 10
    """
    # print('Applying DFS Re-ranking')
    query_num = query_features.shape[0]
    gallery_num = gallery_features.shape[0]
    similarity = np.dot(query_features, gallery_features.T)
    ranks = np.argsort(-similarity, axis=1)

    knn = KNN(gallery_features, method='cosine')
    sims, ids = knn.search(gallery_features, 1000)
    trunc_ids = ids
    lap_alpha = get_laplacian(sims[:, :k], ids[:, :k], alpha=alpha, gamma=gamma)

    scores = np.zeros((query_num, gallery_num))
    for i in range(query_num):
        y = np.zeros(gallery_num)
        y[ranks[i,:kq]] = similarity[i,ranks[i,:kq]] ** gamma
        score, _ = linalg.cg(lap_alpha, y, tol=tol, maxiter=maxiter)
        scores[i] = score

    return scores


def fsr_reranking(query_features: np.ndarray, gallery_features:np.ndarray, k=50, kq=10, r=1000, alpha=0.99, gamma=3.0) -> np.ndarray:
    """Computes modified score matrix with FSR reranking method, 
        a rank-r implementation without acceleration.

    Args:
        query_features (numpy.ndarray): 2-D feature matrix.
        gallery_features (numpy.ndarray): 2-D feature matrix.
        k (int): : A hyperparameter. Default is 50.
        kq (int): A hyperparameter. Default is 10.
        r (int): A hyperparameter. Default is 1000.
        alpha (float): A hyperparameter. Default is 0.99.
        gamma (float): A hyperparameter. Default is 3.0.

    Returns:
        numpy.ndarray: modified score matrix.
    
    Examples:
        >>> k, kq = 50, 4
        >>> k, kq = 50, 10
    """
    # print('Applying FSR Re-ranking')
    query_num = query_features.shape[0]
    gallery_num = gallery_features.shape[0]
    similarity = np.dot(query_features, gallery_features.T)
    ranks = np.argsort(-similarity, axis=1)

    knn = KNN(gallery_features, method='cosine')
    sims, ids = knn.search(gallery_features, 1000)
    trunc_ids = ids
    affinity = get_affinity(sims[:, :k], ids[:, :k], gamma=gamma)
    num = affinity.shape[0]
    degrees = affinity @ np.ones(num) + 1e-12
    # mat: degree matrix ^ (-1/2)
    mat = sparse.dia_matrix(
        (degrees ** (-0.5), [0]), shape=(num, num), dtype=np.float32)
    stochastic = mat @ affinity @ mat

    eigenvalues, eigenvectors = scipy.linalg.eigh(stochastic.toarray())
    eigen_ranks = np.argsort(-eigenvalues)
    r=gallery_num
    eigenvalues, eigenvectors = eigenvalues[eigen_ranks[:r]], eigenvectors[:,eigen_ranks[:r]]

    transfer = np.diag(1/(1-alpha*eigenvalues))

    scores = np.zeros((query_num, gallery_num))
    for i in range(query_num):
        y = np.zeros(gallery_num)
        y[ranks[i,:kq]] = similarity[i,ranks[i,:kq]] ** gamma
        score = eigenvectors @ transfer @ eigenvectors.T @ y
        scores[i] = score.reshape(-1)

    return scores


def rdp_reranking(query_features: np.ndarray, gallery_features: np.ndarray, sigma=0.5, knn=30, max_iter_diffusion=20, mu=0.3) -> np.ndarray:
    """Computes modified score matrix with RDP reranking method.

    Args:
        query_features (numpy.ndarray): 2-D feature matrix.
        gallery_features (numpy.ndarray): 2-D feature matrix.
        sigma (float): A hyperparameter. Default is 0.5.
        k (int): A hyperparameter. Default is 30.
        max_iter_diffusion (int): A hyperparameter. Default is 20.
        mu (float): A hyperparameter. Default is 0.3.

    Returns:
        numpy.ndarray: modified score matrix.
    
    Examples:
        >>> knn = 30
        >>> knn = 60
    """
    # print('Applying RDP Re-ranking')
    alpha = 1/(1+mu)
    query_num = query_features.shape[0]
    gallery_num = gallery_features.shape[0]
    all_num = query_num + gallery_num
    
    features = np.concatenate((query_features.astype(np.float32), gallery_features.astype(np.float32)), axis=0)
    original_dist = compute_distance_matrix(features, features, 'euclidean').numpy()

    original_dist = original_dist / np.max(original_dist)
    original_dist = np.power(original_dist, 2).astype(np.float32)

    initial_rank = np.argsort(original_dist).astype(np.int32)
    Sim_matrix = np.exp(-original_dist/sigma**2)

    W_matrix = np.zeros_like(original_dist).astype(np.float32)
    for i in range(all_num):
        W_matrix[i, initial_rank[i,:knn]] = Sim_matrix[i, initial_rank[i,:knn]]
    W_matrix = (W_matrix + np.transpose(W_matrix)) / 2

    X_axis, Y_axis = np.nonzero(W_matrix)
    non_zero_num = np.count_nonzero(W_matrix)
    weight_value = W_matrix[X_axis, Y_axis]

    d = 1 / np.sqrt(np.sum(W_matrix, axis=1, keepdims=True))
    D_matrix = np.dot(d, np.transpose(d))
    S_matrix = np.multiply(W_matrix, D_matrix)
    A_initial = np.eye(all_num, dtype=np.float32)
    A_matrix = np.eye(all_num, dtype=np.float32)

    S_matrix = torch.from_numpy(S_matrix)
    A_initial = torch.from_numpy(A_initial)
    A_matrix = torch.from_numpy(A_matrix)
    
    for iter in range(max_iter_diffusion):
        A_matrix = alpha*torch.mm(torch.mm(S_matrix, A_matrix), S_matrix.t()) + (1-alpha)*A_initial
    scores = A_matrix[:query_num,query_num:].numpy()

    return scores