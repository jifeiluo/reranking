import torch
import time
import numpy as np
from torch.nn import functional as F
from joblib import Parallel, delayed
import pathos.multiprocessing as mp


def compute_distance_matrix(input1, input2, metric='euclidean', device=torch.device('cuda')):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine" or "jensen shannon" or "jaccard".
            Default is "euclidean".
        device (torch.device, optional): 'cpu' or 'cuda'
            Default is torch.device('cuda').

    Returns:
        torch.Tensor: distance matrix.  
    """
    # check input
    if isinstance(input1, np.ndarray):
        input1 = torch.from_numpy(input1)
    if isinstance(input2, np.ndarray):
        input2 = torch.from_numpy(input2)
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.dim()
    )
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    elif metric == 'jensen shannon':
        distmat = jensen_shannon_divergence(input1, input2, device)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = mat1 + mat2
    distmat.addmm_(mat1=input1, mat2=input2.t(), beta=1, alpha=-2)
    distmat.clamp_(min=0)
    return distmat


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    distmat.clamp_(min=0, max=2)
    return distmat


def jensen_shannon_divergence(input1, input2, device=torch.device('cuda')):
    '''Computes the Jensen Shannon divergence.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        device (torch.device, optional): 'cpu' or 'cuda'
            Default is torch.device('cuda').
        
    Returns:
        torch.Tensor: divergence matrix.
    '''
    input1, input2 = input1.to(device), input2.to(device)

    if device==torch.device('cuda'): 
        from reranker.sparse_divergence import sparse_divergence
        js_divergence = sparse_divergence.jensen_shannon(input1, input2)
        torch.cuda.synchronize()
        return js_divergence
    
    elif device==torch.device('cpu'):
        # First round computing
        def first_round_closure(i):
            nozero_index1 = np.nonzero(input1[i])[0]
            expand_input1 = np.tile(input1[i, nozero_index1], (input2.shape[0], 1))
            expand_input1_plus_input2 = expand_input1 + input2[:, nozero_index1]
            result = np.dot(input1[i, nozero_index1].reshape(1,-1), np.log2(2*expand_input1/expand_input1_plus_input2).reshape(input2.shape[0],-1).T)
            return result

        # Second round computing
        def second_round_closure(i):
            nozero_index2 = np.nonzero(input2[i])[0]
            expand_input2 = np.tile(input2[i, nozero_index2], (input1.shape[0], 1))
            expand_input2_plus_input1 = expand_input2 + input1[:, nozero_index2]
            result = np.dot(input2[i, nozero_index2].reshape(1,-1), np.log2(2*expand_input2/expand_input2_plus_input1).reshape(input1.shape[0],-1).T)
            return result
        
        def wrapper_first_round(args):
            return first_round_closure(args)

        def wrapper_second_round(args):
            return second_round_closure(args)

        input1, input2 = input1.numpy(), input2.numpy()
        js_divergence1 = np.concatenate(Parallel(n_jobs=16)(delayed(wrapper_first_round)(i) for i in range(input1.shape[0])))
        js_divergence2 = np.concatenate(Parallel(n_jobs=16)(delayed(wrapper_second_round)(i) for i in range(input2.shape[0])))
        js_divergence = 0.5*(js_divergence1 + js_divergence2.T)
        js_divergence = torch.from_numpy(js_divergence)

        return js_divergence