import os
import time
import torch
import numpy as np
from revisitop import configdataset, load_pickle, compute_map_and_print
from reranker import *

def main():
    dataset = 'roxford5k'
    # dataset = 'rparis6k'
    data_root = 'revisitop/dataset'
    # feature_name = 'dolg'
    feature_name = 'gl18-tl-resnet101-gem-w'

    cfg = configdataset(dataset, data_root)
    feature = load_pickle(os.path.join(data_root, dataset, 'features/{}.pkl'.format(feature_name)))

    qvecs = feature['query']
    vecs = feature['db']

    scores = np.dot(qvecs, vecs.T)
    ranks = np.argsort(-scores, axis=1).T
    compute_map_and_print(dataset, ranks, cfg['gnd'])

    
    scores = aqe_reranking(qvecs, vecs)
    ranks = np.argsort(-scores, axis=1).T
    compute_map_and_print(dataset, ranks, cfg['gnd'])


    dist = cas_reranking(qvecs, vecs, metric='euclidean', k1=6, k2=60, k3=70, k4=7, k5=80)
    ranks = np.argsort(dist, axis=1).T
    compute_map_and_print(dataset, ranks, cfg['gnd'])

if __name__ == '__main__':
    main()