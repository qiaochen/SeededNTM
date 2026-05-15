import numpy as np
import squidpy as sq
import os
import pandas as pd

from scipy.sparse import csr_matrix, diags, identity, issparse
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


def aggregate_mean(data):
    base = np.sum(data, axis=1)
    if len(base.shape) == 1:
        base = base.reshape(-1, 1)
    data = data / base
    means = np.mean(data, axis=0)
    return np.log(means + 1e-15)

class EarlyStopper:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_training_loss = np.inf

    def early_stop(self, training_loss):
        if training_loss < self.min_training_loss:
            self.min_training_loss = training_loss
            self.counter = 0
        elif training_loss > (self.min_training_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def reset(self):
        self.counter = 0
        self.min_training_loss = np.inf
        
def compute_tfidf_rep(X, n_pcs=100, idf=None, return_idf=False):
    if not idf is None:
        if len(np.shape(idf)) == 1:
            idf = idf.reshape(1, -1)
        
    if issparse(X):
        if idf is None:
            idf = np.log(X.shape[0] / ( np.sum(X > 0, axis=0)))
        tfidf = ((X / X.sum(axis=1)).multiply(idf)).astype(np.float32)
    else:
        if idf is None:
            idf = np.log(X.shape[0] / np.sum(X > 0, axis=0, keepdims=True))
        tfidf = ((X / np.sum(X, axis=1, keepdims=True))* idf).astype(np.float32)
    
    out = tfidf
    if not n_pcs is None and (type(n_pcs) == int) and n_pcs > 0 and n_pcs < X.shape[1]:
        tfidf_pca = PCA(n_pcs).fit_transform(tfidf).astype(np.float32)
        out = tfidf_pca
    if return_idf:
        return out, idf
    return out

def compute_topic_prior(adata, marker_genes, temperature=0.8):
    cls2id = {celltype:features['topic_index'] for celltype, features in marker_genes.items()}
    qusi_topics = []

    for celltype in cls2id.keys():
        qusi_topics.append(adata[:, marker_genes[celltype]['features']].X.sum(axis=1) / len(marker_genes[celltype]['features']))
        
    qusi_topics = softmax(np.hstack(qusi_topics)/temperature, axis=1).round(4)
    return qusi_topics


def build_spatial_graph(coords, k_neighbors=6, mode='connectivity'):
    """Build a symmetric k-NN spatial adjacency matrix.

    Args:
        coords: (N, 2) array of spatial coordinates.
        k_neighbors: number of nearest neighbors (default 6 for hex grids).
        mode: 'connectivity' (binary) or 'distance' (weighted).

    Returns:
        Sparse CSR adjacency matrix (N, N), symmetrized.
    """
    A = kneighbors_graph(coords, n_neighbors=k_neighbors, mode=mode, include_self=False)
    A = ((A + A.T) > 0).astype(np.float32)
    return csr_matrix(A)
