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
        
def compute_tfidf_rep(X, n_pcs=100, idf=None, return_idf=False, gene_weights=None):
    """Compute TF-IDF representation with optional gene weighting, then PCA.

    Args:
        X: (N, G) count matrix (sparse or dense).
        n_pcs: number of PCA components. None or -1 to skip PCA.
        idf: pre-computed IDF weights. None to compute from X.
        return_idf: if True, also return the IDF weights.
        gene_weights: (G,) array of per-gene weights (e.g. leverage scores).
            Applied as multiplicative scaling to TF-IDF columns before PCA.
            This amplifies informative genes in the low-rank projection.

    Returns:
        (N, n_pcs) PCA of weighted TF-IDF, or (N, G) if n_pcs is None.
    """
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
    
    if gene_weights is not None:
        gene_weights = np.asarray(gene_weights, dtype=np.float32).ravel()
        if issparse(tfidf):
            tfidf = tfidf.multiply(gene_weights.reshape(1, -1))
        else:
            tfidf = tfidf * gene_weights.reshape(1, -1)
    
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


def compute_leverage_scores(X_ref, n_components=None, regularization=1e-6):
    """Compute per-gene leverage scores from a reference signature matrix.

    Leverage scores measure each gene's contribution to distinguishing cell types
    in the reference, independent of expression variance. High-leverage genes are
    informative for rare cell types that variance-based methods (HVG, PCA) miss.

    Based on FlashDeconv (Yang et al. 2025): l_g = sum_j (V_gj^2 * s_j^2 / sum(s^2))

    Args:
        X_ref: (K, G) reference signature matrix (cell types x genes), or
               (N_cells, G) single-cell matrix that will be aggregated.
        n_components: number of SVD components to use. None = min(K, G, 30).
        regularization: small constant for numerical stability.

    Returns:
        (G,) array of leverage scores, normalized to sum to G (so mean=1).
    """
    X = np.asarray(X_ref, dtype=np.float64)
    if X.shape[0] > X.shape[1]:
        X = X.T  # ensure (smaller_dim, G) for efficient SVD

    # Center columns (genes)
    X_centered = X - X.mean(axis=0, keepdims=True)

    if n_components is None:
        n_components = min(X_centered.shape[0], X_centered.shape[1], 30)

    # Thin SVD: X_centered = U @ diag(s) @ V^T
    from numpy.linalg import svd
    U, s, Vt = svd(X_centered, full_matrices=False)
    V = Vt[:n_components, :].T  # (G, n_components)
    s = s[:n_components]

    # Leverage scores: weighted sum of squared right singular vectors
    s_sq = s ** 2
    weights = s_sq / (s_sq.sum() + regularization)
    leverage = (V ** 2) @ weights  # (G,)

    # Normalize so mean leverage = 1 (unit-free scaling)
    leverage = leverage / (leverage.mean() + regularization)

    return leverage.astype(np.float32)
