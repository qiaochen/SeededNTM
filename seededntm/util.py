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
    """Compute per-gene leverage-based weights from a reference signature matrix.

    Uses FlashDeconv's parameterless design:
    1. Compute raw leverage via SVD (row norms of right singular vectors)
    2. Normalize to probability distribution (sum=1)
    3. Convert to amplitude: sqrt(leverage_prob * G)
    4. Clip to [0.1, 10.0] to prevent extreme values

    This eliminates the need for a manual "power" hyperparameter.
    The sqrt provides inherent softening, and the clip caps the dynamic range
    at 100x (matching FlashDeconv's guard rails).

    Args:
        X_ref: (K, G) reference signature matrix (cell types x genes), or
               (N_cells, G) single-cell matrix that will be transposed.
        n_components: number of SVD components to use. None = min(K, G, 30).
        regularization: small constant for numerical stability.

    Returns:
        (G,) array of gene weights ready to use in compute_tfidf_rep().
        Mean is approximately 1.0, range clipped to [0.1, 10.0].
    """
    X = np.asarray(X_ref, dtype=np.float64)
    if X.shape[0] > X.shape[1]:
        X = X.T  # ensure (smaller_dim, G) for efficient SVD

    G = X.shape[1]

    # Center columns (genes)
    X_centered = X - X.mean(axis=0, keepdims=True)

    if n_components is None:
        n_components = min(X_centered.shape[0], X_centered.shape[1], 30)

    # Thin SVD: X_centered = U @ diag(s) @ V^T
    from numpy.linalg import svd
    try:
        U, s, Vt = svd(X_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback to variance-based scores
        var_scores = np.var(X, axis=0)
        var_scores = var_scores / (var_scores.sum() + regularization)
        weights = np.sqrt(var_scores * G + regularization)
        return np.clip(weights, 0.1, 10.0).astype(np.float32)

    V = Vt[:n_components, :].T  # (G, n_components)
    s = s[:n_components]

    # Leverage scores: weighted sum of squared right singular vectors
    # Weight by s^2 / (s^2 + eps) for soft rank selection (FlashDeconv style)
    s_sq = s ** 2
    weights = s_sq / (s_sq + regularization)
    leverage = np.sum((V ** 2) * weights, axis=1)  # (G,)

    # Normalize to probability distribution (sum = 1)
    leverage = leverage / (leverage.sum() + regularization)

    # Convert to amplitude scale (FlashDeconv parameterless design)
    gene_weights = np.sqrt(leverage * G + regularization)

    # Normalize mean to 1.0 so overall TF-IDF scale is preserved
    gene_weights = gene_weights / (gene_weights.mean() + regularization)

    # Clip dynamic range as final guard rail
    gene_weights = np.clip(gene_weights, 0.1, 10.0)

    return gene_weights.astype(np.float32)


def compute_countsketch_rep(X, sketch_dim=512, leverage_scores=None, seed=0):
    """Dimensionality reduction via leverage-weighted CountSketch.

    An alternative to PCA that:
    - Runs in O(nnz(X)) time (one sparse matmul)
    - Integrates leverage scores directly into sketch amplitudes
    - Preserves norms in expectation (Johnson-Lindenstrauss property)

    The output dimensions are random linear combinations of genes, weighted
    by leverage scores. Unlike PCA, dimensions are not ordered by variance.

    Args:
        X: (N, G) count matrix (sparse or dense). Should be TF-IDF normalized.
        sketch_dim: target dimensionality (default 512, similar to FlashDeconv).
        leverage_scores: (G,) gene weights from compute_leverage_scores().
            If None, uses uniform weights (standard CountSketch).
        seed: random seed for reproducibility.

    Returns:
        (N, sketch_dim) dense array of sketched representations.
    """
    from scipy import sparse as sp

    rng = np.random.RandomState(seed)
    n_samples, n_genes = X.shape

    # Default to uniform weights
    if leverage_scores is None:
        lev_prob = np.ones(n_genes) / n_genes
    else:
        lev = np.asarray(leverage_scores, dtype=np.float64).ravel()
        lev_prob = lev / (lev.sum() + 1e-10)

    # CountSketch: each gene hashes to one bucket with random sign
    bucket_assignments = rng.randint(0, sketch_dim, size=n_genes)
    signs = rng.choice([-1.0, 1.0], size=n_genes)

    # Leverage-weighted amplitudes (FlashDeconv design)
    scale_factors = np.sqrt(lev_prob * n_genes + 1e-10)
    scale_factors = np.clip(scale_factors, 0.1, 10.0)

    # Build sparse sketch matrix Omega: (n_genes, sketch_dim)
    data = signs * scale_factors
    row_idx = np.arange(n_genes)
    col_idx = bucket_assignments

    Omega = sp.csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(n_genes, sketch_dim),
        dtype=np.float64,
    )

    # Normalize columns for numerical stability
    col_norms = np.sqrt(np.asarray(Omega.power(2).sum(axis=0)).flatten())
    col_norms = np.maximum(col_norms, 1e-10)
    scale = np.sqrt(n_genes / sketch_dim)
    Omega = Omega.multiply(scale / col_norms)

    # Project: X_sketch = X @ Omega
    if issparse(X):
        X_sketch = X @ Omega
        X_sketch = X_sketch.toarray()
    else:
        X_sketch = X @ Omega.toarray()

    return X_sketch.astype(np.float32)
