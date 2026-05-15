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


def compute_reffree_leverage(
    X_counts,
    seeds_dict,
    var_names,
    method="pseudo_sig",
    n_topics=None,
    temperature=0.8,
    regularization=1e-6,
):
    """Compute reference-free gene weights using only seed genes and ST data.

    Three strategies available:
    - "pseudo_sig": Build pseudo-signature matrix from seed-guided topic_prior,
      then compute leverage via SVD (recommended).
    - "self_leverage_k": SVD of TF-IDF with top-K components (K=num_topics),
      extracting cell-type-scale structure.
    - "seed_specificity": Upweight genes that are specific markers (appear in
      fewer cell types).

    All methods return weights in the same format as compute_leverage_scores():
    clipped to [0.1, 10.0], mean ~1.0.

    Args:
        X_counts: (N, G) raw count matrix (sparse or dense).
        seeds_dict: dict with structure {"CellType": {"features": [...], "topic_index": int}}.
        var_names: (G,) gene names matching columns of X_counts.
        method: one of "pseudo_sig", "self_leverage_k", "seed_specificity".
        n_topics: number of topics. Inferred from seeds_dict if None.
        temperature: softmax temperature for topic_prior computation.
        regularization: numerical stability constant.

    Returns:
        (G,) array of gene weights, clipped to [0.1, 10.0], mean ~1.0.
    """
    G = X_counts.shape[1]
    var_names = np.asarray(var_names)

    if n_topics is None:
        n_topics = len(seeds_dict)

    if method == "pseudo_sig":
        return _reffree_pseudo_sig(
            X_counts, seeds_dict, var_names, n_topics, temperature, regularization
        )
    elif method == "self_leverage_k":
        return _reffree_self_leverage_k(
            X_counts, n_topics, regularization
        )
    elif method == "seed_specificity":
        return _reffree_seed_specificity(
            seeds_dict, var_names, G, regularization
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use pseudo_sig, self_leverage_k, or seed_specificity.")


def _reffree_pseudo_sig(X_counts, seeds_dict, var_names, n_topics, temperature, reg):
    """Approach 1: Seed-guided pseudo-signatures from ST data."""
    from scipy.special import softmax as sp_softmax

    N, G = X_counts.shape
    X = X_counts
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    # Recompute topic_prior from seeds (don't rely on stored value)
    var_list = list(var_names)
    var_set = set(var_list)
    var_to_idx = {g: i for i, g in enumerate(var_list)}

    scores = np.zeros((N, n_topics), dtype=np.float64)
    seed_gene_indices = set()

    for ct_name, record in seeds_dict.items():
        topic_idx = record["topic_index"]
        feats = [g for g in record["features"] if g in var_set]
        if len(feats) == 0:
            continue
        feat_idx = [var_to_idx[g] for g in feats]
        seed_gene_indices.update(feat_idx)
        scores[:, topic_idx] = X[:, feat_idx].sum(axis=1) / len(feats)

    topic_prior = sp_softmax(scores / temperature, axis=1)

    # Normalize counts (CPM + log1p) before computing pseudo-signatures
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_norm = np.log1p(X / row_sums * 1e4)

    # Weighted mean per topic: pseudo_sig (K, G)
    denom = topic_prior.sum(axis=0, keepdims=True).T  # (K, 1)
    denom[denom == 0] = 1
    pseudo_sig = (topic_prior.T @ X_norm) / denom  # (K, G)

    # Circularity mitigation: zero out seed gene columns
    seed_idx_array = np.array(sorted(seed_gene_indices))
    if len(seed_idx_array) > 0:
        pseudo_sig[:, seed_idx_array] = 0.0

    # Compute leverage via SVD (same as compute_leverage_scores but inline to
    # avoid the transpose heuristic bug when K < G is already guaranteed)
    X_centered = pseudo_sig - pseudo_sig.mean(axis=0, keepdims=True)
    n_components = min(X_centered.shape[0], X_centered.shape[1], 30)

    from numpy.linalg import svd
    try:
        U, s, Vt = svd(X_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.ones(G, dtype=np.float32)

    V = Vt[:n_components, :].T  # (G, n_components)
    s = s[:n_components]

    s_sq = s ** 2
    weights = s_sq / (s_sq + reg)
    leverage = np.sum((V ** 2) * weights, axis=1)  # (G,)

    # FlashDeconv-style normalization
    leverage = leverage / (leverage.sum() + reg)
    gene_weights = np.sqrt(leverage * G + reg)
    gene_weights = gene_weights / (gene_weights.mean() + reg)
    gene_weights = np.clip(gene_weights, 0.1, 10.0)

    return gene_weights.astype(np.float32)


def _reffree_self_leverage_k(X_counts, n_topics, reg):
    """Approach 2a: Self-leverage from top-K SVD of TF-IDF."""
    N, G = X_counts.shape

    # Compute TF-IDF (no PCA, no gene weights)
    if issparse(X_counts):
        idf = np.log(N / (np.asarray((X_counts > 0).sum(axis=0)).ravel() + 1))
        row_sums = np.asarray(X_counts.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        from scipy.sparse import diags as sp_diags
        tf = X_counts.multiply(1.0 / row_sums[:, np.newaxis])
        tfidf = tf.multiply(idf).toarray()
    else:
        X = np.asarray(X_counts, dtype=np.float64)
        idf = np.log(N / (np.sum(X > 0, axis=0) + 1))
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        tfidf = (X / row_sums) * idf

    # Top-K SVD (K = num_topics): captures cell-type-scale structure
    from numpy.linalg import svd
    # For efficiency on large matrices, use randomized SVD
    n_components = min(n_topics, G, N)
    try:
        from sklearn.utils.extmath import randomized_svd
        U, s, Vt = randomized_svd(tfidf, n_components=n_components, random_state=42)
    except Exception:
        U, s, Vt = svd(tfidf, full_matrices=False)
        U, s, Vt = U[:, :n_components], s[:n_components], Vt[:n_components, :]

    V = Vt.T  # (G, K)

    # Leverage from K-dimensional subspace
    s_sq = s ** 2
    weights = s_sq / (s_sq + reg)
    leverage = np.sum((V ** 2) * weights, axis=1)  # (G,)

    # FlashDeconv-style normalization
    leverage = leverage / (leverage.sum() + reg)
    gene_weights = np.sqrt(leverage * G + reg)
    gene_weights = gene_weights / (gene_weights.mean() + reg)
    gene_weights = np.clip(gene_weights, 0.1, 10.0)

    return gene_weights.astype(np.float32)


def _reffree_seed_specificity(seeds_dict, var_names, G, reg):
    """Approach 3: Gene weighting by seed marker specificity."""
    var_list = list(var_names)
    var_set = set(var_list)
    var_to_idx = {g: i for i, g in enumerate(var_list)}

    # Count how many cell types each gene appears in as a marker
    gene_type_count = np.zeros(G, dtype=np.float64)
    gene_is_seed = np.zeros(G, dtype=bool)

    for ct_name, record in seeds_dict.items():
        feats = [g for g in record["features"] if g in var_set]
        for g in feats:
            idx = var_to_idx[g]
            gene_type_count[idx] += 1
            gene_is_seed[idx] = True

    # Weight: more specific markers (fewer types) get higher weight
    # w = 1 + alpha / n_types_sharing for seeds, 1.0 for non-seeds
    alpha = 2.0  # tunable scaling factor
    gene_weights = np.ones(G, dtype=np.float64)
    mask = gene_is_seed
    gene_weights[mask] = 1.0 + alpha / gene_type_count[mask]

    # Normalize mean to 1.0 and clip
    gene_weights = gene_weights / (gene_weights.mean() + reg)
    gene_weights = np.clip(gene_weights, 0.1, 10.0)

    return gene_weights.astype(np.float32)
