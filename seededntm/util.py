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
    K = len(marker_genes)
    N = adata.n_obs
    scores = np.zeros((N, K), dtype=np.float64)

    for celltype, info in marker_genes.items():
        topic_idx = info['topic_index']
        feats = info.get('features', info.get('genes', []))
        expr = adata[:, feats].X.sum(axis=1) / len(feats)
        if issparse(expr):
            expr = np.asarray(expr).ravel()
        scores[:, topic_idx] = np.asarray(expr).ravel()

    return softmax(scores / temperature, axis=1).round(4)


def compute_topic_prior_idf(adata, marker_genes, temperature=0.8):
    """IDF-weighted topic prior: weight seed genes by inverse document frequency.

    Instead of uniform averaging across seed genes, this weights each gene by
    its IDF (rarity), so ubiquitously expressed genes contribute less noise.
    """
    X = adata.X
    if issparse(X):
        df = np.asarray((X > 0).sum(axis=0)).ravel()
    else:
        df = (X > 0).sum(axis=0).ravel()
    N = X.shape[0]
    idf = np.log(N / (df + 1)).astype(np.float64)

    var_names = list(adata.var_names)
    var_set = set(var_names)
    var_to_idx = {g: i for i, g in enumerate(var_names)}

    K = len(marker_genes)
    scores = np.zeros((N, K), dtype=np.float64)

    for ct, info in marker_genes.items():
        topic_idx = info['topic_index']
        feats = [g for g in info.get('features', info.get('genes', [])) if g in var_set]
        if len(feats) == 0:
            continue
        feat_idx = [var_to_idx[g] for g in feats]
        weights = idf[feat_idx]

        if issparse(X):
            expr = X[:, feat_idx].toarray()
        else:
            expr = np.asarray(X[:, feat_idx])

        scores[:, topic_idx] = (expr * weights).sum(axis=1) / weights.sum()

    return softmax(scores / temperature, axis=1).round(4)


def spatial_smooth_prior(prior_matrix, coords, k_neighbors=6):
    """Smooth topic prior using spatial k-NN neighborhood averaging.

    Each spot's prior becomes the mean of itself and its k spatial neighbors.
    Rows are re-normalized to sum to 1 after smoothing.

    Args:
        prior_matrix: (N, K) topic prior matrix (rows sum to 1).
        coords: (N, 2) spatial coordinates.
        k_neighbors: number of spatial neighbors (default 6 for hex grids).

    Returns:
        (N, K) spatially-smoothed prior, rows sum to 1.
    """
    A = kneighbors_graph(coords, n_neighbors=k_neighbors, mode='connectivity', include_self=True)
    A = ((A + A.T) > 0).astype(np.float32)
    A_csr = csr_matrix(A)

    degree = np.asarray(A_csr.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0

    smoothed = A_csr @ prior_matrix
    smoothed = smoothed / degree[:, np.newaxis]

    row_sums = smoothed.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    smoothed = smoothed / row_sums

    return smoothed.round(4)


def compute_topic_prior_scoregenes(adata, marker_genes, temperature=0.8, ctrl_size=50):
    """Background-corrected topic prior using Seurat-style gene scoring.

    Uses sc.tl.score_genes (Tirosh et al. 2016) to compute per-cell, per-topic
    scores with expression-matched control gene subtraction. This removes cell
    size and housekeeping confounders that bias the uniform-mean approach.

    Args:
        adata: AnnData with .X expression matrix.
        marker_genes: dict {"CellType": {"features": [...], "topic_index": int}}.
        temperature: softmax temperature for converting scores to probabilities.
        ctrl_size: number of control genes per expression bin (default 50).

    Returns:
        (N, K) topic prior matrix, rows sum to 1.
    """
    import scanpy as sc

    cls2id = {ct: feat['topic_index'] for ct, feat in marker_genes.items()}
    var_set = set(adata.var_names)
    scores = []

    for ct in cls2id.keys():
        gene_list = [g for g in marker_genes[ct]['features'] if g in var_set]
        if len(gene_list) == 0:
            scores.append(np.zeros((adata.n_obs, 1)))
            continue
        score_key = f'_prior_sg_{ct}'
        sc.tl.score_genes(adata, gene_list, score_name=score_key,
                          ctrl_size=ctrl_size, random_state=0)
        scores.append(adata.obs[score_key].values.reshape(-1, 1))
        del adata.obs[score_key]

    raw_scores = np.hstack(scores)
    topic_prior = softmax(raw_scores / temperature, axis=1).round(4)
    return topic_prior


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


def _compute_profile_similarity(adata, seeds_dict):
    """Compute expression-weighted profile similarity between seed-defined types.

    For each seed topic, identifies the top 10% of cells by mean seed gene
    expression, computes their mean log1p-CPM transcriptome profile, then
    returns the mean pairwise cosine similarity across all topic profiles.

    High similarity means cell types share expression programs (model needs
    flexibility). Low similarity means types are distinct (trust the prior).
    """
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

    X = adata.X
    n_spots, n_genes = X.shape
    if issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X, dtype=np.float64)

    row_sums = X_dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_norm = np.log1p(X_dense / row_sums * 1e4)

    var_names = list(adata.var_names)
    var_to_idx = {g: i for i, g in enumerate(var_names)}
    K = len(seeds_dict)

    topic_profiles = []
    for _ct_name, record in seeds_dict.items():
        gene_idx = [var_to_idx[g] for g in record["features"] if g in var_to_idx]
        if gene_idx:
            cell_scores = X_norm[:, gene_idx].mean(axis=1)
            threshold = np.percentile(cell_scores, 90)
            mask = cell_scores >= threshold
            if mask.sum() > 5:
                profile = X_norm[mask].mean(axis=0)
            else:
                profile = X_norm.mean(axis=0)
        else:
            profile = X_norm.mean(axis=0)
        topic_profiles.append(profile)

    profiles = np.stack(topic_profiles)
    sim = _cos_sim(profiles)
    np.fill_diagonal(sim, 0)
    mean_sim = float(sim[np.triu_indices(K, k=1)].mean())
    return mean_sim


def rebalance_prior(prior):
    """Rebalance topic prior so the population marginal is uniform across topics.

    Rescales each column of the prior matrix so that the mean across all cells
    is 1/K for every topic. Rows are re-normalized to sum to 1.

    This addresses class imbalance: when the prior is dominated by one type
    (e.g., 89% Tumor in CRC_II), the CE loss gives negligible gradient for
    rare types. Rebalancing ensures equal gradient contribution from all topics.

    Args:
        prior: (N, K) array of per-cell topic prior probabilities.

    Returns:
        (N, K) rebalanced prior with uniform marginal.
    """
    K = prior.shape[1]
    mean_per_topic = prior.mean(axis=0)
    mean_per_topic = np.maximum(mean_per_topic, 1e-8)
    scale = (1.0 / K) / mean_per_topic
    balanced = prior * scale[np.newaxis, :]
    row_sums = balanced.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return balanced / row_sums


def auto_configure(adata, seeds_dict, has_spatial=None):
    """Automatically select SeedTopic hyperparameters from dataset properties.

    Returns a single recommended configuration. The key decisions:
    - Targeted panels (seed_cov > 0.1): use IDF representation, standard prior, reg=0.99
    - Genome-wide panels: use leverage-weighted representation, spatial prior, reg=0.99

    Args:
        adata: AnnData object with expression matrix in .X
        seeds_dict: dict mapping cell type names to {features, topic_index}
        has_spatial: override for spatial coordinate detection (default: auto-detect)

    Returns:
        dict with keys:
            reg_topic_prior, prior_mode, representation: recommended config
            K, use_nb_obs, early_stop: training settings
            diagnostics: computed data properties
    """
    import logging
    logger = logging.getLogger(__name__)

    n_spots, n_genes = adata.shape
    n_seed_types = len(seeds_dict)

    if has_spatial is None:
        has_spatial = "spatial" in adata.obsm

    # --- Platform detection via seed gene coverage ---
    var_set = set(adata.var_names)
    all_seed_genes = set()
    for r in seeds_dict.values():
        all_seed_genes.update(g for g in r["features"] if g in var_set)
    seed_cov = len(all_seed_genes) / n_genes
    targeted_panel = seed_cov > 0.1

    # --- Seed gene overlap (drives representation choice) ---
    gene_type_count = {}
    for r in seeds_dict.values():
        for g in r["features"]:
            if g in var_set:
                gene_type_count[g] = gene_type_count.get(g, 0) + 1
    total_seed_mentions = sum(len(r["features"]) for r in seeds_dict.values())
    n_unique_seeds = len(gene_type_count)
    overlap_ratio = 1.0 - (n_unique_seeds / max(total_seed_mentions, 1))

    # --- Build config ---
    K = n_seed_types
    use_nb_obs = True
    early_stop = n_spots > 50000

    if targeted_panel:
        config = {
            "reg_topic_prior": 0.99,
            "prior_mode": "standard",
            "representation": "idf_pca",
        }
    else:
        config = {
            "reg_topic_prior": 0.99,
            "prior_mode": "spatial" if has_spatial else "standard",
            "representation": "idf_lev_seed_specificity_pca",
        }

    # --- Diagnostic info ---
    top10_sim = _compute_profile_similarity(adata, seeds_dict)
    if top10_sim > 0.9 and n_seed_types < 8:
        logger.info(
            f"auto_configure: high profile similarity ({top10_sim:.3f}) with few "
            f"seed types ({n_seed_types}) suggests potential benefit from "
            f"multi-scale K (try --K {n_seed_types * 2})"
        )

    logger.info(
        f"auto_configure: seed_cov={seed_cov:.4f} "
        f"({'targeted' if targeted_panel else 'genome-wide'}), "
        f"overlap_ratio={overlap_ratio:.3f}, K={K}, "
        f"reg={config['reg_topic_prior']}, prior_mode={config['prior_mode']}"
    )

    return {
        "reg_topic_prior": config["reg_topic_prior"],
        "prior_mode": config["prior_mode"],
        "representation": config["representation"],
        "K": K,
        "use_nb_obs": use_nb_obs,
        "early_stop": early_stop,
        "diagnostics": {
            "top10_sim": round(top10_sim, 4),
            "seed_cov": round(seed_cov, 4),
            "targeted_panel": targeted_panel,
            "overlap_ratio": round(overlap_ratio, 4),
            "n_seed_types": n_seed_types,
            "n_genes": n_genes,
            "n_spots": n_spots,
        },
    }
