"""Unit tests verifying critical operations BEFORE running SeedTopic benchmark experiments.

Covers:
  1. Seed genes exist in spatial_adata.var_names (ref-based and ref-free)
  2. K matches between seeds file and expected dataset config
  3. Raw counts in spatial_adata.X (non-negative integers)
  4. compute_representation produces correct shape with no NaN
  5. compute_topic_prior produces valid probability distributions
  6. SeededNTM model instantiation with use_nb_obs=True
  7. Single training step (n_epochs=1) runs without error
  8. Spatial coordinates exist in adata.obsm["spatial"]
  9. Ground truth indices match spatial_adata.obs_names

Run from project root:
    pytest experiments/benchmark_fair/test_benchmark_correctness.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy.sparse import issparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "experiments" / "benchmark_fair" / "data" / "processed"

DATASETS = ["visium_NPC", "xenium_BC", "visiumHD_CRC_I", "visiumHD_CRC_II"]

EXPECTED_K = {
    "visium_NPC": 7,
    "xenium_BC": 19,
    "visiumHD_CRC_I": 5,
    "visiumHD_CRC_II": 6,
}

SEED_FILES = ["seeds.json", "seeds_leiden_chatgpt.json"]

REPRESENTATIONS = ["idf_pca", "idf_lev_seed_spec", "lev_pseudo_sig"]

PRIOR_MODES = ["standard", "spatial", "idf_spatial"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=DATASETS)
def dataset_name(request):
    return request.param


@pytest.fixture
def dataset_dir(dataset_name):
    d = DATA_DIR / dataset_name
    if not d.exists():
        pytest.skip(f"Processed data not found: {d}")
    return d


@pytest.fixture
def spatial_adata(dataset_dir):
    h5ad_path = dataset_dir / "spatial_adata.h5ad"
    if not h5ad_path.exists():
        pytest.skip(f"spatial_adata.h5ad not found: {h5ad_path}")
    return sc.read_h5ad(h5ad_path)


@pytest.fixture
def seeds_refbased(dataset_dir):
    path = dataset_dir / "seeds.json"
    if not path.exists():
        pytest.skip(f"seeds.json not found: {path}")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def seeds_reffree(dataset_dir):
    path = dataset_dir / "seeds_leiden_chatgpt.json"
    if not path.exists():
        pytest.skip(f"seeds_leiden_chatgpt.json not found: {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Test 1: Seed genes exist in spatial_adata.var_names
# ---------------------------------------------------------------------------


class TestSeedGenesExist:
    @pytest.mark.parametrize("seed_file", SEED_FILES)
    def test_all_seed_genes_in_var_names(self, dataset_dir, spatial_adata, seed_file):
        seed_path = dataset_dir / seed_file
        if not seed_path.exists():
            pytest.skip(f"{seed_file} not found: {seed_path}")

        with open(seed_path) as f:
            seeds = json.load(f)

        var_set = set(spatial_adata.var_names)
        missing = {}
        for cell_type, record in seeds.items():
            genes = record["features"]
            missing_genes = [g for g in genes if g not in var_set]
            if missing_genes:
                missing[cell_type] = missing_genes

        assert not missing, (
            f"Missing seed genes in {dataset_dir.name}/{seed_file}:\n"
            + "\n".join(f"  {ct}: {genes}" for ct, genes in missing.items())
        )


# ---------------------------------------------------------------------------
# Test 2: K matches between seeds file and dataset config
# ---------------------------------------------------------------------------


class TestKMatches:
    @pytest.mark.parametrize("seed_file", SEED_FILES)
    def test_k_equals_expected(self, dataset_name, dataset_dir, seed_file):
        seed_path = dataset_dir / seed_file
        if not seed_path.exists():
            pytest.skip(f"{seed_file} not found: {seed_path}")

        with open(seed_path) as f:
            seeds = json.load(f)

        expected_k = EXPECTED_K[dataset_name]
        actual_k = len(seeds)
        assert actual_k == expected_k, (
            f"{dataset_name}/{seed_file}: expected K={expected_k}, got {actual_k}. "
            f"Keys: {list(seeds.keys())}"
        )


# ---------------------------------------------------------------------------
# Test 3: Raw counts in spatial_adata.X
# ---------------------------------------------------------------------------


class TestRawCounts:
    def test_non_negative(self, spatial_adata, dataset_name):
        X = spatial_adata.X
        if issparse(X):
            data = X.data
        else:
            data = X.ravel()
        assert np.all(data >= 0), (
            f"{dataset_name}: spatial_adata.X contains negative values"
        )

    def test_integer_values(self, spatial_adata, dataset_name):
        X = spatial_adata.X
        if issparse(X):
            sample = X.data[:10000]
        else:
            sample = X.ravel()[:10000]
        sample = np.asarray(sample, dtype=np.float64)
        non_integer = np.abs(sample - np.round(sample)) > 1e-6
        frac_non_int = non_integer.sum() / len(sample)
        assert frac_non_int < 0.01, (
            f"{dataset_name}: spatial_adata.X appears normalized "
            f"({frac_non_int*100:.1f}% non-integer values in sample). "
            f"Expected raw counts."
        )

    def test_some_values_greater_than_one(self, spatial_adata, dataset_name):
        X = spatial_adata.X
        if issparse(X):
            max_val = X.data.max()
        else:
            max_val = X.max()
        assert max_val > 1, (
            f"{dataset_name}: max value in X is {max_val}. "
            f"Raw counts should have values > 1."
        )


# ---------------------------------------------------------------------------
# Test 4: compute_representation produces correct shape and non-NaN
# ---------------------------------------------------------------------------


class TestRepresentation:
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_representation_shape_and_values(
        self, spatial_adata, seeds_refbased, dataset_name, representation
    ):
        from experiments.run_seedtopic import compute_representation

        X_counts = spatial_adata.X
        var_names = np.array(spatial_adata.var_names)
        n_topics = len(seeds_refbased)

        rep_name_map = {
            "idf_pca": "idf_pca",
            "idf_lev_seed_spec": "idf_lev_seed_specificity_pca",
            "lev_pseudo_sig": "lev_pseudo_sig_pca",
        }
        rep_arg = rep_name_map[representation]

        rep = compute_representation(X_counts, var_names, seeds_refbased, rep_arg, n_topics)

        n_spots = spatial_adata.n_obs
        expected_cols = min(100, n_spots - 1, len(var_names) - 1)
        assert rep.shape == (n_spots, expected_cols), (
            f"{dataset_name}/{representation}: expected shape "
            f"({n_spots}, {expected_cols}), got {rep.shape}"
        )
        assert not np.any(np.isnan(rep)), (
            f"{dataset_name}/{representation}: representation contains NaN values"
        )
        assert not np.any(np.isinf(rep)), (
            f"{dataset_name}/{representation}: representation contains Inf values"
        )


# ---------------------------------------------------------------------------
# Test 5: compute_topic_prior produces valid probability distributions
# ---------------------------------------------------------------------------


class TestTopicPrior:
    @pytest.mark.parametrize("prior_mode", PRIOR_MODES)
    def test_prior_valid_distribution(
        self, spatial_adata, seeds_refbased, dataset_name, prior_mode
    ):
        from seededntm.util import (
            compute_topic_prior,
            compute_topic_prior_idf,
            spatial_smooth_prior,
        )

        if prior_mode in ("spatial", "idf_spatial"):
            if "spatial" not in spatial_adata.obsm:
                pytest.skip(f"{dataset_name}: no spatial coordinates for {prior_mode}")

        if prior_mode == "idf_spatial":
            raw_prior = compute_topic_prior_idf(spatial_adata, seeds_refbased, temperature=0.8)
            coords = np.asarray(spatial_adata.obsm["spatial"])
            topic_prior = spatial_smooth_prior(raw_prior, coords, k_neighbors=6)
        elif prior_mode == "spatial":
            raw_prior = compute_topic_prior(spatial_adata, seeds_refbased, temperature=0.8)
            coords = np.asarray(spatial_adata.obsm["spatial"])
            topic_prior = spatial_smooth_prior(raw_prior, coords, k_neighbors=6)
        else:
            topic_prior = compute_topic_prior(spatial_adata, seeds_refbased, temperature=0.8)

        assert topic_prior.shape == (spatial_adata.n_obs, len(seeds_refbased)), (
            f"{dataset_name}/{prior_mode}: expected shape "
            f"({spatial_adata.n_obs}, {len(seeds_refbased)}), got {topic_prior.shape}"
        )
        assert np.all(topic_prior >= 0), (
            f"{dataset_name}/{prior_mode}: prior contains negative values"
        )
        row_sums = topic_prior.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-3), (
            f"{dataset_name}/{prior_mode}: prior rows don't sum to 1. "
            f"Range: [{row_sums.min():.4f}, {row_sums.max():.4f}]"
        )
        assert not np.any(np.isnan(topic_prior)), (
            f"{dataset_name}/{prior_mode}: prior contains NaN values"
        )


# ---------------------------------------------------------------------------
# Test 6: SeededNTM model instantiation with use_nb_obs=True
# ---------------------------------------------------------------------------


class TestModelInstantiation:
    def test_model_instantiates_nb_obs(self, spatial_adata, seeds_refbased, dataset_name):
        import torch
        from seededntm.model import SeededNTM

        n_topics = len(seeds_refbased)
        n_genes = spatial_adata.n_vars
        input_dim = 100

        var_names = list(spatial_adata.var_names)
        condition_mask = np.zeros((n_topics, n_genes), dtype=bool)
        for ct_name, record in seeds_refbased.items():
            feat_list = record["features"]
            for g in feat_list:
                if g in var_names:
                    condition_mask[record["topic_index"], var_names.index(g)] = True

        X = spatial_adata.X
        if issparse(X):
            init_bg = X / X.sum(axis=1)
            init_bg = np.asarray(init_bg.mean(axis=0)).ravel()
        else:
            init_bg = X / X.sum(axis=1, keepdims=True)
            init_bg = init_bg.mean(axis=0)
        init_bg = np.log(init_bg + 1e-15)

        device = torch.device("cpu")
        model = SeededNTM(
            input_dim=input_dim,
            out_dim_rna=n_genes,
            init_bg_count=init_bg,
            condition_mask=condition_mask,
            n_topics=n_topics,
            enc_hid_dim=64,
            dropout=0.2,
            wt_fusion_top_seed=1.0,
            use_nb_obs=True,
            is_group_mode=False,
            pos_scale=0.5,
            device=device,
        )
        assert model is not None
        assert model.use_nb_obs is True
        assert model.n_topics == n_topics


# ---------------------------------------------------------------------------
# Test 7: Single training step runs without error
# ---------------------------------------------------------------------------


class TestTrainingStep:
    def test_single_epoch_no_crash(self, spatial_adata, seeds_refbased, dataset_dir, dataset_name):
        """Run 1 epoch of training to verify the full pipeline works."""
        import tempfile
        import torch
        from seededntm.main import do_exp
        from experiments.run_seedtopic import compute_representation
        from seededntm.util import compute_topic_prior

        n_topics = len(seeds_refbased)
        X_counts = spatial_adata.X
        var_names = np.array(spatial_adata.var_names)

        rep = compute_representation(X_counts, var_names, seeds_refbased, "idf_pca", n_topics)
        topic_prior = compute_topic_prior(spatial_adata, seeds_refbased, temperature=0.8)

        spatial_adata.obsm["seedtopic_input_rep"] = rep
        spatial_adata.obsm["topic_prior"] = topic_prior

        if issparse(spatial_adata.X):
            spatial_adata.obsm["rna_count"] = spatial_adata.X.toarray()
        else:
            spatial_adata.obsm["rna_count"] = spatial_adata.X.copy()

        seed_path = str(dataset_dir / "seeds.json")

        with tempfile.TemporaryDirectory() as tmpdir:
            h5ad_path = str(Path(tmpdir) / "test_adata.h5ad")
            spatial_adata.write_h5ad(h5ad_path)
            exp_outdir = str(Path(tmpdir) / "output")

            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            do_exp(
                adata_h5ad_path=h5ad_path,
                condition_feat_path=seed_path,
                key_input="seedtopic_input_rep",
                key_count_out="rna_count",
                key_topic_prior="topic_prior",
                n_topics=n_topics,
                reg_topic_prior=0.99,
                wt_fusion_top_seed=1.0,
                batch_size=min(512, spatial_adata.n_obs),
                learning_rate=0.01,
                n_epochs=1,
                use_nb_obs=True,
                early_stop=False,
                exp_outdir=exp_outdir,
                device=device,
            )

            df_topic_path = Path(exp_outdir) / "df_topic.csv"
            assert df_topic_path.exists(), (
                f"{dataset_name}: training did not produce df_topic.csv"
            )
            df_topic = pd.read_csv(df_topic_path, index_col=0)
            assert df_topic.shape == (spatial_adata.n_obs, n_topics), (
                f"{dataset_name}: df_topic shape mismatch. "
                f"Expected ({spatial_adata.n_obs}, {n_topics}), got {df_topic.shape}"
            )


# ---------------------------------------------------------------------------
# Test 8: Spatial coordinates exist
# ---------------------------------------------------------------------------


class TestSpatialCoordinates:
    def test_spatial_obsm_exists(self, spatial_adata, dataset_name):
        assert "spatial" in spatial_adata.obsm, (
            f"{dataset_name}: adata.obsm['spatial'] not found. "
            f"Available keys: {list(spatial_adata.obsm.keys())}"
        )

    def test_spatial_shape(self, spatial_adata, dataset_name):
        if "spatial" not in spatial_adata.obsm:
            pytest.skip(f"{dataset_name}: no spatial coordinates")
        coords = spatial_adata.obsm["spatial"]
        assert coords.shape[0] == spatial_adata.n_obs, (
            f"{dataset_name}: spatial coords rows ({coords.shape[0]}) != "
            f"n_obs ({spatial_adata.n_obs})"
        )
        assert coords.shape[1] >= 2, (
            f"{dataset_name}: spatial coords should have >=2 columns, got {coords.shape[1]}"
        )

    def test_spatial_no_nan(self, spatial_adata, dataset_name):
        if "spatial" not in spatial_adata.obsm:
            pytest.skip(f"{dataset_name}: no spatial coordinates")
        coords = np.asarray(spatial_adata.obsm["spatial"])
        assert not np.any(np.isnan(coords)), (
            f"{dataset_name}: spatial coordinates contain NaN"
        )


# ---------------------------------------------------------------------------
# Test 9: Ground truth exists and matches adata
# ---------------------------------------------------------------------------


class TestGroundTruth:
    def test_ground_truth_exists(self, dataset_dir, dataset_name):
        gt_path = dataset_dir / "ground_truth.csv"
        assert gt_path.exists(), (
            f"{dataset_name}: ground_truth.csv not found at {gt_path}"
        )

    def test_ground_truth_indices_match(self, dataset_dir, spatial_adata, dataset_name):
        gt_path = dataset_dir / "ground_truth.csv"
        if not gt_path.exists():
            pytest.skip(f"{dataset_name}: ground_truth.csv not found")

        gt = pd.read_csv(gt_path)
        assert "ground_truth" in gt.columns, (
            f"{dataset_name}: ground_truth.csv missing 'ground_truth' column. "
            f"Columns: {gt.columns.tolist()}"
        )
        assert len(gt) == spatial_adata.n_obs, (
            f"{dataset_name}: ground_truth.csv has {len(gt)} rows but "
            f"spatial_adata has {spatial_adata.n_obs} spots. Must match."
        )
        n_unique = gt["ground_truth"].nunique()
        assert n_unique > 1, (
            f"{dataset_name}: ground_truth has only {n_unique} unique label(s)"
        )
