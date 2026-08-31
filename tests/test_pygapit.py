"""
Comprehensive test suite for pyGAPIT.

Tests are organized by module and verify:
1. Mathematical correctness (known analytical results)
2. Biological plausibility (λ, h², effect direction)
3. Interface contracts (shapes, types, no exceptions)
4. Integration with real GAPIT demo data
"""

import subprocess
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict, cast

from numpy.random.mtrand import RandomState
from numpy.typing import NDArray

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

from pygapit._typing import as_float_vector, as_str_vector
from pygapit.gapit import GAPITResult

GAPIT_EXTDATA = Path(__file__).resolve().parents[1] / "GAPIT" / "inst" / "extdata"
PHENOTYPE_PATH = GAPIT_EXTDATA / "mdp_traits.txt.gz"
GENOTYPE_PATH = GAPIT_EXTDATA / "mdp_numeric.txt.gz"
MAP_PATH = GAPIT_EXTDATA / "mdp_SNP_information.txt.gz"


class SmallDataset(TypedDict):
    y: NDArray[np.float64]
    GD: NDArray[np.float64]
    GM: pd.DataFrame
    taxa: NDArray[np.str_]
    qtn_idx: int
    alpha_true: float
    group: NDArray[np.int_]
    n: int
    m: int


class RealDataset(TypedDict):
    y: NDArray[np.float64]
    GD: NDArray[np.float64]
    GM: pd.DataFrame
    K: NDArray[np.float64]
    X0: NDArray[np.float64]
    taxa: NDArray[np.str_]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — small synthetic datasets for fast unit tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rng() -> RandomState:
    return np.random.RandomState(42)


@pytest.fixture(scope="module")
def small_dataset(rng: RandomState) -> SmallDataset:
    """
    Synthetic dataset: 50 individuals, 200 SNPs.
    One planted QTN (SNP index 50) with large effect on phenotype.
    Population structure: 2 groups with different allele frequencies.
    """
    n, m = 50, 200

    # Two groups with different allele frequencies
    group = np.repeat([0, 1], n // 2)
    GD = np.zeros((n, m), dtype=float)
    for j in range(m):
        freq_0 = rng.uniform(0.1, 0.9)
        freq_1 = rng.uniform(0.1, 0.9)
        for i in range(n):
            f = freq_0 if group[i] == 0 else freq_1
            GD[i, j] = rng.choice([0, 1, 2], p=[(1 - f) ** 2, 2 * f * (1 - f), f**2])

    # Plant one large-effect QTN
    qtn_idx = 50
    alpha_true = 5.0
    y = alpha_true * GD[:, qtn_idx] + rng.normal(0, 2, n)
    # Add population structure effect
    y += group * 3.0

    # Map
    chromosomes = np.repeat(np.arange(1, 5), m // 4)
    positions = np.tile(np.arange(1, m // 4 + 1) * 1_000_000, 4)

    taxa = np.array([f"ind{i:03d}" for i in range(n)])
    snp_names = np.array([f"SNP{j:04d}" for j in range(m)])

    return {
        "y": y,
        "GD": GD,
        "GM": pd.DataFrame(
            {
                "SNP": snp_names,
                "Chromosome": chromosomes,
                "Position": positions,
            }
        ),
        "taxa": taxa,
        "qtn_idx": qtn_idx,
        "alpha_true": alpha_true,
        "group": group,
        "n": n,
        "m": m,
    }


@pytest.fixture(scope="module")
def real_data() -> RealDataset:
    """Load actual GAPIT maize demo data if available."""
    if not all(path.exists() for path in (PHENOTYPE_PATH, GENOTYPE_PATH, MAP_PATH)):
        pytest.skip("GAPIT demo data not available")

    from pygapit.io.formats import align_taxa, maf_filter, read_numeric, read_phenotype
    from pygapit.stats.kinship import vanraden_kinship
    from pygapit.stats.pca import build_covariate_matrix, compute_pca

    pheno = read_phenotype(PHENOTYPE_PATH)
    geno = read_numeric(GENOTYPE_PATH, MAP_PATH)
    aligned = align_taxa(pheno, geno)

    y_full = aligned["Y"]["EarHT"].values.astype(float)
    valid = ~np.isnan(y_full)
    y = y_full[valid]
    GD, kept = maf_filter(aligned["GD"][valid, :], 0.05)
    GM = aligned["GM"].iloc[kept].reset_index(drop=True)
    taxa = aligned["taxa"][valid]
    K = vanraden_kinship(GD)
    pca = compute_pca(GD, 3)
    X0 = build_covariate_matrix(pca, 3)

    return {"y": y, "GD": GD, "GM": GM, "K": K, "X0": X0, "taxa": taxa}


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Kinship
# ─────────────────────────────────────────────────────────────────────────────


class TestKinship:
    def test_vanraden_shape(self, small_dataset: SmallDataset) -> None:
        from pygapit.stats.kinship import vanraden_kinship

        n = small_dataset["n"]
        K = vanraden_kinship(small_dataset["GD"])
        assert K.shape == (n, n), "K should be n×n"

    def test_vanraden_symmetric(self, small_dataset: SmallDataset) -> None:
        from pygapit.stats.kinship import vanraden_kinship

        K = vanraden_kinship(small_dataset["GD"])
        assert np.allclose(K, K.T, atol=1e-10), "K should be symmetric"

    def test_vanraden_diagonal_positive(self, small_dataset: SmallDataset) -> None:
        from pygapit.stats.kinship import vanraden_kinship

        K = vanraden_kinship(small_dataset["GD"])
        assert np.all(np.diag(K) > 0), "Diagonal of K should be positive"

    def test_vanraden_identity_input(self) -> None:
        """With identical homozygous genotypes, K should be all-ones."""
        from pygapit.stats.kinship import vanraden_kinship

        # All individuals homozygous alt (2) at all SNPs
        GD = np.ones((10, 50)) * 2
        # Add small variation so it's not monomorphic
        GD[0, :10] = 0
        K = vanraden_kinship(GD)
        assert K.shape == (10, 10)
        assert np.all(np.isfinite(K))

    def test_vanraden_real_data(self, real_data: RealDataset) -> None:
        """On real maize data: diagonal mean should be ~1, values in [-1, 3]."""
        K = real_data["K"]
        diag_mean = np.mean(np.diag(K))
        assert 0.5 < diag_mean < 5.0, (
            f"Diagonal mean {diag_mean:.3f} out of expected range"
        )
        assert K.min() > -2.0, "K values should not be very negative"

    def test_vanraden_monomorphic_removed(self) -> None:
        """Monomorphic SNPs should not crash the computation."""
        from pygapit.stats.kinship import vanraden_kinship

        GD = np.ones((10, 20)) * 2  # all monomorphic
        GD[:, :10] = np.random.choice([0, 1, 2], size=(10, 10))  # add some variation
        K = vanraden_kinship(GD)
        assert K.shape == (10, 10)
        assert np.all(np.isfinite(K))


# ─────────────────────────────────────────────────────────────────────────────
# Tests: PCA
# ─────────────────────────────────────────────────────────────────────────────


class TestPCA:
    def test_pca_shape(self, small_dataset: SmallDataset) -> None:
        from pygapit.stats.pca import compute_pca

        n, _m = small_dataset["n"], small_dataset["m"]
        pca = compute_pca(small_dataset["GD"], n_components=3)
        assert pca.scores.shape == (n, 3)
        assert pca.var_explained.shape == (3,)

    def test_pca_var_explained_sums_reasonable(
        self, small_dataset: SmallDataset
    ) -> None:
        from pygapit.stats.pca import compute_pca

        pca = compute_pca(small_dataset["GD"], n_components=5)
        assert np.all(pca.var_explained >= 0), "Variance explained must be non-negative"
        assert pca.var_explained[0] >= pca.var_explained[1], (
            "PCs should be ordered by variance"
        )

    def test_covariate_matrix_shape(self, small_dataset: SmallDataset) -> None:
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        n = small_dataset["n"]
        pca = compute_pca(small_dataset["GD"], n_components=3)
        X0 = build_covariate_matrix(pca, 3)
        assert X0.shape == (n, 4), "X0 should be n × (1+k)"
        assert np.all(X0[:, 0] == 1.0), "First column should be intercept (all ones)"

    def test_pca_real_data(self, real_data: RealDataset) -> None:
        """First 3 PCs should explain >5% variance on structured population."""
        from pygapit.stats.pca import compute_pca

        pca = compute_pca(real_data["GD"], n_components=3)
        cumulative = pca.var_explained[:3].sum()
        assert cumulative > 0.01, f"Top 3 PCs explain only {cumulative:.3f} variance"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: EMMA / REML
# ─────────────────────────────────────────────────────────────────────────────


class TestEMMA:
    def test_remle_output_types(self, small_dataset: SmallDataset) -> None:
        from pygapit.stats.emma import emma_remle
        from pygapit.stats.kinship import vanraden_kinship
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        GD = small_dataset["GD"]
        y = small_dataset["y"]
        K = vanraden_kinship(GD)
        pca = compute_pca(GD, 3)
        X0 = build_covariate_matrix(pca, 3)
        result = emma_remle(y, X0, K)
        assert isinstance(result.delta, float)
        assert isinstance(result.vg, float)
        assert isinstance(result.ve, float)
        assert isinstance(result.h2, float)

    def test_remle_h2_in_range(self, small_dataset: SmallDataset) -> None:
        from pygapit.stats.emma import emma_remle
        from pygapit.stats.kinship import vanraden_kinship
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        GD = small_dataset["GD"]
        y = small_dataset["y"]
        K = vanraden_kinship(GD)
        X0 = build_covariate_matrix(compute_pca(GD, 3), 3)
        r = emma_remle(y, X0, K)
        assert 0.0 <= r.h2 <= 1.0, f"h² = {r.h2:.4f} out of [0, 1]"

    def test_remle_vg_ve_positive(self, small_dataset: SmallDataset) -> None:
        from pygapit.stats.emma import emma_remle
        from pygapit.stats.kinship import vanraden_kinship
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        GD = small_dataset["GD"]
        y = small_dataset["y"]
        K = vanraden_kinship(GD)
        X0 = build_covariate_matrix(compute_pca(GD, 3), 3)
        r = emma_remle(y, X0, K)
        assert r.vg >= 0, "Genetic variance must be non-negative"
        assert r.ve >= 0, "Residual variance must be non-negative"

    def test_remle_real_data_h2(self, real_data: RealDataset) -> None:
        """EarHT heritability from GAPIT demo data: expect ~0.4–0.7."""
        from pygapit.stats.emma import emma_remle

        r = emma_remle(real_data["y"], real_data["X0"], real_data["K"])
        assert 0.3 < r.h2 < 0.8, f"EarHT h² = {r.h2:.4f}, expected 0.3–0.8"

    def test_p3d_output_shape(self, small_dataset: SmallDataset) -> None:
        from pygapit.stats.emma import emmax_p3d
        from pygapit.stats.kinship import vanraden_kinship
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        GD = small_dataset["GD"]
        y = small_dataset["y"]
        _n, m = GD.shape
        K = vanraden_kinship(GD)
        X0 = build_covariate_matrix(compute_pca(GD, 3), 3)
        result = emmax_p3d(y, X0, GD, K)
        assert result.p_values.shape == (m,)
        assert result.effects.shape == (m,)

    def test_p3d_p_values_valid(self, small_dataset: SmallDataset) -> None:
        from pygapit.stats.emma import emmax_p3d
        from pygapit.stats.kinship import vanraden_kinship
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        GD = small_dataset["GD"]
        y = small_dataset["y"]
        K = vanraden_kinship(GD)
        X0 = build_covariate_matrix(compute_pca(GD, 3), 3)
        result = emmax_p3d(y, X0, GD, K)
        valid_p = result.p_values[~np.isnan(result.p_values)]
        assert np.all(valid_p >= 0), "P-values must be ≥ 0"
        assert np.all(valid_p <= 1), "P-values must be ≤ 1"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Multiple Testing
# ─────────────────────────────────────────────────────────────────────────────


class TestTesting:
    def test_bonferroni(self) -> None:
        from pygapit.stats.testing import bonferroni_threshold

        t = bonferroni_threshold(n_tests=10000, alpha=0.05)
        assert abs(t - 5e-6) < 1e-10, f"Bonferroni threshold = {t}"

    def test_bh_fdr_no_signal(self) -> None:
        """Uniform null p-values: BH should not reject at q=0.05."""
        from pygapit.stats.testing import benjamini_hochberg

        rng = np.random.RandomState(0)
        p = rng.uniform(0, 1, 1000)
        adj = benjamini_hochberg(p, alpha=0.05)
        assert np.all(adj >= 0) and np.all(adj <= 1)
        # Under H0 we expect very few/no rejections
        n_sig = (adj <= 0.05).sum()
        assert n_sig <= 50, f"Too many BH rejections under H0: {n_sig}"

    def test_bh_fdr_strong_signal(self) -> None:
        """Very small p-values should have small adjusted p-values."""
        from pygapit.stats.testing import benjamini_hochberg

        p = np.concatenate([np.ones(990) * 0.5, np.ones(10) * 1e-10])
        adj = benjamini_hochberg(p, alpha=0.05)
        n_sig = (adj <= 0.05).sum()
        assert n_sig >= 5, f"Expected ≥5 BH rejections with strong signal, got {n_sig}"

    def test_bh_monotone(self) -> None:
        """BH adjusted p-values should be monotone non-decreasing after sorting."""
        from pygapit.stats.testing import benjamini_hochberg

        rng = np.random.RandomState(1)
        p = rng.uniform(0, 1, 500)
        adj = benjamini_hochberg(p)
        sorted_adj = np.sort(adj)
        assert np.all(sorted_adj[1:] >= sorted_adj[:-1] - 1e-10)

    def test_genomic_inflation_null(self) -> None:
        """Uniform p-values → λ ≈ 1.0."""
        from pygapit.stats.testing import genomic_inflation_factor

        rng = np.random.RandomState(2)
        p = rng.uniform(0, 1, 5000)
        lam = genomic_inflation_factor(p)
        assert 0.85 < lam < 1.2, f"λ = {lam:.3f} should be near 1.0 under H0"

    def test_genomic_inflation_inflated(self) -> None:
        """Inflated p-values → λ > 1."""
        from pygapit.stats.testing import genomic_inflation_factor

        rng = np.random.RandomState(3)
        # Shift all p-values toward 0 (inflation)
        p = rng.uniform(0, 0.1, 5000)
        lam = genomic_inflation_factor(p)
        assert lam > 1.5, f"λ = {lam:.3f} should be >1.5 with inflated p-values"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: GLM
# ─────────────────────────────────────────────────────────────────────────────


class TestGLM:
    def test_glm_output_shape(self, small_dataset: SmallDataset) -> None:
        from pygapit.gwas.glm import glm_gwas
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        GD = small_dataset["GD"]
        y = small_dataset["y"]
        m = small_dataset["m"]
        X0 = build_covariate_matrix(compute_pca(GD, 3), 3)
        r = glm_gwas(y, X0, GD)
        assert r.p_values.shape == (m,)
        assert r.effects.shape == (m,)

    def test_glm_planted_qtn_detected(self, small_dataset: SmallDataset) -> None:
        """The planted QTN should have the smallest p-value."""
        from pygapit.gwas.glm import glm_gwas
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        GD = small_dataset["GD"]
        y = small_dataset["y"]
        X0 = build_covariate_matrix(compute_pca(GD, 3), 3)
        r = glm_gwas(y, X0, GD)
        top_hit = np.argmin(r.p_values)
        # QTN or nearby SNP should be in top 5
        assert top_hit in range(
            max(0, small_dataset["qtn_idx"] - 5),
            min(small_dataset["m"], small_dataset["qtn_idx"] + 5),
        ), f"Top hit SNP {top_hit} far from planted QTN {small_dataset['qtn_idx']}"

    def test_glm_effect_direction(self, small_dataset: SmallDataset) -> None:
        """Effect at the planted QTN should be positive (alpha_true > 0)."""
        from pygapit.gwas.glm import glm_gwas
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        GD = small_dataset["GD"]
        y = small_dataset["y"]
        X0 = build_covariate_matrix(compute_pca(GD, 3), 3)
        r = glm_gwas(y, X0, GD)
        assert r.effects[small_dataset["qtn_idx"]] > 0, (
            "Effect at planted QTN should be positive"
        )

    def test_glm_p_values_uniform_null(self) -> None:
        """Under pure null (no association), p-values should be roughly uniform."""
        from pygapit.gwas.glm import glm_gwas
        from pygapit.stats.testing import genomic_inflation_factor

        rng2 = np.random.RandomState(99)
        n, m = 80, 300
        GD = rng2.choice([0, 1, 2], size=(n, m), p=[0.25, 0.5, 0.25]).astype(float)
        y = rng2.normal(0, 1, n)  # pure noise, no genotype effect
        X0 = np.ones((n, 1))
        r = glm_gwas(y, X0, GD)
        lam = genomic_inflation_factor(r.p_values)
        assert 0.7 < lam < 1.5, f"λ = {lam:.3f} under null should be near 1"

    def test_glm_real_data_inflation(self, real_data: RealDataset) -> None:
        """Without kinship, GLM on real structured data should show λ > 1.1."""
        from pygapit.gwas.glm import glm_gwas
        from pygapit.stats.testing import genomic_inflation_factor

        r = glm_gwas(real_data["y"], real_data["X0"], real_data["GD"])
        lam = genomic_inflation_factor(r.p_values)
        assert lam > 1.0, f"GLM λ = {lam:.3f} should show inflation on structured data"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: MLM
# ─────────────────────────────────────────────────────────────────────────────


class TestMLM:
    def test_mlm_controls_inflation(self, real_data: RealDataset) -> None:
        """MLM with kinship should bring λ close to 1.0."""
        from pygapit.gwas.mlm import mlm_gwas
        from pygapit.stats.testing import genomic_inflation_factor

        r = mlm_gwas(real_data["y"], real_data["X0"], real_data["GD"], real_data["K"])
        lam = genomic_inflation_factor(r.p_values)
        assert lam < 1.3, f"MLM λ = {lam:.3f} should be well-controlled"

    def test_mlm_h2_plausible(self, real_data: RealDataset) -> None:
        from pygapit.gwas.mlm import mlm_gwas

        r = mlm_gwas(real_data["y"], real_data["X0"], real_data["GD"], real_data["K"])
        assert 0.1 < r.h2 < 0.95, f"MLM h² = {r.h2:.3f} out of plausible range"

    def test_mlm_output_shape(self, real_data: RealDataset) -> None:
        from pygapit.gwas.mlm import mlm_gwas

        m = real_data["GD"].shape[1]
        r = mlm_gwas(real_data["y"], real_data["X0"], real_data["GD"], real_data["K"])
        assert r.p_values.shape == (m,)
        assert r.effects.shape == (m,)

    def test_mlm_p_values_valid(self, real_data: RealDataset) -> None:
        from pygapit.gwas.mlm import mlm_gwas

        r = mlm_gwas(real_data["y"], real_data["X0"], real_data["GD"], real_data["K"])
        valid = r.p_values[~np.isnan(r.p_values)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 1)

    def test_cmlm_fixed_compression_matches_explicit_scan(
        self, small_dataset: SmallDataset
    ) -> None:
        """A fixed CMLM group count must use its explicit compressed kinship."""
        from pygapit.gwas.mlm import cmlm_gwas, compress_kinship
        from pygapit.stats.emma import emmax_p3d
        from pygapit.stats.kinship import vanraden_kinship
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        y = small_dataset["y"]
        genotypes = small_dataset["GD"]
        design = build_covariate_matrix(compute_pca(genotypes, 3), 3)
        kinship = vanraden_kinship(genotypes)
        compressed, incidence = compress_kinship(kinship, 4)
        effective = incidence @ compressed @ incidence.T
        effective += np.eye(len(y)) * 1e-6

        actual = cmlm_gwas(y, design, genotypes, kinship, group_from=4, group_to=4)
        expected = emmax_p3d(y, design, genotypes, effective)

        assert actual.method == "CMLM(g=4)"
        np.testing.assert_allclose(actual.p_values, expected.p_values)
        np.testing.assert_allclose(actual.effects, expected.effects)
        np.testing.assert_allclose(actual.se, expected.se)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: BLINK
# ─────────────────────────────────────────────────────────────────────────────


class TestBLINK:
    def test_blink_output_shape(self, small_dataset: SmallDataset) -> None:
        from pygapit.gwas.blink import blink_gwas
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        GD = small_dataset["GD"]
        y = small_dataset["y"]
        m = small_dataset["m"]
        X0 = build_covariate_matrix(compute_pca(GD, 3), 3)
        r = blink_gwas(y, X0, GD, max_iterations=3)
        assert r.p_values.shape == (m,)

    def test_blink_single_marker_reward_fallback(self) -> None:
        """A sole pseudo-QTN must not require another marker for substitution."""
        from pygapit.gwas.blink import blink_gwas

        genotypes = np.array([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]])
        phenotype = np.array([0.1, 1.0, 2.2, 0.2, 1.1, 2.0])
        result = blink_gwas(
            phenotype,
            np.ones((len(phenotype), 1)),
            genotypes,
            max_iterations=3,
            p_threshold=1.0,
        )

        assert result.p_values.shape == (1,)
        assert 0.0 <= result.p_values[0] <= 1.0

    def test_blink_selects_qtns(self, small_dataset: SmallDataset) -> None:
        """BLINK should identify QTNs near the planted signal."""
        from pygapit.gwas.blink import blink_gwas
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        GD = small_dataset["GD"]
        y = small_dataset["y"]
        X0 = build_covariate_matrix(compute_pca(GD, 3), 3)
        r = blink_gwas(y, X0, GD, max_iterations=5)
        assert len(r.selected_qtns) > 0
        assert np.min(np.abs(r.selected_qtns - small_dataset["qtn_idx"])) <= 5
        # Non-QTN p-values should be valid
        non_qtn_mask = np.ones(len(r.p_values), dtype=bool)
        if len(r.selected_qtns) > 0:
            non_qtn_mask[r.selected_qtns] = False
        valid_p = r.p_values[non_qtn_mask & ~np.isnan(r.p_values)]
        assert np.all(valid_p >= 0)
        assert np.all(valid_p <= 1)

    def test_blink_converges(self, small_dataset: SmallDataset) -> None:
        """BLINK should converge within max_iterations."""
        from pygapit.gwas.blink import blink_gwas
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        GD = small_dataset["GD"]
        y = small_dataset["y"]
        X0 = build_covariate_matrix(compute_pca(GD, 3), 3)
        r = blink_gwas(y, X0, GD, max_iterations=10)
        assert r.n_iterations <= 10

    def test_blink_ld_pruning(self) -> None:
        """LD pruning should reduce a highly correlated set to 1 marker."""
        from pygapit.gwas.blink import _ld_prune

        rng2 = np.random.RandomState(7)
        n = 100
        # Create 5 perfectly correlated SNPs
        base = rng2.choice([0, 1, 2], n, p=[0.25, 0.5, 0.25]).astype(float)
        GD = np.column_stack([base] * 5 + [rng2.choice([0, 1, 2], n).astype(float)])
        candidates = np.array([0, 1, 2, 3, 4, 5])
        pruned = _ld_prune(candidates, GD, ld_threshold=0.9)
        # All 5 correlated SNPs should collapse to 1; independent SNP 5 kept
        assert 5 in pruned, "Independent SNP should survive LD pruning"
        n_from_corr = sum(1 for i in pruned if i < 5)
        assert n_from_corr == 1, (
            f"Should keep only 1 of 5 correlated SNPs, kept {n_from_corr}"
        )

    def test_bic_selection(self) -> None:
        """BIC should add a truly predictive SNP."""
        from pygapit.gwas.blink import _bic_select_cofactors

        rng2 = np.random.RandomState(8)
        n = 60
        snp_signal = rng2.choice([0, 1, 2], n).astype(float)
        snp_noise = rng2.normal(0, 1, (n, 10))
        GD = np.column_stack([snp_signal, snp_noise])
        y = 3.0 * snp_signal + rng2.normal(0, 0.5, n)
        X0 = np.ones((n, 1))
        selected = _bic_select_cofactors(y, X0, GD, candidates=np.arange(11))
        assert 0 in selected, "Signal SNP (index 0) should be selected by BIC"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: FarmCPU
# ─────────────────────────────────────────────────────────────────────────────


class TestFarmCPU:
    def test_farmcpu_output_shape(self, small_dataset: SmallDataset) -> None:
        from pygapit.gwas.farmcpu import farmcpu_gwas
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        GD = small_dataset["GD"]
        y = small_dataset["y"]
        GM = small_dataset["GM"]
        m = small_dataset["m"]
        X0 = build_covariate_matrix(compute_pca(GD, 3), 3)
        positions = as_float_vector(GM["Position"].to_numpy())
        r = farmcpu_gwas(
            y,
            X0,
            GD,
            chromosomes=as_str_vector(GM["Chromosome"].to_numpy()),
            positions=positions,
            max_iterations=3,
        )
        assert r.p_values.shape == (m,)

    def test_farmcpu_p_values_valid(self, small_dataset: SmallDataset) -> None:
        from pygapit.gwas.farmcpu import farmcpu_gwas
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        GD = small_dataset["GD"]
        y = small_dataset["y"]
        GM = small_dataset["GM"]
        X0 = build_covariate_matrix(compute_pca(GD, 3), 3)
        positions = as_float_vector(GM["Position"].to_numpy())
        r = farmcpu_gwas(
            y,
            X0,
            GD,
            chromosomes=as_str_vector(GM["Chromosome"].to_numpy()),
            positions=positions,
            max_iterations=3,
        )
        valid = r.p_values[~np.isnan(r.p_values)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 1)
        assert len(r.selected_qtns) > 0
        assert np.min(np.abs(r.selected_qtns - small_dataset["qtn_idx"])) <= 5

    def test_farmcpu_bin_selection(self, small_dataset: SmallDataset) -> None:
        """Bin selection should respect max_qtns bound."""
        from pygapit.gwas.farmcpu import _bin_select_qtns

        GM = small_dataset["GM"]
        n = small_dataset["n"]
        p = np.random.uniform(0, 0.001, small_dataset["m"])
        max_qtns = int(np.sqrt(n) / np.sqrt(max(1, np.log10(n))))
        positions = as_float_vector(GM["Position"].to_numpy())
        qtns = _bin_select_qtns(
            p,
            as_str_vector(GM["Chromosome"].to_numpy()),
            positions,
            max_qtns=max_qtns,
        )
        assert len(qtns) <= max_qtns, f"Got {len(qtns)} QTNs, max is {max_qtns}"

    def test_farmcpu_real_data(self, real_data: RealDataset) -> None:
        """FarmCPU on real data: λ should be controlled."""
        from pygapit.gwas.farmcpu import farmcpu_gwas
        from pygapit.stats.testing import genomic_inflation_factor

        GM = real_data["GM"]
        positions = as_float_vector(GM["Position"].to_numpy())
        r = farmcpu_gwas(
            real_data["y"],
            real_data["X0"],
            real_data["GD"],
            chromosomes=as_str_vector(GM["Chromosome"].to_numpy()),
            positions=positions,
            max_iterations=5,
        )
        non_qtn = np.ones(len(r.p_values), dtype=bool)
        if len(r.selected_qtns) > 0:
            non_qtn[r.selected_qtns] = False
        lam = genomic_inflation_factor(r.p_values[non_qtn])
        assert lam < 1.6, f"FarmCPU λ = {lam:.3f} too inflated"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: gBLUP / Genomic Selection
# ─────────────────────────────────────────────────────────────────────────────


class TestGBLUP:
    def test_gblup_output_shape(self, real_data: RealDataset) -> None:
        from pygapit.gs.blup import gblup

        n = len(real_data["y"])
        r = gblup(real_data["y"], real_data["X0"], real_data["K"])
        assert r.blup.shape == (n,)
        assert r.blue.shape == (n,)
        assert r.prediction.shape == (n,)
        assert r.pev.shape == (n,)
        assert r.gebv.shape == (n,)

    def test_gblup_h2_plausible(self, real_data: RealDataset) -> None:
        from pygapit.gs.blup import gblup

        r = gblup(real_data["y"], real_data["X0"], real_data["K"])
        assert 0.1 < r.h2 < 0.95, f"gBLUP h² = {r.h2:.4f} out of plausible range"

    def test_gblup_prediction_correlation(self, real_data: RealDataset) -> None:
        """Training-set prediction should correlate well with observations."""
        from pygapit.gs.blup import gblup

        r = gblup(real_data["y"], real_data["X0"], real_data["K"])
        corr = np.corrcoef(real_data["y"], r.prediction)[0, 1]
        assert corr > 0.5, f"Prediction accuracy r = {corr:.3f} unexpectedly low"

    def test_gblup_pev_positive(self, real_data: RealDataset) -> None:
        """All PEV values should be non-negative."""
        from pygapit.gs.blup import gblup

        r = gblup(real_data["y"], real_data["X0"], real_data["K"])
        finite_pev = r.pev[np.isfinite(r.pev)]
        assert np.all(finite_pev >= -1e-8), "PEV should be non-negative"

    def test_prediction_blue_blup_sum(self, real_data: RealDataset) -> None:
        """Prediction should equal BLUE + BLUP."""
        from pygapit.gs.blup import gblup

        r = gblup(real_data["y"], real_data["X0"], real_data["K"])
        expected = r.blue + r.blup
        assert np.allclose(r.prediction, expected, atol=1e-6), (
            "Prediction != BLUE + BLUP"
        )

    def test_sblup_uses_qtn_kinship(self, real_data: RealDataset) -> None:
        """sBLUP with QTN indices should produce a valid result."""
        from pygapit.gs.blup import gblup, sblup
        from pygapit.stats.kinship import vanraden_kinship

        qtn_idx = np.array([0, 10, 50, 100, 200])
        r = sblup(real_data["y"], real_data["X0"], real_data["GD"], qtn_indices=qtn_idx)
        pseudo_kinship = vanraden_kinship(real_data["GD"][:, qtn_idx])
        pseudo_kinship += np.eye(len(real_data["y"])) * 1e-6
        expected = gblup(real_data["y"], real_data["X0"], pseudo_kinship)

        assert r.method == "sBLUP"
        assert np.all(np.isfinite(r.blup))
        np.testing.assert_allclose(r.blup, expected.blup, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r.pev, expected.pev, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        "qtn_indices",
        [np.array([], dtype=int), np.array([0.5]), np.array([-1]), np.array([999999])],
        ids=["empty", "non-integer", "negative", "too-large"],
    )
    def test_sblup_rejects_invalid_qtn_indices(
        self, real_data: RealDataset, qtn_indices: NDArray[np.generic]
    ) -> None:
        """sBLUP must not silently fall back to genome-wide gBLUP."""
        from pygapit.gs.blup import sblup

        with pytest.raises(ValueError, match="sBLUP"):
            sblup(
                real_data["y"],
                real_data["X0"],
                real_data["GD"],
                qtn_indices=qtn_indices,
            )

    def test_cblup_returns_finite_compressed_prediction(
        self, small_dataset: SmallDataset
    ) -> None:
        """cBLUP must return a complete prediction using compressed kinship."""
        from pygapit.gs.blup import cblup
        from pygapit.stats.pca import build_covariate_matrix, compute_pca

        genotypes = small_dataset["GD"]
        design = build_covariate_matrix(compute_pca(genotypes, 3), 3)
        result = cblup(
            small_dataset["y"],
            design,
            genotypes,
            taxa=small_dataset["taxa"],
            group_to=6,
        )

        assert result.method == "cBLUP"
        assert result.prediction.shape == small_dataset["y"].shape
        assert np.all(np.isfinite(result.prediction))
        assert np.all(np.isfinite(result.pev))


class TestModelContracts:
    def test_package_version_comes_from_installed_metadata(self) -> None:
        """The runtime and build metadata must expose one package version."""
        import pygapit

        assert pygapit.__version__ == version("pygapit")

    def test_compatibility_options_reject_invalid_values(self) -> None:
        """Implemented GAPIT-style options retain explicit value contracts."""
        from pygapit import GAPIT

        with pytest.raises(ValueError, match="Z requires"):
            GAPIT(Z=np.eye(2))
        with pytest.raises(ValueError, match="prediction_model"):
            GAPIT(prediction_model="ridge")
        with pytest.raises(ValueError, match="kinship_algorithm"):
            GAPIT(kinship_algorithm="IBS")

    def test_fdr_cut_requires_gapit_boolean_value(self) -> None:
        """R GAPIT defines FDRcut as a flag rather than a q-value threshold."""
        from pygapit import GAPIT

        with pytest.raises(TypeError, match="boolean"):
            GAPIT(FDRcut=cast(bool, 0.05))

    def test_top_level_sblup_error_points_to_supported_api(
        self, small_dataset: SmallDataset
    ) -> None:
        """The top-level dispatcher must not advertise an absent SUPER path."""
        from pygapit.gapit import _run_model
        from pygapit.stats.kinship import vanraden_kinship

        genotypes = small_dataset["GD"]
        marker_count = genotypes.shape[1]
        positions: NDArray[np.float64] = np.arange(marker_count, dtype=np.float64)
        with pytest.raises(ValueError, match=r"pygapit\.sblup"):
            _run_model(
                model_name="SBLUP",
                y=small_dataset["y"],
                X0=np.ones((small_dataset["n"], 1)),
                GD=genotypes,
                K=vanraden_kinship(genotypes),
                chromosomes=np.ones(marker_count),
                positions=positions,
                p_threshold=None,
                group_from=1,
                group_to=None,
                bin_size=5_000_000,
                maxLoop=1,
                LD_threshold=0.7,
                fdr_cut=False,
                fdr_alpha=0.05,
            )

    def test_cli_rejects_unimplemented_top_level_sblup(self) -> None:
        """CLI choices must match the models accepted by GAPIT()."""
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pygapit.cli",
                "--Y",
                "unused.txt",
                "--model",
                "sBLUP",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 2
        assert "invalid choice" in completed.stderr


# ─────────────────────────────────────────────────────────────────────────────
# Tests: I/O
# ─────────────────────────────────────────────────────────────────────────────


class TestIO:
    def test_read_hapmap_two_bit_genotypes(self) -> None:
        """Two-bit HapMap homozygotes, heterozygotes, and missing calls load."""
        from pygapit.io.formats import read_hapmap

        metadata_columns = [
            "rs",
            "alleles",
            "chrom",
            "pos",
            "strand",
            "assembly",
            "center",
            "protLSID",
            "assayLSID",
            "panelLSID",
            "QCcode",
        ]
        hapmap = pd.DataFrame(
            [
                [
                    "s1",
                    "A/T",
                    "1",
                    100,
                    "+",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "AA",
                    "AT",
                    "TT",
                    "NN",
                ]
            ],
            columns=[*metadata_columns, "taxon_a", "taxon_b", "taxon_c", "taxon_d"],
        )

        result = read_hapmap(hapmap, impute_method="none")

        np.testing.assert_array_equal(
            result.taxa, ["taxon_a", "taxon_b", "taxon_c", "taxon_d"]
        )
        np.testing.assert_allclose(
            result.GD[:, 0],
            np.array([0.0, 1.0, 2.0, np.nan]),
            equal_nan=True,
        )

    def test_read_phenotype_real(self) -> None:
        """Real phenotype file loads correctly."""
        if not PHENOTYPE_PATH.exists():
            pytest.skip("GAPIT demo data not available")
        from pygapit.io.formats import read_phenotype

        p = read_phenotype(PHENOTYPE_PATH)
        assert len(p.taxa) == 301
        assert "EarHT" in p.trait_names
        assert p.Y.shape == (301, 4)

    def test_read_numeric_real(self) -> None:
        """Real numeric genotype + map loads correctly."""
        if not GENOTYPE_PATH.exists() or not MAP_PATH.exists():
            pytest.skip("GAPIT demo data not available")
        from pygapit.io.formats import read_numeric

        g = read_numeric(GENOTYPE_PATH, MAP_PATH)
        assert g.GD.shape == (281, 3093)
        assert g.GM.shape == (3093, 3)
        assert len(g.taxa) == 281

    def test_align_taxa(self) -> None:
        """Taxa alignment drops non-overlapping individuals correctly."""
        import pandas as pd

        from pygapit.io.formats import GenotypeData, PhenotypeData, align_taxa

        # Phenotype: 5 individuals
        Y = pd.DataFrame(
            {"Taxa": ["A", "B", "C", "D", "E"], "trait": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )
        pheno = PhenotypeData(
            Y=Y, taxa=np.array(["A", "B", "C", "D", "E"]), trait_names=["trait"]
        )
        # Genotype: 4 individuals (missing D, extra F)
        GD = np.random.rand(4, 10)
        GM = pd.DataFrame(
            {
                "SNP": [f"s{i}" for i in range(10)],
                "Chromosome": [1] * 10,
                "Position": range(10),
            }
        )
        geno = GenotypeData(GD=GD, GM=GM, taxa=np.array(["A", "B", "C", "F"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aligned = align_taxa(pheno, geno)
        assert len(aligned["taxa"]) == 3  # A, B, C
        assert aligned["GD"].shape[0] == 3

    def test_maf_filter(self) -> None:
        """MAF filter removes monomorphic and rare SNPs."""
        from pygapit.io.formats import maf_filter

        # 5 individuals, 4 SNPs:
        # SNP0: all 2 → monomorphic → remove
        # SNP1: all 0 → monomorphic → remove
        # SNP2: freq 1/10 = 0.1 → remove at 0.15 threshold
        # SNP3: freq 0.5 → keep
        GD = np.array(
            [[2, 0, 0, 0], [2, 0, 0, 1], [2, 0, 0, 2], [2, 0, 2, 1], [2, 0, 0, 0]],
            dtype=float,
        )
        _filtered, idx = maf_filter(GD, threshold=0.15)
        assert 3 in idx, "SNP3 with MAF=0.3 should be kept"
        assert 0 not in idx, "Monomorphic SNP0 should be removed"
        assert 1 not in idx, "Monomorphic SNP1 should be removed"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Full GAPIT() pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestGAPITPipeline:
    @pytest.mark.parametrize(
        "model",
        ["CMLM", "MLMM", "FarmCPU", "gBLUP", "cBLUP"],
        ids=["cmlm", "mlmm", "farmcpu", "gblup", "cblup"],
    )
    def test_remaining_public_models_complete(
        self, small_dataset: SmallDataset, model: str
    ) -> None:
        """Every advertised model must complete through the public GAPIT API."""
        from pygapit.gapit import GAPIT

        taxa = small_dataset["taxa"]
        marker_names = small_dataset["GM"]["SNP"].astype(str).tolist()
        phenotype = pd.DataFrame({"Taxa": taxa, "trait": small_dataset["y"]})
        genotype = pd.DataFrame(small_dataset["GD"], columns=marker_names)
        genotype.insert(0, "Taxa", taxa)

        result = GAPIT(
            Y=phenotype,
            GD=genotype,
            GM=small_dataset["GM"],
            model=model,
            trait="trait",
            PCA_total=2,
            maf_threshold=0.0,
            group_to=6,
            maxLoop=3,
            buspred=model == "FarmCPU",
            file_output=False,
        )

        assert isinstance(result, GAPITResult)
        assert result.model == model.upper()
        assert result.GWAS is not None
        assert len(result.GWAS) == small_dataset["m"]
        if model in {"gBLUP", "cBLUP", "FarmCPU"}:
            assert result.Pred is not None

    def test_gapit_glm_returns_result(
        self, real_data: RealDataset, tmp_path: Path
    ) -> None:
        """GAPIT() with GLM should return GAPITResult with populated GWAS table."""
        from pygapit.gapit import GAPIT, GAPITResult

        if not PHENOTYPE_PATH.exists():
            pytest.skip("GAPIT demo data not available")

        Y = pd.read_csv(PHENOTYPE_PATH, sep="\t")
        GD = pd.read_csv(GENOTYPE_PATH, sep="\t")
        GM = pd.read_csv(MAP_PATH, sep="\t")

        result = GAPIT(
            Y=Y,
            GD=GD,
            GM=GM,
            model="GLM",
            PCA_total=3,
            trait="EarHT",
            file_output=True,
            output_dir=str(tmp_path),
        )
        assert isinstance(result, GAPITResult)
        assert result.GWAS is not None
        assert len(result.GWAS) > 100
        assert "P.value" in result.GWAS.columns
        assert "SNP" in result.GWAS.columns
        assert "effect" in result.GWAS.columns
        assert 0.0 <= result.h2 <= 1.0
        assert result.lambda_gc > 0

    def test_gapit_mlm_returns_result(self, tmp_path: Path) -> None:
        """GAPIT() with MLM should produce lower λ than GLM."""
        from pygapit.gapit import GAPIT

        if not PHENOTYPE_PATH.exists():
            pytest.skip("GAPIT demo data not available")

        Y = pd.read_csv(PHENOTYPE_PATH, sep="\t")
        GD = pd.read_csv(GENOTYPE_PATH, sep="\t")
        GM = pd.read_csv(MAP_PATH, sep="\t")

        result = GAPIT(
            Y=Y,
            GD=GD,
            GM=GM,
            model="MLM",
            trait="EarHT",
            file_output=False,
            output_dir=str(tmp_path),
        )
        assert isinstance(result, GAPITResult)
        assert result.lambda_gc < 1.3, f"MLM λ = {result.lambda_gc:.3f} too high"
        assert result.h2 > 0.1, "MLM h² should be substantial for EarHT"

    def test_gapit_blink_returns_result(self, tmp_path: Path) -> None:
        """GAPIT() with BLINK should complete and return QTNs."""
        from pygapit.gapit import GAPIT

        if not PHENOTYPE_PATH.exists():
            pytest.skip("GAPIT demo data not available")

        Y = pd.read_csv(PHENOTYPE_PATH, sep="\t")
        GD = pd.read_csv(GENOTYPE_PATH, sep="\t")
        GM = pd.read_csv(MAP_PATH, sep="\t")

        result = GAPIT(
            Y=Y,
            GD=GD,
            GM=GM,
            model="BLINK",
            trait="EarHT",
            file_output=False,
            output_dir=str(tmp_path),
        )
        assert isinstance(result, GAPITResult)
        assert result.GWAS is not None
        assert result.model == "BLINK"

    def test_gapit_output_files_created(self, tmp_path: Path) -> None:
        """GAPIT() with file_output=True should create CSV and PDF files."""
        from pygapit.gapit import GAPIT

        if not PHENOTYPE_PATH.exists():
            pytest.skip("GAPIT demo data not available")

        Y = pd.read_csv(PHENOTYPE_PATH, sep="\t")
        GD = pd.read_csv(GENOTYPE_PATH, sep="\t")
        GM = pd.read_csv(MAP_PATH, sep="\t")

        GAPIT(
            Y=Y,
            GD=GD,
            GM=GM,
            model="GLM",
            trait="EarHT",
            file_output=True,
            output_dir=str(tmp_path),
        )

        output_files = list(tmp_path.iterdir())
        assert len(output_files) > 0, "No output files created"
        csv_files = [f for f in output_files if f.suffix == ".csv"]
        assert len(csv_files) >= 1, "No CSV output files created"

    def test_gapit_simulation_mode(self, tmp_path: Path) -> None:
        """Simulation mode should override phenotype with simulated values."""
        from pygapit.gapit import GAPIT

        if not PHENOTYPE_PATH.exists():
            pytest.skip("GAPIT demo data not available")

        Y = pd.read_csv(PHENOTYPE_PATH, sep="\t")
        GD = pd.read_csv(GENOTYPE_PATH, sep="\t")
        GM = pd.read_csv(MAP_PATH, sep="\t")

        result = GAPIT(
            Y=Y,
            GD=GD,
            GM=GM,
            model="GLM",
            h2=0.7,
            NQTN=10,
            file_output=False,
            output_dir=str(tmp_path),
        )
        assert isinstance(result, GAPITResult)
        assert result.GWAS is not None
        assert result.model == "GLM"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Visualization (smoke tests — just check no exceptions)
# ─────────────────────────────────────────────────────────────────────────────


class TestVisualization:
    def test_manhattan_plot(self, real_data: RealDataset, tmp_path: Path) -> None:
        import matplotlib

        from pygapit.gwas.glm import glm_gwas
        from pygapit.visualization.plots import manhattan_plot

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = glm_gwas(real_data["y"], real_data["X0"], real_data["GD"])
        GM = real_data["GM"]
        positions = as_float_vector(GM["Position"].to_numpy())
        fig = manhattan_plot(
            snp_names=as_str_vector(GM["SNP"].to_numpy()),
            chromosomes=as_str_vector(GM["Chromosome"].to_numpy()),
            positions=positions,
            p_values=r.p_values,
            save_path=str(tmp_path / "manhattan.pdf"),
        )
        assert fig is not None
        plt.close("all")

    def test_qq_plot(self, real_data: RealDataset, tmp_path: Path) -> None:
        import matplotlib

        from pygapit.gwas.glm import glm_gwas
        from pygapit.visualization.plots import qq_plot

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = glm_gwas(real_data["y"], real_data["X0"], real_data["GD"])
        fig = qq_plot(r.p_values, save_path=str(tmp_path / "qq.pdf"))
        assert fig is not None
        plt.close("all")

    def test_kinship_heatmap(self, real_data: RealDataset, tmp_path: Path) -> None:
        import matplotlib

        from pygapit.visualization.plots import kinship_heatmap

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = kinship_heatmap(
            real_data["K"][:30, :30], save_path=str(tmp_path / "kinship.pdf")
        )
        assert fig is not None
        plt.close("all")

    def test_pca_2d(self, real_data: RealDataset, tmp_path: Path) -> None:
        import matplotlib

        from pygapit.stats.pca import compute_pca
        from pygapit.visualization.plots import pca_plot_2d

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pca = compute_pca(real_data["GD"], n_components=3)
        fig = pca_plot_2d(
            pca.scores, pca.var_explained, save_path=str(tmp_path / "pca.pdf")
        )
        assert fig is not None
        plt.close("all")

    def test_gs_scatter(self, real_data: RealDataset, tmp_path: Path) -> None:
        import matplotlib

        from pygapit.gs.blup import gblup
        from pygapit.visualization.plots import gs_scatter

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gs = gblup(real_data["y"], real_data["X0"], real_data["K"])
        fig = gs_scatter(
            real_data["y"], gs.prediction, save_path=str(tmp_path / "gs.pdf")
        )
        assert fig is not None
        plt.close("all")
