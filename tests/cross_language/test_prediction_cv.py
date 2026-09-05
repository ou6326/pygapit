"""Fold-local REML and ridge predictions against bundled R and direct equations."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from pygapit.gs.validation import _ridge_fit, cross_validate_rrblup
from pygapit.models.genomic_prediction import RR_BLUP
from tests.cross_language.r_bridge import RBridge


def test_rrblup_cv_matches_r_reml_and_marker_equations(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
) -> None:
    r_bridge.source(r_root, "GAPIT.emma.R")
    r_bridge.source(r_root, "GAPIT.replaceNaN.R")
    r_bridge.source(r_root, "GAPIT.emma.REMLE.R")
    reference = r_bridge.function(
        "function(y, gd) {"
        " ids <- rep(0:2, each=4); pred <- numeric(length(y)); penalties <- numeric(3);"
        " for (f in 0:2) {"
        " train <- ids != f; test <- ids == f;"
        " means <- colMeans(gd[train,,drop=FALSE]);"
        " z <- sweep(gd[train,,drop=FALSE], 2, means);"
        " zp <- sweep(gd[test,,drop=FALSE], 2, means);"
        " yt <- y[train]; m <- ncol(z);"
        " fit <- GAPIT.emma.REMLE(yt, matrix(1,length(yt),1), tcrossprod(z)/m);"
        " penalty <- fit$delta * m; penalties[f+1] <- penalty;"
        " b <- solve(crossprod(z) + diag(penalty,m), crossprod(z, yt-mean(yt)));"
        " pred[test] <- mean(yt) + zp %*% b;"
        " }; list(pred=pred, penalties=penalties) }"
    )
    result = reference(
        r_bridge.float_vector(fixed_phenotype), r_bridge.matrix(fixed_genotypes)
    )
    actual = cross_validate_rrblup(fixed_phenotype, fixed_genotypes, n_folds=3)
    np.testing.assert_allclose(
        actual.predictions,
        r_bridge.float_array(r_bridge.component(result, "pred")).ravel(),
        rtol=2e-6,
        atol=1e-9,
    )
    # Bundled R calls uniroot with its default tolerance (~1.2e-4 in
    # log-delta); Python refines the root further. Predictions stay tighter.
    np.testing.assert_allclose(
        actual.regularization,
        r_bridge.float_array(r_bridge.component(result, "penalties")).ravel(),
        rtol=2e-4,
        atol=1e-9,
    )


@pytest.mark.parametrize("penalty", [None, 2.5])
@pytest.mark.parametrize("marker_copies", [1, 3])
def test_rrblup_full_fit_centering_intercept_and_effects_against_r(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
    penalty: float | None,
    marker_copies: int,
) -> None:
    """Pin the centered random component, not raw-marker score or phenotype.

    The bundled GAPIT checkout has no rrBLUP::mixed.solve implementation.
    Use its REML estimator and independent augmented mixed-model equations.
    """
    for name in ("GAPIT.emma.R", "GAPIT.replaceNaN.R", "GAPIT.emma.REMLE.R"):
        r_bridge.source(r_root, name)
    reference = r_bridge.function(
        "function(y, gd, penalty) {"
        " means <- colMeans(gd); z <- sweep(gd, 2, means); m <- ncol(z);"
        " if (is.null(penalty)) {"
        " fit <- GAPIT.emma.REMLE(y, matrix(1,length(y),1), tcrossprod(z)/m);"
        " penalty <- m * fit$delta; };"
        " design <- cbind(1,z);"
        " coef <- solve(crossprod(design) + diag(c(0,rep(penalty,m))), crossprod(design,y));"
        " effects <- coef[-1]; gebv <- z %*% effects;"
        " list(means=means, intercept=coef[1], effects=effects, gebv=gebv,"
        " prediction=coef[1]+gebv, raw_score=gd %*% effects) }"
    )
    y = fixed_phenotype + 20
    expanded = np.asarray(
        np.tile(fixed_genotypes, (1, marker_copies)), dtype=np.float64
    )
    z = expanded + np.arange(expanded.shape[1], dtype=np.float64)
    result = reference(
        r_bridge.float_vector(y),
        r_bridge.matrix(z),
        r_bridge.evaluate("NULL") if penalty is None else penalty,
    )
    # Probing at training means + unit marker changes exposes each fitted
    # effect independently, without relying on the internal solver branch.
    probes = np.vstack((z.mean(axis=0), z.mean(axis=0) + np.eye(z.shape[1])))
    gebv, predictions, _ = _ridge_fit(y, z, probes, penalty)
    public_gebv, _ = RR_BLUP(y, z, lambda_=penalty, n_folds=3)
    r_gebv = r_bridge.float_array(r_bridge.component(result, "gebv")).ravel()
    r_intercept = r_bridge.float_array(r_bridge.component(result, "intercept")).item()
    np.testing.assert_allclose(
        z.mean(axis=0), r_bridge.float_array(r_bridge.component(result, "means"))
    )
    np.testing.assert_allclose(predictions[0], r_intercept, atol=1e-10)
    np.testing.assert_allclose(
        predictions[1:] - predictions[0],
        r_bridge.float_array(r_bridge.component(result, "effects")).ravel(),
        rtol=2e-4,
        atol=1e-8,
    )
    np.testing.assert_allclose(public_gebv, r_gebv, rtol=2e-4, atol=1e-8)
    np.testing.assert_allclose(
        gebv + predictions[0],
        r_bridge.float_array(r_bridge.component(result, "prediction")).ravel(),
        rtol=2e-6,
        atol=1e-8,
    )
    assert abs(float(public_gebv.mean())) < 1e-12
    assert not np.allclose(
        public_gebv,
        r_bridge.float_array(r_bridge.component(result, "raw_score")).ravel(),
    )


def test_rrblup_training_gebv_matches_gapit_emmax_blup(
    r_bridge: RBridge,
    r_root: Path,
    fixed_genotypes: NDArray[np.float64],
    fixed_phenotype: NDArray[np.float64],
) -> None:
    """Compare the random component to GAPIT's adjacent prediction path."""
    for name in (
        "GAPIT.emma.R",
        "GAPIT.replaceNaN.R",
        "GAPIT.emma.REMLE.R",
        "GAPIT.Timmer.R",
        "GAPIT.Memory.R",
    ):
        r_bridge.source(r_root, name)
    reference = r_bridge.source_function(r_root, "GAPIT.EMMAxP3D.R", "GAPIT.EMMAxP3D")
    z = fixed_genotypes - fixed_genotypes.mean(axis=0)
    y = fixed_phenotype
    r_null = r_bridge.evaluate("NULL")
    result = reference(
        ys=r_bridge.matrix(y[np.newaxis, :]),
        xs=r_bridge.matrix(z[:, :1]),
        K=r_bridge.matrix(z @ z.T / z.shape[1]),
        X0=r_bridge.matrix(np.ones((len(y), 1))),
        CVI=r_bridge.matrix(
            np.column_stack((np.arange(len(y), dtype=np.float64), np.ones(len(y))))
        ),
        file_from=1,
        file_to=1,
        file_fragment=1,
        fullGD=True,
        SNP_P3D=True,
        Timmer=r_null,
        Memory=r_null,
        optOnly=True,
    )
    gebv, _ = RR_BLUP(y, fixed_genotypes, n_folds=3)
    np.testing.assert_allclose(
        gebv,
        r_bridge.float_array(r_bridge.component(result, "BLUP")).ravel(),
        rtol=2e-4,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        y.mean() + gebv,
        np.sum(r_bridge.float_array(r_bridge.component(result, "BLUE")), axis=1)
        + r_bridge.float_array(r_bridge.component(result, "BLUP")).ravel(),
        rtol=2e-6,
        atol=1e-8,
    )
