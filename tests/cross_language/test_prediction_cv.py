"""Fold-local REML and ridge predictions against bundled R and direct equations."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pygapit.gs.validation import cross_validate_rrblup
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
