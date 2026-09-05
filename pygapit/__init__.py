"""GAPIT-style genomic association and prediction tools for Python."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pygapit-ng")
except PackageNotFoundError:  # source tree imported without installation
    __version__ = "0+unknown"
__author__ = "pyGAPIT contributors (based on GAPIT by Jiabo Wang & Zhiwu Zhang)"
__license__ = "GPL-3.0"

from .gapit import GAPIT, GAPITOutputFiles, GAPITResult, ModelRunResult
from .gs.blup import (
    GBLUPResult,
    SUPERSelectionResult,
    cblup,
    gblup,
    predict_new,
    sblup,
    select_super_qtns,
)
from .gs.validation import (
    PredictionCVResult,
    cross_validate_gblup,
    cross_validate_rrblup,
)
from .gwas.blink import BLINKResult, blink_gwas
from .gwas.farmcpu import FarmCPUResult, farmcpu_gwas
from .gwas.glm import GLMResult, glm_gwas
from .gwas.mlm import MLMResult, cmlm_gwas, mlm_gwas
from .gwas.mlmm import MLMMResult, mlmm_gwas
from .io.formats import (
    AlignedData,
    GenotypeData,
    PhenotypeData,
    align_inputs,
    align_taxa,
    maf_filter,
    read_hapmap,
    read_numeric,
    read_phenotype,
)
from .stats.emma import emma_remle, emmax_p3d
from .stats.kinship import vanraden_kinship, zhang_kinship
from .stats.pca import build_covariate_matrix, compute_pca
from .stats.testing import (
    benjamini_hochberg,
    bonferroni_threshold,
    genomic_inflation_factor,
    get_significant_snps,
)
from .visualization.plots import (
    gs_scatter,
    kinship_heatmap,
    manhattan_interactive,
    manhattan_plot,
    pca_plot_2d,
    pca_plot_3d_interactive,
    phenotype_distribution,
    qq_plot,
)

__all__ = [
    "GAPIT",
    "AlignedData",
    "BLINKResult",
    "FarmCPUResult",
    "GAPITOutputFiles",
    "GAPITResult",
    "GBLUPResult",
    "GLMResult",
    "GenotypeData",
    "MLMMResult",
    "MLMResult",
    "ModelRunResult",
    "PhenotypeData",
    "PredictionCVResult",
    "SUPERSelectionResult",
    "align_inputs",
    "align_taxa",
    "benjamini_hochberg",
    "blink_gwas",
    "bonferroni_threshold",
    "build_covariate_matrix",
    "cblup",
    "cmlm_gwas",
    "compute_pca",
    "cross_validate_gblup",
    "cross_validate_rrblup",
    "emma_remle",
    "emmax_p3d",
    "farmcpu_gwas",
    "gblup",
    "genomic_inflation_factor",
    "get_significant_snps",
    "glm_gwas",
    "gs_scatter",
    "kinship_heatmap",
    "maf_filter",
    "manhattan_interactive",
    "manhattan_plot",
    "mlm_gwas",
    "mlmm_gwas",
    "pca_plot_2d",
    "pca_plot_3d_interactive",
    "phenotype_distribution",
    "predict_new",
    "qq_plot",
    "read_hapmap",
    "read_numeric",
    "read_phenotype",
    "sblup",
    "select_super_qtns",
    "vanraden_kinship",
    "zhang_kinship",
]
