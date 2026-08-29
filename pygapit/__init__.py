"""
pyGAPIT - Genome Association and Prediction Integrated Tool (Python)
A complete Python reimplementation of the R GAPIT package.
"""

__version__ = "2.0.0"
__author__ = "pyGAPIT contributors (based on GAPIT by Jiabo Wang & Zhiwu Zhang)"
__license__ = "GPL-3.0"

from .gapit import GAPIT, GAPITResult
from .gs.blup import GBLUPResult, cblup, gblup, predict_new, sblup
from .gwas.blink import BLINKResult, blink_gwas
from .gwas.farmcpu import FarmCPUResult, farmcpu_gwas
from .gwas.glm import GLMResult, glm_gwas
from .gwas.mlm import MLMResult, cmlm_gwas, mlm_gwas
from .gwas.mlmm import MLMMResult, mlmm_gwas
from .io.formats import (
    GenotypeData,
    PhenotypeData,
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
    "BLINKResult",
    "FarmCPUResult",
    "GAPITResult",
    "GBLUPResult",
    "GLMResult",
    "GenotypeData",
    "MLMMResult",
    "MLMResult",
    "PhenotypeData",
    "align_taxa",
    "benjamini_hochberg",
    "blink_gwas",
    "bonferroni_threshold",
    "build_covariate_matrix",
    "cblup",
    "cmlm_gwas",
    "compute_pca",
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
    "vanraden_kinship",
    "zhang_kinship",
]
