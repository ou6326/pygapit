"""Mechanical checks for the public surface frozen for pyGAPIT 2.0."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable

import pygapit
import pygapit.gs
import pygapit.gwas
import pygapit.io
import pygapit.stats
import pygapit.visualization
from pygapit import (
    GenotypeStore,
    GenotypeView,
    Visualization,
    import_hapmap_genotype_store,
    import_numeric_genotype_store,
    manhattan,
    open_genotype_store,
    pca_plot_3d,
    write_genotype_store,
)

_TOP_LEVEL_EXPORTS = {
    "GAPIT",
    "AlignedData",
    "ArrayGenotypeStore",
    "BLINKResult",
    "FarmCPUResult",
    "GAPITOutputFiles",
    "GAPITResult",
    "GBLUPResult",
    "GLMResult",
    "GenotypeData",
    "GenotypeStore",
    "GenotypeView",
    "HDF5GenotypeStore",
    "LabeledGenotypeStore",
    "LargeDataMode",
    "MLMMResult",
    "MLMResult",
    "MarkerChunkedGenotypeStore",
    "ModelRunResult",
    "NumpyGenotypeStore",
    "PhenotypeData",
    "PredictionCVResult",
    "RRBLUPResult",
    "SUPERSelectionResult",
    "Visualization",
    "ZarrGenotypeStore",
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
    "import_hapmap_genotype_store",
    "import_numeric_genotype_store",
    "kinship_heatmap",
    "maf_filter",
    "manhattan",
    "mlm_gwas",
    "mlmm_gwas",
    "open_genotype_store",
    "pca_plot_2d",
    "pca_plot_3d",
    "phenotype_distribution",
    "predict_new",
    "qq_plot",
    "read_hapmap",
    "read_numeric",
    "read_phenotype",
    "rrblup",
    "save_plot",
    "sblup",
    "select_super_qtns",
    "vanraden_kinship",
    "write_genotype_store",
    "write_hdf5_genotype",
    "write_numpy_genotype",
    "write_zarr_genotype",
    "zhang_kinship",
}

_MODULE_EXPORTS = {
    pygapit.io: {
        "AlignedData",
        "ArrayGenotypeStore",
        "GenotypeData",
        "GenotypeStore",
        "GenotypeView",
        "HDF5GenotypeStore",
        "LabeledGenotypeStore",
        "MarkerChunkedGenotypeStore",
        "NumpyGenotypeStore",
        "PhenotypeData",
        "ZarrGenotypeStore",
        "align_inputs",
        "align_taxa",
        "as_genotype_store",
        "import_hapmap_genotype_store",
        "import_numeric_genotype_store",
        "impute_missing",
        "maf_filter",
        "open_genotype_store",
        "read_hapmap",
        "read_numeric",
        "read_phenotype",
        "write_genotype_store",
        "write_hdf5_genotype",
        "write_numpy_genotype",
        "write_zarr_genotype",
    },
    pygapit.gwas: {
        "BLINKResult",
        "FarmCPUResult",
        "GLMResult",
        "MLMMResult",
        "MLMResult",
        "blink_gwas",
        "cmlm_gwas",
        "farmcpu_gwas",
        "glm_gwas",
        "glm_scan_with_cofactors",
        "mlm_gwas",
        "mlmm_gwas",
    },
    pygapit.gs: {
        "GBLUPResult",
        "PredictionCVResult",
        "RRBLUPResult",
        "SUPERSelectionResult",
        "cblup",
        "cross_validate_gblup",
        "cross_validate_rrblup",
        "gblup",
        "predict_new",
        "rrblup",
        "sblup",
        "select_super_qtns",
    },
    pygapit.stats: {
        "EMMAResult",
        "GWASResult",
        "PCAResult",
        "benjamini_hochberg",
        "bonferroni_threshold",
        "build_covariate_matrix",
        "compute_pca",
        "emma_remle",
        "emmax_p3d",
        "genomic_inflation_factor",
        "get_significant_snps",
        "scale_kinship",
        "vanraden_kinship",
        "zhang_kinship",
    },
    pygapit.visualization: {
        "LargeDataMode",
        "ManhattanPlotData",
        "OutputBackend",
        "StaticStyle",
        "Visualization",
        "gs_scatter",
        "kinship_heatmap",
        "manhattan",
        "pca_plot_2d",
        "pca_plot_3d",
        "phenotype_distribution",
        "prepare_genomic_axis",
        "prepare_manhattan_data",
        "qq_plot",
        "save_plot",
    },
}


def _parameter_names(function: Callable[..., object]) -> tuple[str, ...]:
    return tuple(inspect.signature(function).parameters)


def test_top_level_exports_are_frozen() -> None:
    assert set(pygapit.__all__) == _TOP_LEVEL_EXPORTS


def test_subpackage_exports_are_frozen() -> None:
    for module, expected in _MODULE_EXPORTS.items():
        assert set(module.__all__) == expected


def test_storage_boundary_signatures_are_frozen() -> None:
    signatures: Iterable[tuple[Callable[..., object], tuple[str, ...]]] = (
        (GenotypeStore.read_markers, ("self", "marker_slice", "sample_indices")),
        (
            GenotypeView.__init__,
            ("self", "parent", "sample_indices", "marker_indices"),
        ),
        (GenotypeView.read_markers, ("self", "marker_slice", "sample_indices")),
        (open_genotype_store, ("path", "backend")),
        (
            write_genotype_store,
            ("path", "genotype", "backend", "marker_chunk_size"),
        ),
        (
            import_hapmap_genotype_store,
            (
                "store_path",
                "filepath",
                "major_allele_zero",
                "impute_method",
                "backend",
                "marker_chunk_size",
            ),
        ),
        (
            import_numeric_genotype_store,
            (
                "store_path",
                "gd_path",
                "gm_path",
                "impute_method",
                "backend",
                "marker_chunk_size",
                "marker_workspace_mib",
            ),
        ),
    )
    for function, expected in signatures:
        assert _parameter_names(function) == expected


def test_visualization_boundary_signatures_are_frozen() -> None:
    assert _parameter_names(manhattan) == (
        "snp_names",
        "chromosomes",
        "positions",
        "p_values",
        "title",
        "significance_threshold",
        "suggestive_threshold",
        "highlight_snps",
        "effects",
        "maf",
        "figsize",
        "point_size",
        "large_data",
        "backend",
    )
    assert _parameter_names(pca_plot_3d) == (
        "scores",
        "var_explained",
        "taxa",
        "groups",
        "title",
        "backend",
    )

    # Static lookup keeps this runtime contract check independent of overload
    # resolution performed by the type checker.
    render = inspect.getattr_static(Visualization, "render")
    output = inspect.getattr_static(pygapit.visualization, "save_plot")
    assert callable(render)
    assert callable(output)
    assert _parameter_names(render) == ("self", "backend")
    assert _parameter_names(output) == ("plot", "path", "backend", "style")


def test_removed_plotting_api_is_not_reexported() -> None:
    removed = {
        "manhattan_interactive",
        "manhattan_plot",
        "pca_plot_3d_interactive",
    }
    assert removed.isdisjoint(pygapit.__all__)
    assert removed.isdisjoint(pygapit.visualization.__all__)
