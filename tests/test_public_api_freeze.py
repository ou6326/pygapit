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


_POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD.name
_KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY.name
_REQUIRED = inspect.Parameter.empty

type SignatureContract = tuple[tuple[str, str, object], ...]


def _signature_contract(function: Callable[..., object]) -> SignatureContract:
    return tuple(
        (parameter.name, parameter.kind.name, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    )


def test_top_level_exports_are_frozen() -> None:
    assert set(pygapit.__all__) == _TOP_LEVEL_EXPORTS


def test_subpackage_exports_are_frozen() -> None:
    for module, expected in _MODULE_EXPORTS.items():
        assert set(module.__all__) == expected


def test_storage_boundary_signatures_are_frozen() -> None:
    signatures: Iterable[tuple[Callable[..., object], SignatureContract]] = (
        (
            GenotypeStore.read_markers,
            (
                ("self", _POSITIONAL, _REQUIRED),
                ("marker_slice", _POSITIONAL, _REQUIRED),
                ("sample_indices", _POSITIONAL, None),
            ),
        ),
        (
            GenotypeView.__init__,
            (
                ("self", _POSITIONAL, _REQUIRED),
                ("parent", _POSITIONAL, _REQUIRED),
                ("sample_indices", _KEYWORD_ONLY, None),
                ("marker_indices", _KEYWORD_ONLY, None),
            ),
        ),
        (
            GenotypeView.read_markers,
            (
                ("self", _POSITIONAL, _REQUIRED),
                ("marker_slice", _POSITIONAL, _REQUIRED),
                ("sample_indices", _POSITIONAL, None),
            ),
        ),
        (
            open_genotype_store,
            (
                ("path", _POSITIONAL, _REQUIRED),
                ("backend", _KEYWORD_ONLY, "auto"),
            ),
        ),
        (
            write_genotype_store,
            (
                ("path", _POSITIONAL, _REQUIRED),
                ("genotype", _POSITIONAL, _REQUIRED),
                ("backend", _KEYWORD_ONLY, "auto"),
                ("marker_chunk_size", _KEYWORD_ONLY, 1024),
            ),
        ),
        (
            import_hapmap_genotype_store,
            (
                ("store_path", _POSITIONAL, _REQUIRED),
                ("filepath", _POSITIONAL, _REQUIRED),
                ("major_allele_zero", _KEYWORD_ONLY, False),
                ("impute_method", _KEYWORD_ONLY, "middle"),
                ("backend", _KEYWORD_ONLY, "auto"),
                ("marker_chunk_size", _KEYWORD_ONLY, 1024),
            ),
        ),
        (
            import_numeric_genotype_store,
            (
                ("store_path", _POSITIONAL, _REQUIRED),
                ("gd_path", _POSITIONAL, _REQUIRED),
                ("gm_path", _POSITIONAL, _REQUIRED),
                ("impute_method", _KEYWORD_ONLY, "middle"),
                ("backend", _KEYWORD_ONLY, "auto"),
                ("marker_chunk_size", _KEYWORD_ONLY, 1024),
                ("marker_workspace_mib", _KEYWORD_ONLY, 32.0),
            ),
        ),
    )
    for function, expected in signatures:
        assert _signature_contract(function) == expected


def test_visualization_boundary_signatures_are_frozen() -> None:
    assert _signature_contract(manhattan) == (
        ("snp_names", _POSITIONAL, _REQUIRED),
        ("chromosomes", _POSITIONAL, _REQUIRED),
        ("positions", _POSITIONAL, _REQUIRED),
        ("p_values", _POSITIONAL, _REQUIRED),
        ("title", _KEYWORD_ONLY, "Manhattan Plot"),
        ("significance_threshold", _KEYWORD_ONLY, None),
        ("suggestive_threshold", _KEYWORD_ONLY, None),
        ("highlight_snps", _KEYWORD_ONLY, None),
        ("effects", _KEYWORD_ONLY, None),
        ("maf", _KEYWORD_ONLY, None),
        ("figsize", _KEYWORD_ONLY, (14, 5)),
        ("point_size", _KEYWORD_ONLY, 1.5),
        ("large_data", _KEYWORD_ONLY, "auto"),
        ("backend", _KEYWORD_ONLY, "bokeh"),
    )
    assert _signature_contract(pca_plot_3d) == (
        ("scores", _POSITIONAL, _REQUIRED),
        ("var_explained", _POSITIONAL, _REQUIRED),
        ("taxa", _POSITIONAL, None),
        ("groups", _POSITIONAL, None),
        ("title", _POSITIONAL, "3D PCA"),
        ("backend", _KEYWORD_ONLY, "plotly"),
    )

    # Static lookup keeps this runtime contract check independent of overload
    # resolution performed by the type checker.
    render = inspect.getattr_static(Visualization, "render")
    output = inspect.getattr_static(pygapit.visualization, "save_plot")
    assert callable(render)
    assert callable(output)
    assert _signature_contract(render) == (
        ("self", _POSITIONAL, _REQUIRED),
        ("backend", _POSITIONAL, None),
    )
    assert _signature_contract(output) == (
        ("plot", _POSITIONAL, _REQUIRED),
        ("path", _POSITIONAL, _REQUIRED),
        ("backend", _KEYWORD_ONLY, None),
        ("style", _KEYWORD_ONLY, "pygapit"),
    )


def test_removed_plotting_api_is_not_reexported() -> None:
    removed = {
        "manhattan_interactive",
        "manhattan_plot",
        "pca_plot_3d_interactive",
    }
    assert removed.isdisjoint(pygapit.__all__)
    assert removed.isdisjoint(pygapit.visualization.__all__)
