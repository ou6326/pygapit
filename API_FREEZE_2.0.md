# pyGAPIT 2.0 public API freeze

This audit compares the public surface of `v1.2.4` with the current main
branch. It records the API decisions that must remain stable through the 2.0
release candidates. Changes to a frozen item require an explicit update to
this document, the migration guide, and the public-contract tests.

The version and citation metadata are deliberately outside this audit. They
will change only when a 2.0 release candidate is prepared.

## Compatibility decisions

| Area | Decision for 2.0 | Rationale |
|---|---|---|
| Statistical model functions and result classes | Preserve | Their names and statistical meanings remain the stable scientific API. Accepting `GenotypeStore` in addition to arrays is an additive extension. |
| `marker_workspace_mib` | Preserve the name and units | The name already exists in 1.2.4 and consistently means the target MiB for a temporary marker/genotype block. Renaming it would create broad churn without clarifying the resource contract. |
| Existing `GenotypeStore` protocols and NumPy/HDF5 backends | Preserve and freeze | The thin `shape` plus `read_markers()` contract already exists in 1.2.4 and now supports the complete disk-backed model workflow. |
| Zarr storage and streaming imports | Add and freeze | These are the new 2.0 storage capabilities and need stable construction, metadata, and fallback semantics. |
| Visualization | Breaking replacement and freeze | Backend-native plotting functions are replaced by typed `Visualization` objects that retain a HoloViews specification and a per-object display backend. |
| Python support | Breaking change | 2.0 requires Python 3.11 or newer; the 1.2.x line remains the Python 3.10-compatible line. |

## Frozen top-level surface

The names in `pygapit.__all__` are the primary public API. The explicitly
exported names in `pygapit.io`, `pygapit.gwas`, `pygapit.gs`,
`pygapit.stats`, and `pygapit.visualization` are also public. Their export
sets are protected by `tests/test_public_api_freeze.py`.

Underscore-prefixed modules and names remain implementation details. A public
object's documented attributes and result-field meanings remain part of its
contract even when its implementation lives in an underscore-prefixed module.

## Storage contract

The following interfaces are frozen for 2.0:

- `GenotypeStore.shape` returns `(sample_count, marker_count)`.
- `GenotypeStore.read_markers(marker_slice, sample_indices=None)` returns a
  two-dimensional, float64-compatible, read-only block, preserving requested
  sample and marker order.
- `LabeledGenotypeStore` adds `taxa`, `marker_ids`, `chromosomes`, and
  `positions`, aligned to the two matrix dimensions.
- `MarkerChunkedGenotypeStore.marker_chunk_size` reports a positive physical
  marker chunk width when known, otherwise `None`.
- `GenotypeView` composes sample and marker selections without changing the
  parent store or exposing backend-specific behavior.
- `open_genotype_store(path, backend=...)` supports `"auto"`, `"numpy"`,
  `"hdf5"`, and `"zarr"`, with overloads returning the selected concrete
  backend.
- `write_genotype_store(path, genotype, backend=...,
  marker_chunk_size=...)` accepts labeled in-memory data, stores, and internal
  streaming sources. Explicit optional backends fail clearly when their
  dependency is absent. Automatic output prefers HDF5, then Zarr, then the
  dependency-free NumPy backend.
- `import_hapmap_genotype_store()` and `import_numeric_genotype_store()` keep
  the destination store path first, matching the write functions.

The backend-specific `write_numpy_genotype()`, `write_hdf5_genotype()`, and
`write_zarr_genotype()` functions remain public for callers that need an
explicit physical format.

## Visualization contract

The following behavior is frozen for 2.0:

- Plot constructors return `Visualization[BackendT]`, not backend-native
  figures and not bare HoloViews objects.
- `Visualization.backend` is a per-object notebook preference. Changing it
  does not mutate HoloViews global state.
- `Visualization.specification` exposes the underlying HoloViews graph for
  advanced composition.
- `Visualization.render()` returns a backend-native figure on demand and
  rejects a renderer outside `supported_backends`.
- Two-dimensional plots support Matplotlib, Bokeh, and Plotly. `pca_plot_3d()`
  supports Matplotlib and Plotly because HoloViews has no Bokeh `Scatter3D`.
- `save_plot()` uses an explicit backend when supplied, Matplotlib for static
  image suffixes or styles, and otherwise the visualization's preferred
  backend. Seaborn and SciencePlots styles apply only to Matplotlib.

The following 1.2.4 APIs are intentionally removed rather than deprecated:

| Removed in 2.0 | Replacement |
|---|---|
| `manhattan_plot()` | `manhattan()` plus `save_plot()` when output is needed |
| `manhattan_interactive()` | `manhattan(..., backend="bokeh" | "plotly")` |
| `pca_plot_3d_interactive()` | `pca_plot_3d()` |
| plotting `save_path=` parameters | construct a `Visualization`, then call `save_plot()` |

No compatibility wrappers will be added for these names. This keeps backend
selection and output at one typed boundary and avoids carrying two plotting
models into the 2.x line.

## Release-hardening gates

Before `2.0.0rc1`:

1. the backend-by-model parity suite must pass for ndarray, NumPy mmap, HDF5,
   and Zarr;
2. the GAPIT 3.5 end-to-end parity audit must classify every difference as a
   bug or an intentional documented divergence;
3. large-data runs must report wall time and process peak RSS for import,
   kinship, PCA, model scans, and Manhattan rendering;
4. minimal, `bigdata`, and `styles` installations must pass their applicable
   tests on the supported Python range; and
5. `MIGRATING.md` and `CHANGELOG.md` must describe the final 1.2.4 to 2.0
   contract rather than the implementation chronology.
