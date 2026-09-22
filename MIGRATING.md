# Migrating from pyGAPIT 1.2.4

pyGAPIT 2.0 replaces backend-specific plotting functions with one
HoloViews-based visualization contract, requires Python 3.12 or newer, and
corrects GAPIT 3.5 statistical parity issues in FarmCPU and sBLUP. The
statistical model interfaces remain available; the breaking changes described
here are concentrated in visualization, supported runtimes, and the corrected
result values.

## Python and dependencies

- Upgrade to Python 3.12 or newer. CI covers Python 3.12 through 3.14.
- HoloViews, Matplotlib, Bokeh, Plotly, and Datashader are runtime dependencies.
- HDF5 and Zarr remain optional under `pygapit-ng[bigdata]`.
- Seaborn and SciencePlots remain optional under `pygapit-ng[styles]`.

## Plot construction and output

Plotting functions now return typed `Visualization` objects that retain a
HoloViews specification and an independent notebook backend preference. They
no longer save files or return backend-native figures during construction.

```python
from pygapit import manhattan, save_plot

plot = manhattan(snp_names, chromosomes, positions, p_values, backend="bokeh")
plot  # Interactive Bokeh output in Jupyter.

# Change only this object's notebook backend.
plot.backend = "plotly"

# Backend-native figures, when direct access is needed.
matplotlib_figure = plot.render("matplotlib")
plotly_figure = plot.render("plotly")

# HTML uses the object's preferred backend. Static suffixes select Matplotlib.
save_plot(plot, "manhattan.html")
save_plot(plot, "manhattan.pdf", style="science")
```

The underlying plot specification remains backend-neutral and is available as
`plot.specification` for advanced HoloViews composition. Matplotlib is suited
to static report output; Bokeh and Plotly provide interactive output.
`pca_plot_3d()` accepts only Matplotlib or Plotly because HoloViews does not
implement `Scatter3D` for Bokeh; this restriction is enforced by type checkers
and at runtime.

## Renamed plotting functions

| 1.2.4 | 2.0 | Migration |
|---|---|---|
| `manhattan_plot(...)` | `manhattan(...)` | Build the plot, then call `save_plot()` |
| `manhattan_interactive(...)` | `manhattan(...)` | Pass `effects` and `maf`, and select Bokeh or Plotly with `backend=` |
| `pca_plot_3d_interactive(...)` | `pca_plot_3d(...)` | Use the default Plotly backend or select Matplotlib |

The following functions keep their names but now return `Visualization`
objects:

- `qq_plot()`
- `kinship_heatmap()`
- `pca_plot_2d()`
- `gs_scatter()`
- `phenotype_distribution()`

Their former `save_path=` arguments have been removed. Replace them with a
separate output call:

```python
# 1.2.4
qq_plot(p_values, save_path="qq.pdf")

# 2.0
plot = qq_plot(p_values)
save_plot(plot, "qq.pdf")
```

## Large Manhattan plots

`manhattan()` uses Datashader automatically at 250,000 markers or more. The
aggregate retains the maximum `-log10(p)` per output pixel and overlays exact
significant markers. Use `large_data="points"` to force exact points or
`large_data="aggregate"` to force aggregation.

## Disk-backed genotype data

The existing in-memory APIs continue to accept NumPy arrays. The new storage
APIs add NumPy memory-map, HDF5, and Zarr backends without requiring callers to
load the complete genotype matrix. Automatic writes prefer HDF5, then Zarr,
then the dependency-free NumPy backend.

The Zarr backend uses the Zarr Python 3.x API and currently writes Zarr format
2 stores. Install optional storage backends with:

```bash
pip install "pygapit-ng[bigdata]"
```

## Statistical result changes

Three GAPIT 3.5 parity corrections change values that 1.2.4 reported:

- Static-bin FarmCPU no longer derives a pseudo-kinship REML variance fit.
  GAPIT's static-bin path returns no variance components, so `vg`, `ve`, and
  `h2` are fixed at `0.0` instead of an unsupported estimate.
- sBLUP `PEV` for intentionally low-rank pseudo-kinship now inverts the
  kinship at GAPIT's `MASS::ginv` tolerance instead of NumPy's smaller default
  cutoff, so PEV and prediction-error values match the R reference.
- MLMM now follows GAPIT's complete forward and backward cofactor path, so the
  retained model matches the R reference.

The public function names, argument meanings, and result fields are unchanged;
only these previously incorrect values are corrected.
