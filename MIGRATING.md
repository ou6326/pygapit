# Migrating from pyGAPIT 1.2.4

The next major release replaces backend-specific plotting functions with one
HoloViews-based visualization contract and requires Python 3.11 or newer. The
statistical model interfaces remain available; the breaking changes described
here are concentrated in visualization and supported runtimes.

## Python and dependencies

- Upgrade to Python 3.11 or newer. CI covers Python 3.11 through 3.14.
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

| 1.2.4 | Next major release | Migration |
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

# Next major release
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
