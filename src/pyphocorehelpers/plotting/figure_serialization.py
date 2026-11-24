"""MATLAB-.fig-like save/load for matplotlib figures.

Pickles a live ``matplotlib.figure.Figure`` (plus optional caller ``plot_data`` /
``metadata``) into a ``.mplfig`` archive so a fresh notebook can restore an
interactive figure.

Notes
-----
Custom artists, callbacks, and GUI widgets may fail to unpickle. Prefer a
matching matplotlib version when loading.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import dill as pickle
import matplotlib as mpl
from attrs import define, field, Factory
from matplotlib.figure import Figure, FigureBase

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes


MPLFIG_FORMAT_NAME: str = 'pho_mplfig'
MPLFIG_FORMAT_VERSION: int = 1
MPLFIG_SUFFIX: str = '.mplfig'


def _ensure_mplfig_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    if path.suffix == '':
        path = path.with_suffix(MPLFIG_SUFFIX)
    return path


def _coerce_figure(fig_or_container: Any) -> Figure:
    """Extract a matplotlib Figure from a Figure, MatplotlibRenderPlots-like container, or object with `.fig` / `.figure`."""
    if isinstance(fig_or_container, FigureBase):
        if isinstance(fig_or_container, Figure):
            return fig_or_container
        raise TypeError(f'Expected a top-level matplotlib.figure.Figure, got {type(fig_or_container)!r}')

    for attr_name in ('fig', 'figure'):
        if hasattr(fig_or_container, attr_name):
            candidate = getattr(fig_or_container, attr_name)
            if isinstance(candidate, Figure):
                return candidate
            # MatplotlibRenderPlots.fig may unwrap a single-item list via property
            if isinstance(candidate, (list, tuple)) and len(candidate) >= 1 and isinstance(candidate[0], Figure):
                return candidate[0]
    ## END for attr_name in ('fig', 'figure')...

    if hasattr(fig_or_container, 'figures'):
        figures = getattr(fig_or_container, 'figures')
        if isinstance(figures, (list, tuple)) and len(figures) >= 1 and isinstance(figures[0], Figure):
            return figures[0]

    raise TypeError(f'Could not extract a matplotlib.figure.Figure from {type(fig_or_container)!r}')


@metadata_attributes(short_name='MplFigArchive', tags=['matplotlib', 'figure', 'serialization', 'pickle', 'mplfig'], input_requires=[], output_provides=[], uses=[], used_by=['save_mplfig', 'load_mplfig'], creation_date='2026-11-24 00:00', related_items=[])
@define(slots=False)
class MplFigArchive:
    """Container returned by ``load_mplfig`` holding the restored figure and sidecar data.

    Usage:
        from pyphocorehelpers.plotting.figure_serialization import load_mplfig
        archive = load_mplfig('output/my_plot.mplfig')
        fig = archive.figure  # or archive.fig
        plot_data = archive.plot_data
    """
    figure: Figure
    plot_data: Dict[str, Any] = field(default=Factory(dict))
    metadata: Dict[str, Any] = field(default=Factory(dict))
    path: Optional[Path] = None
    format_version: int = MPLFIG_FORMAT_VERSION
    matplotlib_version: Optional[str] = None
    created_at: Optional[str] = None


    @property
    def fig(self) -> Figure:
        """Alias for ``figure`` (MATLAB / MatplotlibRenderPlots style)."""
        return self.figure



@function_attributes(short_name='save_mplfig', tags=['matplotlib', 'figure', 'serialization', 'pickle', 'mplfig', 'export'], input_requires=[], output_provides=['.mplfig'], uses=['dill'], used_by=[], creation_date='2026-11-24 00:00', related_items=['load_mplfig', 'MplFigArchive'])
def save_mplfig(fig: Any, path: Union[str, Path], plot_data: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None, overwrite: bool = True) -> Path:
    """Pickle a live matplotlib Figure (+ optional plot_data/metadata) to a ``.mplfig`` archive.

    Analogous to MATLAB's ``savefig`` / ``.fig`` format: restores an interactive figure in a fresh notebook via ``load_mplfig``.

    Usage:
        from pyphocorehelpers.plotting.figure_serialization import save_mplfig, load_mplfig

        save_mplfig(fig, 'output/my_plot.mplfig', plot_data={'x': x, 'y': y}, metadata={'note': 'session foo'})
        archive = load_mplfig('output/my_plot.mplfig')
        fig2 = archive.figure

    Args:
        fig: A ``matplotlib.figure.Figure``, or a container with ``.fig`` / ``.figure`` / ``.figures``.
        path: Destination path. If no suffix is given, ``.mplfig`` is appended.
        plot_data: Optional caller-supplied mapping of related arrays/objects to store alongside the figure.
        metadata: Optional plain dict (title, context string, notes, …).
        overwrite: If False and the file exists, raise FileExistsError.

    Returns:
        Path to the written ``.mplfig`` file.

    Notes:
        Custom artists, callbacks, and GUI widgets may fail to pickle/unpickle.
    """
    figure = _coerce_figure(fig)
    out_path = _ensure_mplfig_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and (not overwrite):
        raise FileExistsError(f'save_mplfig: file exists and overwrite=False: {out_path}')

    payload: Dict[str, Any] = {
        'format': MPLFIG_FORMAT_NAME,
        'format_version': MPLFIG_FORMAT_VERSION,
        'matplotlib_version': mpl.__version__,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'figure': figure,
        'plot_data': dict(plot_data) if (plot_data is not None) else {},
        'metadata': dict(metadata) if (metadata is not None) else {},
    }
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


@function_attributes(short_name='load_mplfig', tags=['matplotlib', 'figure', 'serialization', 'pickle', 'mplfig', 'load'], input_requires=['.mplfig'], output_provides=['MplFigArchive'], uses=['dill', 'PhoActiveFigureManager2D.reshow_figure'], used_by=[], creation_date='2026-11-24 00:00', related_items=['save_mplfig', 'MplFigArchive'])
def load_mplfig(path: Union[str, Path], *, attach_to_current_pyplot: bool = True) -> MplFigArchive:
    """Load a ``.mplfig`` archive; return an ``MplFigArchive`` with the restored figure.

    Usage:
        from pyphocorehelpers.plotting.figure_serialization import load_mplfig
        archive = load_mplfig('output/my_plot.mplfig')
        fig = archive.figure
        plot_data = archive.plot_data

    Args:
        path: Path to a ``.mplfig`` file (``.mplfig`` appended if missing).
        attach_to_current_pyplot: If True, register the figure with the active pyplot
            figure manager so notebook display and ``plt.show()`` work.

    Returns:
        ``MplFigArchive`` with ``.figure``, ``.plot_data``, ``.metadata``, ``.path``.

    Notes:
        Prefer a matching matplotlib version. Custom artists / callbacks may fail to unpickle.
    """
    in_path = _ensure_mplfig_path(path)
    if not in_path.exists():
        raise FileNotFoundError(f'load_mplfig: file not found: {in_path}')

    with open(in_path, 'rb') as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise TypeError(f'load_mplfig: expected a dict payload, got {type(payload)!r}')
    if payload.get('format') != MPLFIG_FORMAT_NAME:
        raise ValueError(f'load_mplfig: unrecognized format {payload.get("format")!r}; expected {MPLFIG_FORMAT_NAME!r}')

    figure = payload.get('figure')
    if not isinstance(figure, Figure):
        raise TypeError(f'load_mplfig: payload["figure"] is not a matplotlib.figure.Figure (got {type(figure)!r})')

    if attach_to_current_pyplot:
        from pyphocorehelpers.plotting.figure_management import PhoActiveFigureManager2D
        PhoActiveFigureManager2D.reshow_figure(figure)

    return MplFigArchive(
        figure=figure,
        plot_data=dict(payload.get('plot_data') or {}),
        metadata=dict(payload.get('metadata') or {}),
        path=in_path,
        format_version=int(payload.get('format_version', MPLFIG_FORMAT_VERSION)),
        matplotlib_version=payload.get('matplotlib_version'),
        created_at=payload.get('created_at'),
    )
