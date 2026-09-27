"""Optional chart rendering for query results.

A chart is only attempted when the result has a shape that plots meaningfully:
two columns, where the second is numeric. Anything else returns ``None`` and the
caller shows the table alone.
"""

from __future__ import annotations

from numbers import Number
from pathlib import Path

from .db import QueryResult


def is_chartable(result: QueryResult) -> bool:
    """True when the result is a two-column, numeric-valued series."""
    if len(result.columns) != 2 or not result.rows:
        return False
    return all(isinstance(row[1], Number) and row[1] is not None for row in result.rows)


def render_bar_chart(result: QueryResult, path: str | Path, title: str = "") -> Path | None:
    """Render a bar chart of a two-column result to ``path``.

    Returns the written path, or ``None`` if the result is not chartable. Uses a
    non-interactive backend so it works headlessly.
    """
    if not is_chartable(result):
        return None

    import matplotlib

    matplotlib.use("Agg")  # headless: no display required
    import matplotlib.pyplot as plt

    labels = [str(row[0]) for row in result.rows]
    values = [row[1] for row in result.rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#3b6ea5")
    ax.set_xlabel(result.columns[0])
    ax.set_ylabel(result.columns[1])
    if title:
        ax.set_title(title)
    if len(labels) > 6:
        plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
