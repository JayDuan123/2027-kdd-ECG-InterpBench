"""Shared styling for paper figures."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARIAL_DIR = ROOT.parent / ".local/share/fonts/arial-mscorefonts"
ARIAL_FILES = ("Arial.TTF", "Arialbd.TTF", "Ariali.TTF", "Arialbi.TTF")


def configure_paper_fonts() -> None:
    """Register and require Arial for Matplotlib paper figures."""
    import matplotlib
    from matplotlib import font_manager

    missing = [name for name in ARIAL_FILES if not (DEFAULT_ARIAL_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"missing Arial font files in {DEFAULT_ARIAL_DIR}: {missing}")
    for name in ARIAL_FILES:
        font_manager.fontManager.addfont(str(DEFAULT_ARIAL_DIR / name))
    resolved = Path(font_manager.findfont("Arial", fallback_to_default=False)).resolve()
    if resolved.parent != DEFAULT_ARIAL_DIR.resolve():
        raise RuntimeError(f"Arial resolved to unexpected font: {resolved}")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )
