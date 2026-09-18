"""Where the figure generators read and write.

In:  data/raw/ for everything tracked here, and a local persona-cloud store for the rest.
Out: nothing; every other module in this package asks for its paths here.
Feeds: every figure of the paper.

The tracked inputs sit in `data/raw/`. The persona clouds, the per-role vectors and the sweep cache
built from them are too large to track, so they live in a store outside the repository, at
`$PERSONA_EM_DATA` or `~/persona-em-data`. A generator that needs one of them calls the resolver
below, which raises with the pull instructions from `geometry/README.md` when the store is absent.
"""
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"

PAPER = pathlib.Path(os.environ.get("PAPER") or pathlib.Path.home() / "work/papers/persona_misalign")
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"

STORE = pathlib.Path(os.environ.get("PERSONA_EM_DATA") or pathlib.Path.home() / "persona-em-data")

_PULL = ("It is not tracked in this repository. geometry/README.md gives the two commands that "
         "build it from the HuggingFace dataset unrulyabstractions/em-assistant-axis, and "
         "PERSONA_EM_DATA points at the store it lands in (now {store}).")


def _need(path, what):
    if not path.exists():
        raise SystemExit(f"missing {what}: {path}\n" + _PULL.format(store=STORE))
    return path


def clouds():
    """The mirror of the dataset's per-role vectors, laid out as the dataset is."""
    return _need(STORE / "hf", "the persona-cloud mirror")


def cloud_store():
    """The store's root, which also holds the per-organism runs pulled from their own repositories."""
    return _need(STORE, "the persona-cloud store")


class _Store:
    """A directory in the store that is checked the moment a file under it is named.

    The generators hold their cache directory as a plain path and join a file onto it deep inside
    a loop. Checking here turns a bare FileNotFoundError from there into the pull instructions.
    """

    def __init__(self, path, what):
        self._path, self._what = path, what

    def __truediv__(self, other):
        return _need(self._path, self._what) / other

    def __str__(self):
        return str(self._path)

    def exists(self):
        return self._path.exists()


# One .npz per family and organism: the whole cast's vectors at the five exported depths.
SWEEP_CACHE = _Store(STORE / "excess_sweep_cache", "the sweep cache")


def geom(name):
    """One family's exported persona geometry, `geom_<family>.json`."""
    return _need(STORE / "geom" / f"{name}.json", f"the exported geometry {name}")


def geom_files():
    """Every exported family geometry, sorted."""
    return sorted(_need(STORE / "geom", "the exported geometry").glob("geom_*.json"))
