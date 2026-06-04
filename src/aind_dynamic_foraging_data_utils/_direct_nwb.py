"""
A dependency-free, fast drop-in for the small subset of the pynwb ``NWBFile`` API
that :mod:`aind_dynamic_foraging_data_utils.nwb_utils` relies on.

The standard reader path (``pynwb`` / ``hdmf_zarr``) eagerly materialises the entire
NWB object graph just to read a handful of tables and time series. For an AIND
dynamic-foraging session that graph includes *all* of the fiber-photometry containers,
so reading behaviour-only data (trials / events) pays for FIP it never uses. Over S3
that graph construction is the dominant cost (several seconds).

This module bypasses pynwb and reads directly with the native storage library:

* **HDF5** files are opened with ``h5py`` (the same trick used elsewhere to read NWBs
  much faster than pynwb), and
* **Zarr** files with ``zarr.open_consolidated`` - which fetches all array metadata in a
  single request via the consolidated ``.zmetadata``, instead of one request per array
  (the latter is what makes a plain ``zarr.open`` slow over S3).

Only the data actually requested (the trials table, the lick/reward time series, FIP
interfaces, scratch metadata) is read. The class :class:`DirectNWBAdapter` mimics exactly
the pynwb attribute/indexing patterns the readers use, so the downstream DataFrame-building
code in ``nwb_utils`` is reused unchanged and produces identical output.

``h5py`` and ``zarr`` are already transitive dependencies (via ``pynwb`` / ``hdmf_zarr``),
so this adds nothing to the dependency footprint.
"""

from __future__ import annotations

import datetime
import os

import numpy as np
import pandas as pd

# Internal NWB column that pynwb's ``DynamicTable.to_dataframe()`` exposes as the index.
_TABLE_INDEX_COL = "id"


def _decode(value):
    """Decode a single ``bytes`` value to ``str``; pass everything else through."""
    return value.decode() if isinstance(value, bytes) else value


def _decode_list(values):
    """Decode an iterable of possibly-bytes values to a list."""
    return [_decode(v) for v in values]


def _maybe_datetime(text):
    """Return a ``datetime`` if ``text`` is an ISO-8601 timestamp, else ``None``."""
    try:
        return datetime.datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None


def _get_node(root, path):
    """Return the dataset/group at an internal ``path`` (or ``None`` if absent).

    Works for both ``h5py`` and ``zarr`` groups, which share posix-style indexing.
    """
    try:
        return root[path.strip("/")]
    except (KeyError, ValueError):
        return None


def _decode_array(array):
    """Decode byte-string elements of an array to ``str`` (numeric arrays unchanged)."""
    array = np.asarray(array)
    if array.dtype.kind in ("O", "S"):
        return np.array([_decode(x) for x in array], dtype=object)
    return array


def _read_scalar(root, path):
    """Read a scalar/short metadata field, mirroring pynwb's interpretation.

    Handles HDF5 scalar datasets (``shape == ()``), single-element arrays (HDF5 and Zarr
    both store NWB scalars this way in places), byte decoding, and ISO-8601 -> datetime.
    Missing fields return ``None`` (as pynwb does for unset metadata).
    """
    node = _get_node(root, path)
    if node is None:
        return None
    if getattr(node, "shape", None) == ():  # HDF5 scalar dataset
        values = [node[()]]
    else:
        values = node[:]
    if len(values) == 0:
        return None
    first = _decode(values[0])
    if isinstance(first, str):
        as_dt = _maybe_datetime(first)
        if as_dt is not None and len(values) == 1:
            return as_dt
        if len(values) > 1:
            return _decode_list(values)
        return first
    if len(values) > 1:
        return np.asarray(values)
    return first


def _read_table(root, table_path):
    """Reconstruct a pynwb-equivalent DataFrame for a DynamicTable at ``table_path``."""
    group = root[table_path.strip("/")]
    colnames = _decode_list(group.attrs["colnames"])
    columns = {}
    for col in colnames:
        index_name = col + "_index"
        if index_name in group:  # ragged (VectorIndex) column -> list per row
            flat = _decode_array(group[col][:])
            ends = np.asarray(group[index_name][:])
            starts = np.concatenate([[0], ends[:-1]])
            columns[col] = [flat[start:end] for start, end in zip(starts, ends)]
        else:
            values = group[col][:]
            if getattr(values, "ndim", 1) > 1:
                # Multi-dimensional column (e.g. an N x 2 interval): pynwb's
                # to_dataframe yields one sub-array per row.
                columns[col] = list(np.asarray(values))
            else:
                columns[col] = _decode_array(values)
    ids = np.asarray(group[_TABLE_INDEX_COL][:])
    return pd.DataFrame(columns, index=pd.Index(ids, name=_TABLE_INDEX_COL))


class _Dataset:
    """Wraps an h5py/zarr dataset so that slicing reads and decodes byte-strings.

    Keeps reads lazy (only on ``[:]``) while making string ``data`` arrays come back as
    ``str`` to match pynwb (e.g. reward-type annotations).
    """

    def __init__(self, node):
        """Store the underlying h5py/zarr dataset."""
        self._node = node

    def __getitem__(self, key):
        """Read the requested slice and decode any byte-strings."""
        return _decode_array(self._node[key])

    def __len__(self):
        """Return the length of the underlying dataset."""
        return len(self._node)


class _DirectTimeSeries:
    """Mimics an NWB ``TimeSeries`` with ``.data`` and ``.timestamps`` accessors."""

    def __init__(self, group):
        """Store the TimeSeries group."""
        self._group = group

    @property
    def data(self):
        """The ``data`` dataset, slice-readable via ``[:]``."""
        return _Dataset(self._group["data"])

    @property
    def timestamps(self):
        """The ``timestamps`` dataset, slice-readable via ``[:]``."""
        return _Dataset(self._group["timestamps"])


class _DirectTimeSeriesDict(dict):
    """A dict of TimeSeries supporting ``|`` with another mapping (acquisition | FIP)."""


class _DirectTrials:
    """Mimics ``nwb.trials`` for the access patterns used by nwb_utils.

    Supports ``.to_dataframe()``, ``[:]``, ``['col']`` (positional), and ``.col[:]``.
    """

    def __init__(self, root):
        """Store the store root; the trials frame is built lazily and cached."""
        self._root = root
        self._df = None

    def to_dataframe(self):
        """Return a pynwb-equivalent trials DataFrame (index 'id')."""
        if self._df is None:
            self._df = _read_table(self._root, "/intervals/trials")
        return self._df.copy()

    def __getitem__(self, key):
        """``[:]`` returns the DataFrame; a column name returns it positionally."""
        if isinstance(key, slice):  # nwb.trials[:]
            return self.to_dataframe()
        return self.to_dataframe()[key].reset_index(drop=True).to_numpy()

    def __getattr__(self, name):
        """Expose a trial column as an ndarray (e.g. ``nwb.trials.goCue_start_time``)."""
        if name.startswith("_"):
            raise AttributeError(name)
        df = self.to_dataframe()
        if name in df.columns:
            return df[name].reset_index(drop=True).to_numpy()
        raise AttributeError(name)


class _DirectAcquisition:
    """Mimics ``nwb.acquisition`` (mapping of name -> TimeSeries)."""

    def __init__(self, root):
        """Store the store root."""
        self._root = root

    def keys(self):
        """Return the acquisition entry names."""
        return list(self._root["acquisition"].keys()) if "acquisition" in self._root else []

    def __iter__(self):
        """Iterate over acquisition names."""
        return iter(self.keys())

    def __len__(self):
        """Return the number of acquisition entries."""
        return len(self.keys())

    def __contains__(self, key):
        """Return whether ``key`` is an acquisition entry."""
        return key in self.keys()

    def __getitem__(self, key):
        """Return the named acquisition TimeSeries."""
        return _DirectTimeSeries(self._root["acquisition"][key])

    def __or__(self, other):
        """Merge with another mapping (``nwb.acquisition | data_interfaces``)."""
        merged = _DirectTimeSeriesDict((k, self[k]) for k in self.keys())
        merged.update(other)
        return merged


class _DirectProcessingModule:
    """Mimics a processing module, exposing ``.data_interfaces``."""

    def __init__(self, root, name):
        """Store the named processing module group."""
        self._group = root["processing"][name]

    @property
    def data_interfaces(self):
        """Return a name -> TimeSeries mapping of the module's data interfaces."""
        return _DirectTimeSeriesDict(
            (k, _DirectTimeSeries(self._group[k])) for k in self._group.keys()
        )


class _DirectProcessing:
    """Mimics ``nwb.processing`` (mapping of module name -> module)."""

    def __init__(self, root):
        """Store the store root."""
        self._root = root

    def keys(self):
        """Return the processing module names."""
        return list(self._root["processing"].keys()) if "processing" in self._root else []

    def __iter__(self):
        """Iterate over processing module names."""
        return iter(self.keys())

    def __len__(self):
        """Return the number of processing modules."""
        return len(self.keys())

    def __contains__(self, key):
        """Return whether ``key`` is a processing module."""
        return key in self.keys()

    def __getitem__(self, key):
        """Return the named processing module."""
        return _DirectProcessingModule(self._root, key)


class _DirectScratchTable:
    """Wraps a scratch table so that ``table.column[i]`` works as in pynwb."""

    def __init__(self, df):
        """Store the backing DataFrame without triggering ``__getattr__``."""
        object.__setattr__(self, "_df", df)

    def __getattr__(self, name):
        """Expose a scratch column by attribute access."""
        df = object.__getattribute__(self, "_df")
        if name in df.columns:
            return df[name]
        raise AttributeError(name)

    def __getitem__(self, key):
        """Expose a scratch column by item access."""
        return object.__getattribute__(self, "_df")[key]


class _DirectScratch:
    """Mimics ``nwb.scratch``; truthy only when scratch entries exist."""

    def __init__(self, root):
        """Store the store root."""
        self._root = root

    def keys(self):
        """Return the scratch entry names."""
        return list(self._root["scratch"].keys()) if "scratch" in self._root else []

    def __bool__(self):
        """Return whether any scratch entries exist."""
        return len(self.keys()) > 0

    def __len__(self):
        """Return the number of scratch entries."""
        return len(self.keys())

    def __iter__(self):
        """Iterate over scratch entry names."""
        return iter(self.keys())

    def __contains__(self, key):
        """Return whether ``key`` is a scratch entry."""
        return key in self.keys()

    def __getitem__(self, key):
        """Return the named scratch table."""
        return _DirectScratchTable(_read_table(self._root, f"/scratch/{key}"))


class _DirectSubject:
    """Mimics ``nwb.subject`` for the fields nwb_utils reads."""

    def __init__(self, root):
        """Store the store root."""
        self._root = root

    @property
    def subject_id(self):
        """The subject id."""
        return _read_scalar(self._root, "/general/subject/subject_id")

    @property
    def weight(self):
        """The subject weight (as stored, typically a string)."""
        return _read_scalar(self._root, "/general/subject/weight")

    @property
    def description(self):
        """The subject description."""
        return _read_scalar(self._root, "/general/subject/description")


def _open_root(path):
    """Open the raw storage handle for ``path``: h5py.File (HDF5) or zarr Group (Zarr).

    Mirrors ``load_nwb_from_filename``'s format detection. Zarr is opened with consolidated
    metadata when available (one metadata request instead of one per array), which is what
    makes reads fast - especially over S3.
    """
    is_zarr = (
        os.path.isdir(path)
        or (path.startswith("s3://") and path.endswith(".nwb"))
        or (path.startswith("s3://") and path.endswith(".nwb.zarr"))
    )
    if is_zarr:
        import zarr

        store = zarr.storage.FSStore(path, mode="r")
        try:
            return zarr.open_consolidated(store, mode="r")
        except (KeyError, ValueError):
            # No consolidated metadata (older files) -> plain open still works.
            return zarr.open(store, mode="r")
    if os.path.isfile(path):
        import h5py

        return h5py.File(path, mode="r")
    raise FileNotFoundError(path)


class DirectNWBAdapter:
    """A fast, dependency-free stand-in for a pynwb ``NWBFile``.

    Implements only the subset of the pynwb API used by ``nwb_utils``, reading directly
    from the HDF5/Zarr store so the DataFrame-building code can run unchanged while
    skipping pynwb's object-graph construction (and any FIP it does not need).
    """

    def __init__(self, path):
        """Open the store at ``path`` (HDF5 or Zarr, local or S3)."""
        self._path = str(path)
        self._root = _open_root(self._path)

    # -- scalar metadata --------------------------------------------------
    @property
    def session_id(self):
        """The session id."""
        return _read_scalar(self._root, "/general/session_id")

    @property
    def session_description(self):
        """The session description."""
        return _read_scalar(self._root, "/session_description")

    @property
    def session_start_time(self):
        """The session start time (as a ``datetime``)."""
        return _read_scalar(self._root, "/session_start_time")

    @property
    def experiment_description(self):
        """The experiment description."""
        return _read_scalar(self._root, "/general/experiment_description")

    @property
    def protocol(self):
        """The protocol (task) name."""
        return _read_scalar(self._root, "/general/protocol")

    @property
    def experimenter(self):
        """The experimenter(s), normalised to a tuple to match pynwb."""
        # pynwb returns experimenter as a tuple; normalise so that downstream
        # ``nwb.experimenter[0]`` behaves identically.
        value = _read_scalar(self._root, "/general/experimenter")
        if value is None:
            return None
        if isinstance(value, str):
            return (value,)
        try:
            return tuple(value)
        except TypeError:
            return (value,)

    @property
    def subject(self):
        """The subject accessor."""
        return _DirectSubject(self._root)

    # -- tables / groups --------------------------------------------------
    @property
    def trials(self):
        """The trials table accessor."""
        return _DirectTrials(self._root)

    @property
    def acquisition(self):
        """The acquisition mapping."""
        return _DirectAcquisition(self._root)

    @property
    def processing(self):
        """The processing-modules mapping."""
        return _DirectProcessing(self._root)

    @property
    def scratch(self):
        """The scratch mapping."""
        return _DirectScratch(self._root)

    def __repr__(self):
        """Return a debug representation."""
        return f"DirectNWBAdapter({self._path!r})"


def load_direct_nwb(path):
    """Return a :class:`DirectNWBAdapter` for ``path`` (HDF5 or Zarr, local or S3)."""
    return DirectNWBAdapter(path)
