from __future__ import annotations

import sys
import os
import multiprocessing as mp
import psutil
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ._common import error, num_nodes_per_cell
from ._exceptions import ReadError, WriteError
from ._files import is_buffer
from ._mesh import CellBlock, Mesh

extension_to_filetypes = {}
reader_map = {}
_writer_map = {}


def register_format(
    format_name: str, extensions: list[str], reader, writer_map
) -> None:
    for ext in extensions:
        if ext not in extension_to_filetypes:
            extension_to_filetypes[ext] = []
        extension_to_filetypes[ext].append(format_name)

    if reader is not None:
        reader_map[format_name] = reader

    _writer_map.update(writer_map)


def deregister_format(format_name: str):
    for value in extension_to_filetypes.values():
        if format_name in value:
            value.remove(format_name)

    if format_name in reader_map:
        reader_map.pop(format_name)

    if format_name in _writer_map:
        _writer_map.pop(format_name)


def _filetypes_from_path(path: Path) -> list[str]:
    ext = ""
    out = []
    for suffix in reversed(path.suffixes):
        ext = (suffix + ext).lower()
        try:
            out += reversed(extension_to_filetypes[ext])
        except KeyError:
            pass

    if not out:
        raise ReadError(f"Could not deduce file format from path '{path}'.")
    return out


def get_available_cores():
    return (
        min(mp.cpu_count(), len(os.sched_getaffinity(0)))
        if hasattr(os, "sched_getaffinity")
        else mp.cpu_count()
    )


def get_available_memory():
    return psutil.virtual_memory().available


def estimate_optimal_processes(files):
    total_size = sum(os.stat(file).st_size for file in files)
    available_memory = get_available_memory()
    available_cores = get_available_cores()

    if total_size == 0:
        return available_cores

    memory_based_limit = max(1, available_memory // (total_size / len(files)))
    return min(available_cores, memory_based_limit)


def read(filename, file_format: str | None = None, **kwargs) -> Mesh:
    """Reads an unstructured mesh with added data.

    :param filenames: The files/PathLikes to read from.
    :type filenames: str

    :returns mesh{2,3}d: The mesh data.
    """
    if is_buffer(filename, "r"):
        return _read_buffer(filename, file_format)

    return _read_file(Path(filename), file_format, **kwargs)


def _read_buffer(filename, file_format: str | None, **kwargs):
    if file_format is None:
        raise ReadError("File format must be given if buffer is used")
    if file_format == "tetgen":
        raise ReadError(
            "tetgen format is spread across multiple files "
            "and so cannot be read from a buffer"
        )
    if file_format not in reader_map:
        raise ReadError(f"Unknown file format '{file_format}'")

    return reader_map[file_format](filename, **kwargs)


def _read_file(path: Path, file_format: str | None, **kwargs):
    if not path.exists():
        raise ReadError(f"File {path} not found.")

    if file_format:
        possible_file_formats = [file_format]
    else:
        # deduce possible file formats from extension
        possible_file_formats = _filetypes_from_path(path)

    for file_format in possible_file_formats:
        if file_format not in reader_map:
            raise ReadError(f"Unknown file format '{file_format}' of '{path}'.")

        try:
            return reader_map[file_format](str(path), **kwargs)
        except ReadError as e:
            print(e)

    if len(possible_file_formats) == 1:
        msg = f"Couldn't read file {path} as {possible_file_formats[0]}"
    else:
        lst = ", ".join(possible_file_formats)
        msg = f"Couldn't read file {path} as either of {lst}"

    error(msg)
    sys.exit(1)


def write_points_cells(
    filename,
    points: ArrayLike,
    cells: dict[str, ArrayLike] | list[tuple[str, ArrayLike] | CellBlock],
    point_data: dict[str, ArrayLike] | None = None,
    cell_data: dict[str, list[ArrayLike]] | None = None,
    user_data: dict[str, list[ArrayLike]] | None = None,
    field_data=None,
    point_sets: dict[str, ArrayLike] | None = None,
    cell_sets: dict[str, list[ArrayLike]] | None = None,
    file_format: str | None = None,
    **kwargs,
):
    points = np.asarray(points)
    mesh = Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        user_data=user_data,
        field_data=field_data,
        point_sets=point_sets,
        cell_sets=cell_sets,
    )
    mesh.write(filename, file_format=file_format, **kwargs)


def write(filename, mesh: Mesh, file_format: str | None = None, **kwargs):
    """Writes mesh together with data to a file.

    :params filename: File to write to.
    :type filename: str

    :params point_data: Named additional point data to write to the file.
    :type point_data: dict
    """
    if is_buffer(filename, "r"):
        if file_format is None:
            raise WriteError("File format must be supplied if `filename` is a buffer")
        if file_format == "tetgen":
            raise WriteError(
                "tetgen format is spread across multiple files, and so cannot be written to a buffer"
            )
    else:
        path = Path(filename)
        if not file_format:
            # deduce possible file formats from extension
            file_formats = _filetypes_from_path(path)
            # just take the first one
            file_format = file_formats[0]

    try:
        writer = _writer_map[file_format]
    except KeyError:
        formats = sorted(list(_writer_map.keys()))
        raise WriteError(f"Unknown format '{file_format}'. Pick one of {formats}")

    # check cells for sanity
    for cell_block in mesh.cells:
        key = cell_block.type
        value = cell_block.data
        if key in num_nodes_per_cell:
            if value.shape[1] != num_nodes_per_cell[key]:
                raise WriteError(
                    f"Unexpected cells array shape {value.shape} for {key} cells. "
                    + f"Expected shape [:, {num_nodes_per_cell[key]}]."
                )
        else:
            # we allow custom keys <https://github.com/nschloe/meshio/issues/501> and
            # cannot check those
            pass

    # Write
    return writer(filename, mesh, **kwargs)
