import os
import pathlib
import multiprocessing as mp
from .. import ansys, flac3d, gmsh, mdpa, ply, stl, vtk, vtu, xdmf
from .._helpers import _filetypes_from_path, read, reader_map


def add_args(parser):

    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )

    parser.add_argument("infile", type=str, nargs="*", help="mesh file to convert")


def parallel_func(inputs) -> None:
    
    file, input_format = inputs

    size = os.stat(file).st_size
    print(f"File size before: {size / 1024 ** 2:.2f} MB")
    mesh = read(file, file_format=input_format)

    # # Some converters (like VTK) require `points` to be contiguous.
    # mesh.points = np.ascontiguousarray(mesh.points)

    # write it out
    if input_format == "gmsh":
        gmsh.write(file, mesh, binary=True)
    elif input_format == "ansys":
        ansys.write(file, mesh, binary=True)
    elif input_format == "flac3d":
        flac3d.write(file, mesh, binary=True)
    elif input_format == "mdpa":
        mdpa.write(file, mesh, binary=True)
    elif input_format == "ply":
        ply.write(file, mesh, binary=True)
    elif input_format == "stl":
        stl.write(file, mesh, binary=True)
    elif input_format == "vtk":
        vtk.write(file, mesh, binary=True)
    elif input_format == "vtu":
        vtu.write(file, mesh, binary=True)
    elif input_format == "xdmf":
        xdmf.write(file, mesh, data_format="HDF")
    else:
        print(f"Don't know how to convert {file} to binary format.")
        exit(1)

    size = os.stat(file).st_size
    print(f"File size after: {size / 1024 ** 2:.2f} MB")


def binary(args):
    if not isinstance(args.infile, list):

        if args.input_format:
            input_format = args.input_format
        else:
            input_format = _filetypes_from_path(pathlib.Path(args.infile))[0]

        parallel_func((args.infile, input_format))

    else:

        if args.input_format:
            input_format = [args.input_format] * len(args.infile)
        else:
            input_formats = [
                _filetypes_from_path(pathlib.Path(file))[0] for file in args.infile
            ]

        flexible_args = list(zip(args.infile, input_formats))

        with mp.Pool(processes=4) as pool:
            pool.map(parallel_func, flexible_args)
