import os
import pathlib
import multiprocessing as mp
from argparse import ArgumentParser
from .. import ansys, flac3d, gmsh, mdpa, ply, stl, vtk, vtu, xdmf
from .._helpers import _filetypes_from_path, read, reader_map, estimate_optimal_processes


def add_args(parser: ArgumentParser):

    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )
    parser.add_argument(
        "--num-processes",
        "-np",
        type=int,
        help="number of processes",
        default=None,
    )

    parser.add_argument("infile", type=str, nargs="*", help="mesh file to convert")


def parallel_func(inputs) -> None:

    file, input_format = inputs

    size = os.stat(file).st_size
    print(f"File size before: {size / 1024 ** 2:.2f} MB")

    if input_format == "vtu":
        if vtu.check_data_format(file, "binary"):
            print(f"{file} is already binary")
            return

    mesh = read(file, file_format=input_format)

    # # Some converters (like VTK) require `points` to be contiguous.
    # mesh.points = np.ascontiguousarray(mesh.points)

    # write it out
    if input_format == "gmsh":
        gmsh.write("b" + file, mesh, binary=True)
    elif input_format == "ansys":
        ansys.write("b" + file, mesh, binary=True)
    elif input_format == "flac3d":
        flac3d.write("b" + file, mesh, binary=True)
    elif input_format == "mdpa":
        mdpa.write("b" + file, mesh, binary=True)
    elif input_format == "ply":
        ply.write("b" + file, mesh, binary=True)
    elif input_format == "stl":
        stl.write("b" + file, mesh, binary=True)
    elif input_format == "vtk":
        vtk.write("b" + file, mesh, binary=True)
    elif input_format == "vtu":
        vtu.write("b" + file, mesh, binary=True)
    elif input_format == "xdmf":
        xdmf.write("b" + file, mesh, data_format="HDF")
    else:
        print(f"Don't know how to convert {file} to binary format.")
        exit(1)

    os.remove(file)
    os.rename("b" + file, file)
    size = os.stat(file).st_size
    print(f"File size after: {size / 1024 ** 2:.2f} MB")


def binary(args: ArgumentParser):
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
            input_formats = [_filetypes_from_path(pathlib.Path(file))[0] for file in args.infile]

        flexible_args = list(zip(args.infile, input_formats))

        if args.num_processes:
            num_processes = args.num_processes
            print(f"Using {num_processes} processes...")
        else:
            num_processes = estimate_optimal_processes(args.infile)
            num_processes //= 2
            print(f"Using {num_processes} processes...")

        with mp.Pool(processes=num_processes) as pool:
            pool.map(parallel_func, flexible_args)
