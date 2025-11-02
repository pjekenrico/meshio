import numpy as np
from _io import TextIOWrapper
from .._exceptions import CorruptionError, ReadError
from .._mesh import CellBlock, Mesh
from .._helpers import register_format


def skip_empty_lines(f: TextIOWrapper):
    while True:
        pos = f.tell()
        line = f.readline().strip()
        if line:
            f.seek(pos)
            break
    return f


def write_2d_array(f: TextIOWrapper, array2d: np.ndarray, format: str):
    fmt = " ".join([format] * array2d.shape[1])
    fmt = "\n".join([fmt] * array2d.shape[0])
    data = fmt % tuple(array2d.ravel())
    f.write(data)


class MTCReader:
    """Helper class for reading MTC files."""

    def __init__(self, filename):
        self.points = []
        self.cells = {}
        self.point_data = {}
        self.cell_data = {}

        with open(filename) as f:
            # Parse header
            num_points, num_components, num_cells, _ = map(int, f.readline().split())
            cells = np.zeros((num_cells, num_components + 1), dtype=int)
            f = skip_empty_lines(f)

            # Read points
            points = [list(map(float, f.readline().split())) for _ in range(num_points)]
            points = np.array(points).reshape(num_points, num_components)
            f = skip_empty_lines(f)

            # Read cells
            first_zero = -1
            for i in range(num_cells):
                line = list(map(int, f.readline().split()))
                cells[i] = line
                if first_zero == -1 and line[-1] == 0:
                    first_zero = i

        # Split into cells and edges
        edges = cells[first_zero:, :-1] - 1
        cells = cells[:first_zero] - 1

        # Identify unique edges
        indices_edges = np.unique(np.sort(edges.flatten()))
        mask = np.ones(num_points, dtype=bool)
        mask[indices_edges] = False
        indices_cells = np.arange(num_points)[mask]

        # Assign tags
        dim_tags = np.zeros((num_points, 2), dtype=int)
        if num_components == 2:
            cellname, edgename = "triangle", "line"
            dim_tags[indices_cells] = [2, 0]
            dim_tags[indices_edges] = [1, 1]
        elif num_components == 3:
            cellname, edgename = "tetra", "triangle"
            dim_tags[indices_cells] = [3, 0]
            dim_tags[indices_edges] = [2, 1]
        else:
            raise CorruptionError(
                f"Unsupported number of components in MTC file: {num_components}"
            )

        geom_data0 = np.zeros((len(cells)))
        geom_data1 = np.ones((len(edges)))
        # you can have edges without cells
        geom_phys = [geom_data0, geom_data1] if len(cells) > 0 else [geom_data1]

        # Merge points and cells
        if len(points) == 0:
            raise ReadError("No points found in file.")
        self.points = points
        if len(cells) > 0:
            self.cells[cellname] = cells
        if len(edges) > 0:
            self.cells[edgename] = edges
        self.point_data = {"gmsh:dim_tags": dim_tags}
        self.cell_data = {
            "gmsh:geometrical": geom_phys,
            "gmsh:physical": geom_phys,
        }


def read(filename):
    reader = MTCReader(filename)
    return Mesh(
        reader.points,
        reader.cells,
        point_data=reader.point_data,
        cell_data=reader.cell_data,
    )


def write(filename: str, mesh: Mesh, dimension=None, precision: int = 17):

    if mesh.points.shape[1] == 2:
        points = np.column_stack([mesh.points, np.zeros_like(mesh.points[:, 0])])
    else:
        points = mesh.points

    prec = str(int(precision))

    tri_list = []
    tet_list = []
    line = np.empty((0,), dtype=int)
    tet = False
    tri = False

    for cell_block in mesh.cells:
        cell_type = cell_block.type
        data = np.asarray(cell_block.data, dtype=int).ravel()
        if cell_type == "triangle":
            tri = True
            tri_list.append(data)
        elif cell_type == "tetra":
            tet = True
            tet_list.append(data)

    if tet_list:
        tetra = np.concatenate(tet_list).astype(int, copy=False)
    else:
        tetra = np.empty((0,), dtype=int)
    if tri_list:
        triangle = np.concatenate(tri_list).astype(int, copy=False)
    else:
        triangle = np.empty((0,), dtype=int)

    if dimension:
        if float(dimension) < 3:
            tet = False

    if tet:
        dim = 3
        triangle = np.array([], dtype=int)
        tetra = tetra.reshape((-1, 4))
    elif tri:
        tetra = np.array([], dtype=int)
        triangle = triangle.reshape((-1, 3))
        if np.all(points[:, 0] == points[0, 0]):
            dim = 2
            points = points[:, 1:]
        elif np.all(points[:, 1] == points[0, 1]):
            dim = 2
            points = points[:, [0, 2]]
        elif np.all(points[:, 2] == points[0, 2]):
            dim = 2
            points = points[:, :2]
        else:
            dim = 2.5
    else:
        raise ValueError("No tetra, and no triangle, cannot export to mtc")

    # Check tetra orientation
    if dim == 3:
        t = tetra[0]
        normal = np.cross(
            points[t[1]] - points[t[0]],
            points[t[2]] - points[t[0]],
        )
        dot = np.dot(normal, points[t[3]] - points[t[0]])
        if dot < 0:
            print("Warning: Orientation of first tet is wrong, flipping them all.")
            tetra = tetra[:, [0, 2, 1, 3]]

    # Apparently Cimlib prefers normals looking down in 2D
    # If normals are still wrong after that, there may be foldovers in your mesh
    if dim == 2:
        # Actually only checking the first normal
        t = triangle[0]
        normal = np.cross(
            points[t[1]] - points[t[0]],
            points[t[2]] - points[t[0]],
        )
        if normal > 0:
            print("Warning: Orientation of first triangle is wrong, flipping them all.")
            triangle = triangle[:, [0, 2, 1]]

    # Regenerating edges to be sure to not have unused edges
    if dim == 3:
        tris1 = tetra[:, [0, 2, 1]]  # Order is very important !
        tris2 = tetra[:, [0, 1, 3]]
        tris3 = tetra[:, [0, 3, 2]]
        tris4 = tetra[:, [1, 2, 3]]

        tris = np.concatenate((tris1, tris2, tris3, tris4), axis=0)
        canon = np.sort(tris, axis=1)

        # Pack to 64-bit keys when possible for fast 1D unique
        max_idx = int(canon.max(initial=0))
        shift = int(np.ceil(np.log2(max_idx + 1))) if max_idx > 0 else 1
        if shift * 3 <= 64:
            a64 = canon[:, 0].astype(np.uint64, copy=False)
            b64 = canon[:, 1].astype(np.uint64, copy=False)
            c64 = canon[:, 2].astype(np.uint64, copy=False)
            keys = a64 | (b64 << np.uint64(shift)) | (c64 << np.uint64(2 * shift))
            _, uniq_idx, uniq_cnt = np.unique(
                keys, return_index=True, return_counts=True
            )
        else:
            canon = np.ascontiguousarray(canon)
            view = canon.view(
                [("a", canon.dtype), ("b", canon.dtype), ("c", canon.dtype)]
            ).ravel()
            _, uniq_idx, uniq_cnt = np.unique(
                view, return_index=True, return_counts=True
            )

        triangle = tris[uniq_idx][uniq_cnt == 1]

    if dim == 2:
        lin1 = triangle[:, [0, 1]]  # Once again, order is very important !
        lin2 = triangle[:, [2, 0]]
        lin3 = triangle[:, [1, 2]]

        lin = np.concatenate((lin1, lin2, lin3), axis=0)
        canon = np.sort(lin, axis=1)

        # Pack to 64-bit keys when possible for fast 1D unique
        max_idx = int(canon.max(initial=0))
        shift = int(np.ceil(np.log2(max_idx + 1))) if max_idx > 0 else 1
        if shift * 2 <= 64:
            a64 = canon[:, 0].astype(np.uint64, copy=False)
            b64 = canon[:, 1].astype(np.uint64, copy=False)
            keys = a64 | (b64 << np.uint64(shift))
            _, uniq_idx, uniq_cnt = np.unique(
                keys, return_index=True, return_counts=True
            )
        else:
            canon = np.ascontiguousarray(canon)
            view = canon.view([("a", canon.dtype), ("b", canon.dtype)]).ravel()
            _, uniq_idx, uniq_cnt = np.unique(
                view, return_index=True, return_counts=True
            )

        line = lin[uniq_idx][uniq_cnt == 1]

    # Detecting used nodes
    used_nodes = np.unique(np.concatenate((tetra.ravel(), triangle.ravel())))  # sorted
    bools_keep = np.zeros(len(points), dtype=bool)
    bools_keep[used_nodes] = True

    # Deleting unused nodes and reindexing
    points = points[bools_keep]
    new_indices = np.cumsum(bools_keep) - 1

    if dim == 3 or dim == 2.5:
        tetra = new_indices[tetra]
        triangle = new_indices[triangle]

    if dim == 2:
        triangle = new_indices[triangle]
        line = new_indices[line]

    ############

    print("Nb points : " + str(len(points)))

    nb_elems = len(triangle) + len(tetra)
    if dim == 2:
        nb_elems += len(line)
        print("Nb elements 1d : " + str(len(line)))

    print("Nb elements 2d : " + str(len(triangle)))
    print("Nb elements 3d : " + str(len(tetra)))
    print("Dimension : " + str(dim) + "\n")

    ############

    print("Writing .t file...")

    # Correction for mtc numbering
    tetra += 1
    triangle += 1
    line += 1

    with open(filename, "w") as fo:
        # Header
        lig = (
            str(len(points))
            + " "
            + str(dim)
            + " "
            + str(nb_elems)
            + " "
            + str(dim + 1)
            + "\n"
        )
        if dim == 2.5:
            lig = str(len(points)) + " 3 " + str(nb_elems) + " 4\n"
        fo.write(lig)

        # Points
        write_2d_array(fo, points, f"%.{prec}g")
        fo.write("\n")

        # Cells: tetrahedra first
        if tetra.size:
            write_2d_array(fo, tetra, "%d")
            fo.write("\n")

        # Triangles: with a trailing 0 in 3D/2.5, plain in 2D
        if triangle.size:
            if dim == 3 or dim == 2.5:
                tri_out = np.column_stack(
                    (triangle, np.zeros((triangle.shape[0], 1), dtype=int))
                )
                write_2d_array(fo, tri_out, "%d")
            else:
                write_2d_array(fo, triangle, "%d")
                fo.write("\n")

        # Lines in 2D: write with trailing 0
        if dim == 2 and line.size:
            line_out = np.column_stack((line, np.zeros((line.shape[0], 1), dtype=int)))
            write_2d_array(fo, line_out, "%d")

    print("Done.")

    return


register_format("mtc", [".t"], read, {"mtc": write})
