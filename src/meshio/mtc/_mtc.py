import numpy as np
from .._exceptions import CorruptionError, ReadError
from .._mesh import CellBlock, Mesh
from .._helpers import register_format


class MTCReader:
    """Helper class for reading MTC files."""

    def __init__(self, filename):
        self.points = {}
        self.cells = {}
        self.point_data = {}
        self.cell_data = {}

        points = []

        with open(filename) as f:
            num_points, num_components, num_cells, _ = f.readline().split()
            num_points = int(num_points)
            num_components = int(num_components)
            num_cells = int(num_cells)

            cells = np.zeros((num_cells, num_components + 1), dtype=int)
            pts = []

            for i in range(num_points):
                t = f.readline()
                t = t.strip("\t\n").split()
                if num_components == 2:
                    pt = [float(t[0]), float(t[1])]
                elif num_components == 3:
                    pt = [float(t[0]), float(t[1]), float(t[2])]
                else:
                    raise ReadError()
                pts.append(pt)

            pts = np.array(pts)
            points.append(pts.reshape(num_points, num_components))
            first_zero = -1

            for i in range(num_cells):
                t = f.readline()
                t = t.strip("\t\n").split()
                if i == 0:
                    while t == []:
                        t = f.readline()
                        t = t.strip("\t\n").split()

                for k in range(num_components + 1):
                    cells[i][k] = int(t[k])

                if first_zero == -1 and cells[i][-1] == 0:
                    first_zero = i

        edges = cells[first_zero:] - 1
        edges = edges[:, :-1]
        cells = cells[:first_zero] - 1

        indices_edges = np.sort(edges.flatten(), axis=0)  # creates a copy
        indices_edges = np.unique(indices_edges, axis=0)[1:]
        mask = np.ones(num_points, dtype=bool)
        mask[indices_edges,] = False
        indices_cells = np.linspace(0, num_points, num_points, dtype=int, endpoint=False)
        indices_cells = indices_cells[mask]
        dim_tags = np.zeros((num_points, 2), dtype=int)

        if num_components == 2:
            cellname = "triangle"
            edgename = "line"
            dim_tags[indices_cells] = np.array([2, 0])
            dim_tags[indices_edges] = np.array([1, 1])
        elif num_components == 3:
            cellname = "tetra"
            edgename = "triangle"
            dim_tags[indices_cells] = np.array([3, 0])
            dim_tags[indices_edges] = np.array([2, 1])

        geom_data0 = np.zeros((len(cells)))
        geom_data1 = np.ones((len(edges)))

        # Now merge across pieces
        if not points:
            raise ReadError()
        self.points = np.concatenate(points)
        if len(cells) != 0:
            self.cells[cellname] = cells
        if len(edges) != 0:
            self.cells[edgename] = edges
        # self.point_data = {"gmsh:dim_tags":dim_tags}
        # self.cell_data = {"gmsh:geometrical":[geom_data0,geom_data1],
        #                   "gmsh:physical":[geom_data0,geom_data1]}


def read(filename):
    reader = MTCReader(filename)
    return Mesh(
        reader.points,
        reader.cells,
        # point_data=reader.point_data,
        # cell_data=reader.cell_data
    )


def write(filename, mesh, dimension=None, precision: int = 17):

    if mesh.points.shape[1] == 2:
        points = np.column_stack([mesh.points, np.zeros_like(mesh.points[:, 0])], dtype=float)
    else:
        points = mesh.points

    prec = str(int(precision))

    tetra = np.array([], dtype=int)
    triangle = np.array([], dtype=int)
    line = np.array([], dtype=int)
    tet = False
    tri = False

    for cell_block in mesh.cells:
        cell_type = cell_block.type
        data = cell_block.data
        if cell_type == "triangle":
            tri = True
            triangle = np.concatenate((triangle, data.flatten()), dtype=int)
        if cell_type == "tetra":
            tet = True
            tetra = np.concatenate((tetra, data.flatten()), dtype=int)

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
            points = points[:, -1:1]
        elif np.all(points[:, 2] == points[0, 2]):
            dim = 2
            points = points[:, :2]
        else:
            dim = 2.5
    else:
        raise ValueError("No tetra, and no triangle, cannot export to mtc")

    # Apparently Cimlib prefers normals looking down in 2D
    # If normals are still wrong after that, there may be foldovers in your mesh
    if dim == 2:
        # Actually only checking the first normal
        normal = np.cross(
            points[triangle[0][1]] - points[triangle[0][0]],
            points[triangle[0][2]] - points[triangle[0][0]],
        )
        if normal > 0:
            triangle = triangle[:, [0, 2, 1]]

    # Regenerating edges to be sure to not have unused edges
    if dim == 3:
        tris1 = tetra[:, [0, 2, 1]]  # Order is very important !
        tris2 = tetra[:, [0, 1, 3]]
        tris3 = tetra[:, [0, 3, 2]]
        tris4 = tetra[:, [1, 2, 3]]

        tris = np.concatenate((tris1, tris2, tris3, tris4), axis=0)
        tris_sorted = np.sort(tris, axis=1)  # creates a copy, may be source of memory error
        tris_sorted, uniq_idx, uniq_cnt = np.unique(
            tris_sorted, axis=0, return_index=True, return_counts=True
        )
        triangle = tris[uniq_idx][uniq_cnt == 1]

    if dim == 2:
        lin1 = triangle[:, [0, 1]]  # Once again, order is very important !
        lin2 = triangle[:, [2, 0]]
        lin3 = triangle[:, [1, 2]]

        lin = np.concatenate((lin1, lin2, lin3), axis=0)
        lin_sorted = np.sort(lin, axis=1)  # creates a copy, may be source of memory error
        lin_sorted, uniq_idx, uniq_cnt = np.unique(
            lin_sorted, axis=0, return_index=True, return_counts=True
        )
        line = lin[uniq_idx][uniq_cnt == 1]

    # Detecting used nodes
    used_elems = np.unique(np.concatenate((tetra.flatten(), triangle.flatten())))  # sorted
    bools_keep = np.zeros(len(points), dtype=bool)
    bools_keep[used_elems] = True

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
        lig = str(len(points)) + " " + str(dim) + " " + str(nb_elems) + " " + str(dim + 1) + "\n"
        if dim == 2.5:
            lig = str(len(points)) + " 3 " + str(nb_elems) + " 4\n"
        fo.write(lig)

        for node in points:
            fo.write(("{0:." + prec + "g} {1:." + prec + "g}").format(node[0], node[1]))
            if dim == 3 or dim == 2.5:
                fo.write((" {0:." + prec + "g}").format(node[2]))
            fo.write(" \n")

        for e in tetra:
            fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " " + str(e[3]) + " \n")

        for e in triangle:
            if dim == 3 or dim == 2.5:
                fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " 0 \n")
            else:
                fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " \n")

        if dim == 2:
            for e in line:
                fo.write(str(e[0]) + " " + str(e[1]) + " 0 \n")

    print("Done.")

    return


register_format("mtc", [".t"], read, {"mtc": write})
