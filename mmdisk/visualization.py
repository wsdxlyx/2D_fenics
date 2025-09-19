import matplotlib as mpl
import meshio
import numpy as np
import pyvista as pv


def compute_colors(densities, base_colors):
    if densities.ndim == 1:
        densities = densities[None, :]

    N = densities.shape[0]

    if N >= len(base_colors):
        raise ValueError("Too many elements to color")

    colors = base_colors[N][:, None] + (
        (base_colors[:N] - base_colors[N])[..., None] * densities[:, None, :]
    ).sum(axis=0)

    return colors.T


def mixed_material_disk_mesh(mesh, densities, with_density=False, clip=True):
    mesh = meshio.read("Disk.inp")

    base_colors = np.array(mpl.colormaps["Set3"].colors)

    if not with_density:
        c_colors = compute_colors(densities, base_colors)
    labels = densities if with_density else c_colors

    cell_colors = np.zeros((len(mesh.cells[0].data), *labels[0].shape))
    cell_name = np.zeros((len(mesh.cells[0].data),))

    for i, c in enumerate(mesh.cell_sets.values()):
        cell_colors[c[0]] = labels[i]
        cell_name[c[0]] = i

    pvmesh = pv.utilities.from_meshio(mesh)
    pvmesh.cell_data["distribution"] = cell_colors.tolist()
    pvmesh.cell_data["name"] = cell_name.tolist()

    if clip:
        m2plot = pvmesh.clip_surface(pv.Plane())
    else:
        m2plot = pvmesh
    return m2plot


def extract_domain_edges(mesh, tube=False):
    edges = []
    u_idx = np.unique(mesh.cell_data["name"])
    for u in u_idx:
        edge = mesh.extract_cells(
            ind=(mesh.cell_data["name"] == u).nonzero()[0]
        ).extract_feature_edges()
        if tube:
            edge = edge.tube()
        edges.append(edge)
    return edges
