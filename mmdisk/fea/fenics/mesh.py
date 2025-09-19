import os

import numpy as np
import pyvista as pv
import xarray as xr
from dolfin import (
    Function,
    FunctionSpace,
    Mesh,
    MeshEditor,
    MeshFunction,
    SubDomain,
    XDMFFile,
)
from mpi4py import MPI
from mmdisk.utils import dimensional_geometry

_properties_of_interest = [
    "Young's Modulus",
    "Poisson Ratio",
    "Yield Strength",
    "CTE",
    "Thermal Conductivity",
    "Density",
    "Specific Heat Capacity",
]


def load_universal_mesh() -> Mesh:
    base_input = os.path.join(
        os.path.dirname(__file__), "meshes", "Disk-Universal.xdmf"
    )
    return load_xdmf_mesh(base_input)


def load_xdmf_mesh(path: str) -> Mesh:
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "meshes", path)
    mesh = Mesh(MPI.COMM_SELF)
    with XDMFFile(path) as file:
        file.read(mesh)
    return mesh


def scale_1D_mesh(mesh: Mesh, r_o: float, r_i: float, thickness: float) -> Mesh:
    r_L = r_o - r_i
    coords = mesh.coordinates()
    r_max = coords[:, 0].max()
    t_max = coords[:, 1].max()

    coords[:, 0] = coords[:, 0] / r_max * r_L + r_i
    coords[:, 1] = coords[:, 1] / t_max * thickness

    return mesh


def mesh_and_properties(disk: xr.Dataset) -> tuple[Mesh, tuple[Function, ...]]:
    """
    Create a mesh and material properties from the disk dataset.
    If 'upper_density' is present, it uses the mesh2d format.
    Otherwise, it loads a universal mesh or a specified mesh.
    """

    if "upper_density" in disk.data_vars:
        mesh, props = mesh2d_and_properties(disk, return_pyvista=False)
    else:
        if "mesh" in disk.attrs and disk.attrs["mesh"] != "universal":
            mesh = load_xdmf_mesh(disk.attrs["mesh"])
        else:
            mesh = load_universal_mesh()

        dim_geom = dimensional_geometry(disk.geometry)
        r_o, r_i, _, thickness = [v.item() for v in dim_geom]
        mesh = scale_1D_mesh(mesh, r_o, r_i, thickness)
        props = get_properties(mesh, disk.properties.to_numpy())

    return mesh, props


def load_grid(upper_density: xr.DataArray) -> pv.UnstructuredGrid:
    """
    Load the grid corresponding to the upper_density.
    """
    ny, nx = upper_density.shape
    pixel_size = upper_density.attrs["pixel_size"]

    grid = pv.read(
        os.path.join(
            os.path.dirname(__file__),
            "meshes",
            f"mesh2d_{ny}_{nx}_{str(pixel_size).replace('.', '')}.vtu",
        )
    )

    return grid


def mesh_max_size(disk: xr.Dataset) -> tuple[float, float]:
    """
    Find the maximum dimension (nodes and cells) of the mesh based on the disk.
    Returns the maximum number of nodes and cells.
    """
    if "upper_density" in disk.data_vars:
        grid = load_grid(disk.upper_density).triangulate()
        return grid.num_points, grid.n_cells
    else:
        if "mesh" in disk.attrs and disk.attrs["mesh"] != "universal":
            mesh = load_xdmf_mesh(disk.attrs["mesh"])
        else:
            mesh = load_universal_mesh()

        return mesh.num_vertices(), mesh.num_cells()


def mesh2d_and_properties(
    disk: xr.Dataset, return_pyvista=False
) -> tuple[Mesh, tuple[Function, ...]]:
    # Find base grid
    grid = load_grid(disk.upper_density)
    grid.cell_data["upper_density"] = disk.upper_density.values.flatten()

    # Assign material properties
    for k in _properties_of_interest:
        grid.cell_data[k] = disk.properties.sel(property=k).values.flatten()

    # Prepare reduced mesh
    red_mesh = grid.triangulate().threshold(0.5, scalars="upper_density")
    vertices = np.array(red_mesh.points)[:, :2]  # Only keep x and y coordinates
    cells = np.array(red_mesh.cells_dict[pv.CellType.TRIANGLE])

    # Create FEniCS mesh
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(vertices.shape[0])
    editor.init_cells(cells.shape[0])

    # Add vertices to the mesh
    for i, node in enumerate(vertices):
        editor.add_vertex(i, node)

    # Add cells to the mesh
    for i, cell in enumerate(cells):
        editor.add_cell(i, cell)
    # Close the mesh editor
    editor.close()

    # Prepare material properties
    V0 = FunctionSpace(mesh, "DG", 0)
    kappa = Function(V0)
    rho = Function(V0)
    cp = Function(V0)
    E = Function(V0)
    sig0 = Function(V0)
    nu = Function(V0)
    alpha_V = Function(V0)

    kappa.vector()[:] = np.ascontiguousarray(
        red_mesh.cell_data["Thermal Conductivity"]
    )  # Thermal conductivity
    rho.vector()[:] = np.ascontiguousarray(red_mesh.cell_data["Density"])  # Density
    cp.vector()[:] = np.ascontiguousarray(
        red_mesh.cell_data["Specific Heat Capacity"]
    )  # Specific heat
    E.vector()[:] = np.ascontiguousarray(
        red_mesh.cell_data["Young's Modulus"]
    )  # Young's modulus
    sig0.vector()[:] = np.ascontiguousarray(
        red_mesh.cell_data["Yield Strength"]
    )  # Yield strength
    nu.vector()[:] = np.ascontiguousarray(
        red_mesh.cell_data["Poisson Ratio"]
    )  # poisson ratio
    alpha_V.vector()[:] = np.ascontiguousarray(red_mesh.cell_data["CTE"])  # CTE

    if return_pyvista:
        return mesh, (rho, cp, kappa, E, sig0, nu, alpha_V), red_mesh

    return mesh, (rho, cp, kappa, E, sig0, nu, alpha_V)


def mesh_and_composition(disk: xr.Dataset):
    # Find base grid
    ny, nx = disk.upper_density.shape
    pixel_size = disk.upper_density.attrs["pixel_size"]

    grid = pv.read(
        os.path.join(
            os.path.dirname(__file__),
            "meshes",
            f"mesh2d_{ny}_{nx}_{str(pixel_size).replace('.', '')}.vtu",
        )
    )
    grid.cell_data["upper_density"] = disk.upper_density.values.flatten()

    # Assign material properties
    for k in disk.data_vars:
        if k.startswith("fgm_comp"):
            # Only include FGM composition data
            grid.cell_data[k] = disk[k].values.flatten()

    # Prepare reduced mesh
    red_mesh = grid.triangulate().threshold(0.5, scalars="upper_density")

    return red_mesh


class MaterialDomain1D(SubDomain):
    def __init__(self, location, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.location = location

    def inside(self, x, on_boundary):
        return x[0] >= self.location - 1e-6


def get_properties(mesh, mat_properties):
    n_secs = mat_properties.shape[0]
    r1 = mesh.coordinates()[:, 0].min()
    r2 = mesh.coordinates()[:, 0].max()
    l = (r2 - r1) / n_secs
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
    for i in range(n_secs):
        MaterialDomain1D(r1 + i * l).mark(subdomains, i)

    kappa = Function(FunctionSpace(mesh, "DG", 0))
    rho = Function(FunctionSpace(mesh, "DG", 0))
    cp = Function(FunctionSpace(mesh, "DG", 0))
    E = Function(FunctionSpace(mesh, "DG", 0))
    sig0 = Function(FunctionSpace(mesh, "DG", 0))
    nu = Function(FunctionSpace(mesh, "DG", 0))
    alpha_V = Function(FunctionSpace(mesh, "DG", 0))

    temp = np.asarray(subdomains.array(), dtype=np.int32)

    kappa.vector()[:] = np.array(mat_properties[temp, 4])  # Thermal conductivity
    rho.vector()[:] = np.array(mat_properties[temp, 5])  # Density
    cp.vector()[:] = np.array(mat_properties[temp, 6])  # Specific heat
    E.vector()[:] = np.array(mat_properties[temp, 0])  # Young's modulus
    sig0.vector()[:] = np.array(mat_properties[temp, 2])  # Yield strength
    nu.vector()[:] = np.array(mat_properties[temp, 1])  # poisson ratio
    alpha_V.vector()[:] = np.array(mat_properties[temp, 3])  # CTE

    return rho, cp, kappa, E, sig0, nu, alpha_V
