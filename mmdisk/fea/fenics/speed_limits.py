import click
import numpy as np
import xarray as xr
from dolfin import (
    Constant,
    DirichletBC,
    FiniteElement,
    Function,
    FunctionSpace,
    Identity,
    LocalSolver,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    VectorElement,
    VectorFunctionSpace,
    as_tensor,
    as_vector,
    assemble_system,
    dev,
    dx,
    inner,
    project,
    solve,
    sqrt,
    sym,
    tr,
)

from mmdisk.cli_utils import dask_cluster_options, start_dask_cluster_and_client
from mmdisk.fea.fenics.mesh import mesh_and_properties
from mmdisk.fea.fenics.utils import (
    AbPETScContext,
    as_3D_tensor,
    get_boundary,
    ppos,
)


def speed_limits_for_disks(disks):
    if disks.sizes["disks"] == 0:
        return xr.DataArray(
            np.empty((0, 2)),
            dims=["disks", "limit_kind"],
            coords={"disks": disks.disks, "limit_kind": ["elastic", "collapse"]},
        )

    speeds = []
    for n in disks.disks:
        disk = disks.sel(disks=n)
        mat_properties = disk.properties.to_numpy()

        mesh, props = mesh_and_properties(disk)

        r_o = mesh.coordinates()[:, 0].max()
        omega_0 = (
            1.0 / r_o * np.sqrt(mat_properties[..., 2] / mat_properties[..., 5])
        ).min()

        speeds.append(
            speed_limits_fea(
                mesh, props, "fix-free", omega_inc=omega_0 * 0.5, EP_Nitermax=10
            )
        )

    return xr.DataArray(
        np.array(speeds),
        dims=["disks", "limit_kind"],
        coords={"disks": disks.disks, "limit_kind": ["elastic", "collapse"]},
    )


def speed_limits_fea(
    mesh,
    properties,
    Displacement_BC="free-free",
    step0_iter_max=1000,
    omega_inc=1000,
    omega_tol=1,
    tol=1e-6,
    EP_Nitermax=50,
    verbose=False,
):
    if not verbose:

        def print(*args, **kwargs):
            pass

    # material properties
    rho, _, _, E, sig0, nu, _ = properties
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2.0 / (1 + nu)
    Et = E / 1e5  # tangent modulus - any better method for perfect plasticity?
    H = E * Et / (E - Et)  # hardening modulus

    # define function space
    x = SpatialCoordinate(mesh)  # Coords
    U = VectorFunctionSpace(mesh, "CG", 1)
    We = VectorElement(
        "Quadrature", mesh.ufl_cell(), degree=1, dim=4, quad_scheme="default"
    )
    W = FunctionSpace(mesh, We)
    W0e = FiniteElement("Quadrature", mesh.ufl_cell(), degree=1, quad_scheme="default")
    W0 = FunctionSpace(mesh, W0e)
    P0 = FunctionSpace(mesh, "DG", 0)

    metadata = {"quadrature_degree": 1, "quadrature_scheme": "default"}
    dxm = dx(metadata=metadata)

    ## DEFINE BCs
    boundary_left, boundary_right, boundary_bot, _ = get_boundary(mesh)

    U_bc_B = DirichletBC(U.sub(1), 0.0, boundary_bot)
    U_bc_L = DirichletBC(U.sub(0), 0.0, boundary_left)
    if Displacement_BC == "fix-free":
        Mech_BC = [U_bc_B, U_bc_L]
    else:
        Mech_BC = [U_bc_B]

    def eps(v):
        return sym(
            as_tensor(
                [
                    [v[0].dx(0), 0, v[0].dx(1)],
                    [0, v[0] / x[0], 0],
                    [v[1].dx(0), 0, v[1].dx(1)],
                ]
            )
        )

    def local_project(v, V, u=None):
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv, v_) * dxm
        b_proj = inner(v, v_) * dxm
        solver = LocalSolver(a_proj, b_proj)
        solver.factorize()
        if u is None:
            u = Function(V)
        solver.solve_local_rhs(u)
        return u

    def collapse_check(sig, sig0):
        s_ = dev(as_3D_tensor(sig))
        sig_eq_ = sqrt(3 / 2.0 * inner(s_, s_))
        sig_eq = local_project(sig_eq_, W0)
        return project(sig_eq / sig0, P0).vector()

    sig = Function(W)
    sig_old = Function(W)
    n_elas = Function(W)
    n_elas_old = Function(W)
    beta = Function(W0)
    beta_old = Function(W0)
    p = Function(W0, name="Cumulative plastic strain")
    u = Function(U, name="Total displacement")
    du = Function(U, name="Iteration correction")
    Du = Function(U, name="Current increment")
    v_ = TrialFunction(U)
    u_ = TestFunction(U)

    def sigma(eps):
        return lmbda * tr(eps) * Identity(3) + 2 * mu * eps

    omega = Constant(0.0)

    F_int = rho * omega**2 * x[0] * u_[0]

    def proj_sig(deps, old_sig, old_p):
        sig_n = as_3D_tensor(old_sig)
        sig_elas = sig_n + sigma(deps)
        s = dev(sig_elas)
        sig_eq = sqrt(3 / 2.0 * inner(s, s))
        f_elas = sig_eq - sig0 - H * old_p
        dp = ppos(f_elas) / (3 * mu + H)
        n_elas = s / sig_eq * ppos(f_elas) / f_elas
        beta = 3 * mu * dp / sig_eq
        new_sig = sig_elas - beta * s
        return (
            as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 2]]),
            as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 2]]),
            beta,
            dp,
        )

    def sigma_tang(e):
        N_elas = as_3D_tensor(n_elas)
        return (
            sigma(e)
            - 3 * mu * (3 * mu / (3 * mu + H) - beta) * inner(N_elas, e) * N_elas
            - 2 * mu * beta * dev(e)
        )

    a_Newton = inner(eps(v_), sigma_tang(eps(u_))) * x[0] * dxm
    res = -inner(eps(u_), as_3D_tensor(sig)) * x[0] * dxm + F_int * x[0] * dxm

    step0_iter = 0
    omega_ctr = 0
    omega_e = 0

    with AbPETScContext() as (A, Res):
        while step0_iter < step0_iter_max:
            step0_iter += 1
            omega_ctr += omega_inc
            omega.assign(Constant(omega_ctr))
            print("Trying Omega = ", omega_ctr)

            assemble_system(a_Newton, res, Mech_BC, A_tensor=A, b_tensor=Res)
            Du.interpolate(Constant((0, 0)))
            nRes0 = Res.norm("l2")
            nRes = nRes0
            print("Residual = ", nRes0)

            niter = 0
            while nRes / nRes0 > tol:
                solve(A, du.vector(), Res)
                Du.assign(Du + du * 1)
                deps = eps(Du)
                sig_, n_elas_, beta_, dp_ = proj_sig(deps, sig_old, p)
                local_project(sig_, W, sig)
                local_project(n_elas_, W, n_elas)
                local_project(beta_, W0, beta)
                assemble_system(a_Newton, res, Mech_BC, A_tensor=A, b_tensor=Res)
                nRes = Res.norm("l2")
                print("Residual = ", nRes)
                niter += 1
                if niter >= EP_Nitermax:
                    print("Too many iterations, check the residual")
                    break

            elastic_factor = collapse_check(sig, sig0)

            if elastic_factor.min() >= 1.0:
                # Collapse
                sig.assign(sig_old)
                beta.assign(beta_old)
                n_elas.assign(n_elas_old)
                if omega_inc > omega_tol:
                    omega_ctr -= omega_inc
                    omega_inc *= 0.5
                else:
                    print("number of iters:", step0_iter)
                    print("max omega = ", omega_ctr)
                    print("omega elastic = ", omega_e)
                    return omega_e, omega_ctr

            else:
                if omega_e == 0 and nRes / nRes0 <= tol and elastic_factor.max() < 1.0:
                    omega_e = omega_ctr * np.sqrt(1.0 / elastic_factor.max())
                u.assign(u + Du)
                p.assign(p + local_project(dp_, W0))
                sig_old.assign(sig)
                beta_old.assign(beta)
                n_elas_old.assign(n_elas)

    return omega_e, -1


@click.command()
@click.option(
    "--case", "-c", type=click.Path(exists=True), help="Input Zarr DB", required=True
)
@click.option(
    "--output",
    "-o",
    type=click.Path(exists=False),
    help="Output database",
    required=False,
)
@dask_cluster_options
def main(case, output, queue, n_cores, n_jobs, time_limit, network, network_sched):
    client = start_dask_cluster_and_client(
        queue,
        n_cores,
        n_jobs,
        time_limit,
        f"{n_cores * 10}GB",
        f"{n_cores * 5}GB",
        network,
        network_sched,
        ["export OMP_NUM_THREADS=1"],
    )
    print(client)
    print(client.dashboard_link)

    datavars_of_interest = ["properties", "heat_time", "geometry", "upper_density"]

    ds = xr.open_zarr(case)
    chunk_def = {k: -1 for k in ds.dims if k != "disks"}
    chunk_def["disks"] = 10

    # Ensure the dataset has the required data variables
    to_drop = [dv for dv in ds.data_vars if dv not in datavars_of_interest]

    chunked = xr.open_zarr(case, chunks=chunk_def, drop_variables=to_drop)

    if output is None:
        output = case
        mode = "a"
    else:
        mode = None

    limits = chunked.map_blocks(
        speed_limits_for_disks,
    )
    zarr_store = limits.to_dataset(name="max_speeds").to_zarr(output, mode=mode)
    # Assign to variable to avoid premature cleanup
    zarr_store


if __name__ == "__main__":
    main()
