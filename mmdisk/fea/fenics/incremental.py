from typing import Callable, Union, Tuple

import numpy as np
from dolfin import (
    Constant,
    DirichletBC,
    FiniteElement,
    Function,
    FunctionSpace,
    Identity,
    LocalSolver,
    LogLevel,
    Mesh,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    VectorElement,
    VectorFunctionSpace,
    as_tensor,
    as_vector,
    assemble_system,
    dev,
    dot,
    dx,
    grad,
    inner,
    interpolate,
    lhs,
    project,
    rhs,
    set_log_level,
    solve,
    sqrt,
    sym,
    tr,
)

from mmdisk.fea.fenics.utils import (
    AbPETScContext,
    as_3D_tensor,
    get_boundary,
    ppos,
)


class ConvergenceError(Exception):
    def __init__(self, message, peeq, residuals):
        self.message = message
        self.peeq = peeq
        self.residuals = residuals
        super().__init__(self.message)


def incremental_cyclic_fea(
    mesh: Mesh,
    properties: Tuple,
    load: Callable,
    omega: float,
    t_list: np.ndarray,
    t_output: np.ndarray,
    period: float,
    Displacement_BC: str = "fix-free",
    plastic_inverval: int = 1,
    tol: float = 1e-5,
    Nitermax: int = 50,
    skip_cold: bool = False,
    verbose: bool = False,
    outputs: Union[None, list, dict] = None,
) -> Tuple[np.ndarray, dict, int]:
    """
    Perform incremental cyclic finite element analysis with thermal and mechanical coupling.
    Parameters
    ----------
    mesh : Mesh
        The mesh to be used for the analysis.
    properties : tuple
        Material properties in the order (rho, cp, kappa, E, sig0, nu, alpha_V).
    load : Callable
        Function to compute the load at a given time.
    omega : float
        Angular frequency of the load.
    t_list : np.ndarray
        Array of time points for the analysis.
    t_output : np.ndarray
        Array of time points for output.
    period : float
        Period of the cyclic loading.
    Displacement_BC : str, optional
        Type of displacement boundary condition, either "fix-free" or "free-free". Default is "fix-free".
    plastic_inverval : int, optional
        Interval for plasticity updates. Default is 1 (every time step).
    tol : float, optional
        Convergence tolerance. Default is 1e-5.
    Nitermax : int, optional
        Maximum number of iterations for convergence. Default is 50.
    skip_cold : bool, optional
        If True, skips the cold dwell for the thermal problem.
    verbose : bool, optional
        If True, enables verbose logging. Default is False.
    outputs : Union[None, list, dict], optional
        Outputs to be collected during the analysis. If None, defaults to collecting "PEEQ".
        If a list, it will be converted to a dictionary with empty arrays. If a dictionary,
        it will be used to store the outputs with preallocated arrays.
        Available outputs: "PEEQ" (plastic strain), "T" (temperature), "u" (displacement),
        "sig" (stress), "eps_vector" (strain vector [eps_rr, eps_theta, eps_zz, eps_rz]).
    Returns
    -------
    t_output : np.ndarray
        Array of time points for output.
    outputs : dict
        Dictionary containing the outputs collected during the analysis.
    flag : int
        Flag indicating the result of the analysis:
        - 0: Normal completion
        - 1: Early stopping due to elastic limits
        - 2: Early stopping due to shakedown limits
    """
    if not verbose:
        set_log_level(LogLevel.ERROR)
    else:
        set_log_level(LogLevel.DEBUG)

    # material properties
    rho, cp, kappa, E, sig0, nu, alpha_V = properties
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2.0 / (1 + nu)
    Et = E / 1e5  # tangent modulus - any better method for perfect plasticity?
    H = E * Et / (E - Et)  # hardening modulus

    # DEFINE THERMAL PROBLEM
    V = FunctionSpace(mesh, "P", 1)  # Temperature space
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)  # Coords
    T_initial = Constant(0.0)
    T_pre = interpolate(T_initial, V)  # Temp. at last time step
    T_old = interpolate(T_initial, V)  # ** Temp. at last mechanical step **
    T_crt = TrialFunction(V)
    dt = Constant(1.0)
    F_thermal = (rho * cp * (T_crt - T_pre) / dt) * v * x[0] * dx + kappa * dot(
        grad(T_crt), grad(v)
    ) * x[0] * dx
    a_thermal, L_thermal = lhs(F_thermal), rhs(F_thermal)

    # DEFINE MECH PROBLEM
    U = VectorFunctionSpace(mesh, "CG", 1)
    We = VectorElement(
        "Quadrature", mesh.ufl_cell(), degree=1, dim=4, quad_scheme="default"
    )
    W = FunctionSpace(mesh, We)
    W0e = FiniteElement("Quadrature", mesh.ufl_cell(), degree=1, quad_scheme="default")
    W0 = FunctionSpace(mesh, W0e)

    metadata = {"quadrature_degree": 1, "quadrature_scheme": "default"}
    dxm = dx(metadata=metadata)

    sig = Function(W)
    sig_old = Function(W)
    n_elas = Function(W)
    beta = Function(W0)
    p = Function(W0, name="Cumulative plastic strain")
    strain = Function(W, name="Strain vector")
    u = Function(U, name="Total displacement")
    du = Function(U, name="Iteration correction")
    Du = Function(U, name="Current increment")
    v_ = TrialFunction(U)
    u_ = TestFunction(U)

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

    def eps_to_vector(v):
        """Convert strain tensor to vector format [eps_rr, eps_theta, eps_zz, eps_rz]"""
        eps_tensor = eps(v)
        return as_vector(
            [eps_tensor[0, 0], eps_tensor[1, 1], eps_tensor[2, 2], eps_tensor[0, 2]]
        )

    def sigma(v, dT):
        return (lmbda * tr(eps(v)) - alpha_V * (3 * lmbda + 2 * mu) * dT) * Identity(
            3
        ) + 2.0 * mu * eps(v)

    def proj_sig(v, dT, old_sig, old_p):
        sig_n = as_3D_tensor(old_sig)
        sig_elas = sig_n + sigma(v, dT)
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

    def update_sig_thermal(dT, old_sig):
        sig_n = as_3D_tensor(old_sig)
        new_sig = sig_n - alpha_V * (3 * lmbda + 2 * mu) * dT * Identity(3)
        return as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 2]])

    def sigma_tang(v):
        N_elas = as_3D_tensor(n_elas)
        e = eps(v)
        return (
            lmbda * tr(e) * Identity(3)
            + 2.0 * mu * e
            - 3 * mu * (3 * mu / (3 * mu + H) - beta) * inner(N_elas, e) * N_elas
            - 2 * mu * beta * dev(e)
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
        else:
            solver.solve_local_rhs(u)
            return

    omega_ = Constant(omega)
    F_int = rho * omega_**2 * x[0] * u_[0]

    a_Newton = inner(eps(v_), sigma_tang(u_)) * x[0] * dxm
    res = -inner(eps(u_), as_3D_tensor(sig)) * x[0] * dxm + F_int * x[0] * dxm

    ## DEFINE BCs
    boundary_left, boundary_right, boundary_bot, _ = get_boundary(mesh)

    T_L = Constant(0.0)
    T_R = Constant(0.0)
    T_bc_L = DirichletBC(V, T_L, boundary_left)
    T_bc_R = DirichletBC(V, T_R, boundary_right)
    Thermal_BC = [T_bc_L, T_bc_R]

    U_bc_B = DirichletBC(U.sub(1), 0.0, boundary_bot)
    #     U_bc_L = DirichletBC(U, Constant((0.,0.)), boundary_left)
    U_bc_L = DirichletBC(U.sub(0), 0.0, boundary_left)
    if Displacement_BC == "fix-free":
        Mech_BC = [U_bc_B, U_bc_L]
    else:
        Mech_BC = [U_bc_B]

    P0 = FunctionSpace(mesh, "DG", 0)
    p_avg = Function(P0)
    T_crt = Function(V)
    dT = Function(V)

    output_matches = {
        "PEEQ": p_avg.vector,
        "T": T_crt.vector,
        "u": u.vector,
        "sig": lambda: project(sig, P0).vector(),
        "strain": strain.vector,
    }

    if outputs is None:
        outputs = {"PEEQ": np.zeros((len(t_output), len(p.vector())))}
    elif isinstance(outputs, (list, tuple)):
        if "PEEQ" not in outputs:
            outputs.append("PEEQ")
        outputs = {
            k: np.zeros((len(t_output), len(output_matches[k]()))) for k in outputs
        }
    else:
        if "PEEQ" not in outputs:
            outputs["PEEQ"] = np.zeros((len(t_output), len(output_matches["PEEQ"]())))
        for k in outputs.keys():
            outputs[k][:] = 0.0

    comp_cycle_marker = np.mod(t_list, period) == 0.0
    out_cycle_marker = np.mod(t_output, period) == 0.0

    # Allocate memory for the solver
    with AbPETScContext() as (A, Res), AbPETScContext() as (At, Lt):
        for n in range(len(t_list) - 1):
            dt.assign(t_list[n + 1] - t_list[n])
            T_R.assign(load(t_list[n + 1]))
            assemble_system(a_thermal, L_thermal, Thermal_BC, A_tensor=At, b_tensor=Lt)

            if comp_cycle_marker[n + 1] and skip_cold:
                T_crt.assign(interpolate(T_initial, V))
                T_pre.assign(T_crt)
            else:
                solve(At, T_crt.vector(), Lt)
                T_pre.assign(T_crt)

            if (n + 1) % plastic_inverval == 0:
                dT.assign(T_crt - T_old)

                sig_ = update_sig_thermal(dT, sig_old)
                local_project(sig_, W, sig)

                assemble_system(a_Newton, res, Mech_BC, A_tensor=A, b_tensor=Res)

                Du.interpolate(Constant((0, 0)))
                nRes0 = Res.norm("l2")
                nRes = nRes0
                niter = 0
                res_tracking = [nRes / nRes0]
                while nRes / nRes0 > tol and niter < Nitermax:
                    solve(A, du.vector(), Res)
                    Du.assign(Du + du * 1.0)
                    sig_, n_elas_, beta_, dp_ = proj_sig(Du, dT, sig_old, p)
                    local_project(sig_, W, sig)
                    local_project(n_elas_, W, n_elas)
                    local_project(beta_, W0, beta)
                    A, Res = assemble_system(
                        a_Newton, res, Mech_BC, A_tensor=A, b_tensor=Res
                    )
                    nRes = Res.norm("l2")
                    res_tracking.append(nRes / nRes0)
                    niter += 1
                    if nRes / nRes0 > tol and niter >= Nitermax:
                        print(res_tracking)
                        for k in outputs.keys():
                            outputs[k][t_output >= t_list[n + 1], :] = np.nan
                        raise ConvergenceError(
                            f"Too many iterations at time {t_list[n + 1]} (step {n + 1}) with residual {nRes / nRes0}",
                            outputs,
                            res_tracking,
                        )

                u.assign(u + Du)
                strain.assign(local_project(eps_to_vector(u), W))
                p.assign(p + local_project(dp_, W0))
                p_avg.assign(project(p, P0))
                T_old.assign(T_crt)
                sig_old.assign(sig)

                time_point_selector = np.isclose(t_list[n + 1], t_output, rtol=1e-7)

                if time_point_selector.any():
                    for k in outputs.keys():
                        out = output_matches[k]().get_local()
                        if len(outputs[k][time_point_selector].shape) > 2:
                            out = out.reshape(
                                (
                                    # time_point_selector.sum(),
                                    -1,
                                    *outputs[k][time_point_selector].shape[2:],
                                )
                            )
                        outputs[k][time_point_selector, : out.shape[0]] = out

            if (
                np.isclose(t_list[n + 1], period)
                and np.isclose(outputs["PEEQ"][t_output <= period], 0.0).all()
            ):
                # Implement early stopping if within the elastic limits
                return t_output, outputs, 1
            elif (t_list[n + 1] >= 5 * period) and (comp_cycle_marker[n + 1]):
                # Implement early stopping if within the shakedown limits after 30 cycles
                time_point_selector = t_output <= t_list[n + 1]
                delta_peeq = np.diff(
                    outputs["PEEQ"][out_cycle_marker & time_point_selector], axis=0
                )
                if (delta_peeq[-3:] < 2e-5).all():
                    return t_output, outputs, 2

    return t_output, outputs, 0
