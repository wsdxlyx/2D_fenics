import numpy as np
from dolfin import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    Identity,
    LogLevel,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    as_tensor,
    as_vector,
    assemble,
    assemble_system,
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
    sym,
    tr,
)

from mmdisk.fea.fenics.utils import (
    AbPETScContext,
    delete_from_csr,
    get_boundary,
    get_loadfunc,
    get_time_list_alt,
    to_scipy_csr,
)


def _get_solver(solver_name):
    if solver_name == "mosek":
        from mmdisk.fea.fenics.shakedown.mosek_optimizer import (
            MosekSDProblem as SDProblem,
        )
    elif solver_name == "cvx":
        from mmdisk.fea.fenics.shakedown.cvx_optimizer import CVXSDProblem as SDProblem
    else:
        try:
            from mmdisk.fea.fenics.shakedown.mosek_optimizer import (
                MosekSDProblem as SDProblem,
            )
        except ImportError:
            from mmdisk.fea.fenics.shakedown.cvx_optimizer import (
                CVXSDProblem as SDProblem,
            )
    return SDProblem


def lower_bound_fea(
    mesh,
    properties,
    cycle,
    time_division=None,
    solver=None,
    omega0=100,
    T0=50,
    scale=10.0,
    num_dir=19,
    verbose=False,
    solver_args=None,
):
    if not verbose:
        set_log_level(LogLevel.ERROR)
    else:
        set_log_level(LogLevel.DEBUG)

    if solver_args is None:
        solver_args = {}
    solver_args["verbose"] = solver_args.get("verbose", verbose)

    if time_division is None:
        time_division = [0.5, 0.05, 0.5, 0.2]

    time_steps = cycle.copy() * time_division
    t_list = get_time_list_alt(cycle, time_steps, 1, skip_cold=False)

    # material properties
    rho, cp, kappa, E, sig0, nu, alpha_V = properties
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2.0 / (1 + nu)
    Et = E / 1e5  # tangent modulus - any better method for perfect plasticity?
    H = E * Et / (E - Et)  # hardening modulus
    Y = sig0.vector()[:]

    # DEFINE THERMAL PROBLEM
    U = FunctionSpace(mesh, "CG", 1)  # Temperature space
    v = TestFunction(U)
    x = SpatialCoordinate(mesh)  # Coords
    T_initial = Constant(0.0)
    T_pre = interpolate(T_initial, U)  # Temp. at last time step
    T_crt = TrialFunction(U)
    dt = Constant(1.0)
    F_thermal = (rho * cp * (T_crt - T_pre) / dt) * v * x[0] * dx + kappa * dot(
        grad(T_crt), grad(v)
    ) * x[0] * dx
    a_thermal, L_thermal = lhs(F_thermal), rhs(F_thermal)

    # DEFINE MECH PROBLEM
    V = VectorFunctionSpace(mesh, "CG", 1)
    V_strain = VectorFunctionSpace(mesh, "DG", 0, dim=4)
    V0 = FunctionSpace(mesh, "DG", 0)

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

    def sigma(v, dT):
        return (lmbda * tr(eps(v)) - alpha_V * (3 * lmbda + 2 * mu) * dT) * Identity(
            3
        ) + 2.0 * mu * eps(v)

    def output_stress(sig):
        sig11 = project(sig[0, 0], V0).vector()
        sig22 = project(sig[1, 1], V0).vector()
        sig33 = project(sig[2, 2], V0).vector()
        sig13 = project(sig[0, 2], V0).vector()
        return np.array([sig11, sig22, sig33, sig13])

    ## DEFINE BCs
    boundary_left, boundary_right, boundary_bot, _ = get_boundary(mesh)

    T_L = Constant(0.0)
    T_R = Constant(0.0)
    T_bc_L = DirichletBC(U, T_L, boundary_left)
    T_bc_R = DirichletBC(U, T_R, boundary_right)
    Thermal_BC = [T_bc_L, T_bc_R]

    U_bc_B = DirichletBC(V.sub(1), 0.0, boundary_bot)
    U_bc_L = DirichletBC(V.sub(0), 0.0, boundary_left)
    Mech_BC = [U_bc_B, U_bc_L]

    metadata = {"quadrature_degree": 1, "quadrature_scheme": "default"}
    dxm = dx(metadata=metadata)

    omega_ = Constant(0.0)
    dT = Function(U)
    v_ = TrialFunction(V)
    u_ = TestFunction(V)

    F_int = rho * omega_**2 * x[0] * u_[0]
    Wint = inner(sigma(v_, dT), eps(u_)) * x[0] * dxm - F_int * x[0] * dxm
    a_m, L_m = lhs(Wint), rhs(Wint)

    T_crt = Function(U, name="Temperature")
    u = Function(V, name="Total displacement")

    def apply_BC_to_H(H):
        dirich_dof_ind = []
        for bc in Mech_BC:
            dirich_dof_ind += list(bc.get_boundary_values().keys())
        dirich_dof_ind.sort()
        H_bc = delete_from_csr(H, dirich_dof_ind)
        return H_bc

    def get_H():
        u_ = TrialFunction(V)
        v = TestFunction(V_strain)

        def eps(v):
            return as_vector(
                [v[0].dx(0), v[0] / x[0], v[1].dx(1), (v[1].dx(0) + v[0].dx(1)) / 2.0]
            )

        b = inner(eps(u_), v) * x[0] * dxm
        B = to_scipy_csr(assemble(b))

        dirich_dof_ind = []
        for bc in Mech_BC:
            dirich_dof_ind += list(bc.get_boundary_values().keys())
        dirich_dof_ind.sort()

        H = B.T.tocsr()

        return apply_BC_to_H(H)

    N_Gauss = 1
    Ni = t_list.shape[0]
    Nj = mesh.cells().shape[0] * N_Gauss
    H = get_H()

    args = {"Ni": Ni, "Nj": Nj, "H": H}

    prob = _get_solver(solver)(args, "Axisymmetric", **solver_args)
    prob.update_matrix(Y)

    with AbPETScContext() as (Am, Lm), AbPETScContext() as (At, Lt):

        def run_simulation(load, omega, t_list, Am=Am, Lm=Lm, At=At, Lt=Lt):
            sig_list = []

            u.assign(interpolate(Constant((0.0, 0.0)), V))
            T_crt.assign(interpolate(Constant(0.0), U))
            omega_.assign(omega)
            dT.assign(interpolate(Constant(0.0), U))

            # solve(a_m == L_m, u, Mech_BC)
            assemble_system(a_m, L_m, Mech_BC, A_tensor=Am, b_tensor=Lm)
            solve(Am, u.vector(), Lm)
            sig = sigma(u, dT)
            sig_list.append(output_stress(sig))

            for n in range(len(t_list) - 1):
                dt.assign(t_list[n + 1] - t_list[n])
                T_R.assign(load(t_list[n + 1]))
                assemble_system(
                    a_thermal, L_thermal, Thermal_BC, A_tensor=At, b_tensor=Lt
                )
                solve(At, T_crt.vector(), Lt)
                T_pre.assign(T_crt)

                dT.assign(T_crt - interpolate(T_initial, U))
                assemble_system(a_m, L_m, Mech_BC, A_tensor=Am, b_tensor=Lm)
                solve(Am, u.vector(), Lm)
                sig = sigma(u, dT)
                sig_list.append(output_stress(sig))
            return sig_list

        def get_sig(omega, T_load):
            load = get_loadfunc(T_load, cycle)
            sig = run_simulation(load, omega, t_list)
            return np.array(sig)

        sig = get_sig(omega0, 0)
        sig = sig.reshape((sig.shape[0], -1), order="f")
        prob.update_c(sig)
        r = prob.optimize_r_EL()
        omega0 = omega0 / r**0.5

        sig = get_sig(1e-5, T0)
        sig = sig.reshape((sig.shape[0], -1), order="f")
        prob.update_c(sig)
        r = prob.optimize_r_EL()
        T0 = T0 / r

        omega0 = omega0 / scale**0.5
        T0 = T0 / scale

        omega_list_EL = []
        T_list_EL = []

        omega_list_SD = []
        T_list_SD = []

        for theta in np.linspace(0, np.pi / 2, num_dir):
            omega = np.sin(theta) ** 0.5 * omega0 + 1e-5
            T_load = np.cos(theta) * T0

            sig = get_sig(omega, T_load)
            sig = sig.reshape((sig.shape[0], -1), order="f")

            prob.update_c(sig)
            r_EL = prob.optimize_r_EL()

            try:
                r = prob.optimize_r()
            except Exception as e:
                print(f"Error in optimization: {e}")
            else:
                T_list_EL.append(T_load / r_EL)
                omega_list_EL.append((omega / r_EL**0.5))

                T_list_SD.append(T_load / r)
                omega_list_SD.append((omega / r**0.5))

    return (
        np.array(T_list_EL),
        np.array(omega_list_EL),
        np.array(T_list_SD),
        np.array(omega_list_SD),
    )
