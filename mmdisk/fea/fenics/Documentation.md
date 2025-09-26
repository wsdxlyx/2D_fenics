# OVERVIEW
This is a documentation for important functions in FEniCS incremental FEA developed by Shuheng Liao. Comments are added by Shen Wang.


## CONTENTS
1. incremental_cyclic_fea( ) in [*incremental.py*](incremental.py)
- ...



## 1. incremental_cyclic_fea()
### Overview
what does this function do

### Parameters
- ___mesh___: _dolfin Mesh object_
- ___properties___: _tuple_
    A tuple containing 7 material properties ($\rho$, $C_p$, $k$, $E$, $\sigma_{y0}$, $\nu$, $\alpha$), with each element a dolfin Function object.
- ___load___: _callable_
    callable function of _get_loadfunc(T_load, cycle)_
- ___omega___: _float_
    Disk rotational speed [rad/s]
- ___t_list___: _np.ndarray_
    time sequence defining thermal cycles
- ___t_output___: _np.ndarray_
    time sequence at which results are output
- ___period___: _float_
    thermal cycle period [s]
- ___Displacement_BC___: _str="fix_free"_
    disk hub and rim boundary conditions, either "fix-free" or "free-free"
- ___plastic_inverval___: _int = 1_
    time increment inverval for plasticity update 
- ___tol___: _float = 1e-5_
    convergence tolerance
- ___Nitermax___: _int = 50_
    maximum number of iterations for convergence
- ___skip_cold___: _bool = False_,
    If True, skips the cold dwell for the thermal problem.
- ___verbose___: _bool = False_,
    If True, enables verbose logging.
- ___outputs___: _Union [None, list, dict] = None_
    Outputs to be collected during the analysis. If None, defaults to collecting "PEEQ". If a list, it will be converted to a dictionary with empty arrays. If a dictionary， it will be used to store the outputs with preallocated arrays. Available outputs: "PEEQ" (equivalent plastic strain), "T" (temperature), "u" (displacement), "sig" (stress), "eps_vector" (strain vector [$\varepsilon_{rr}$, $\varepsilon_{\theta\theta}$, $\varepsilon_{zz}$, $\varepsilon_{rz}$]).



### Returns
- ___t_output___: _np.ndarray_
    the same as input parameters
- ___outputs___: _dict_
    the result variables requested
- ___flag___: _int_
     - 0: normal completion
     - 1: early stopping due to elastic limits
     - 2: early stopping due to shakedown limits

### Internal Funcitons
#### eps()
- __Overview__
     calculate strain tensor from displacement for the axisymmetric problem: 
    $\bm{u} = u_r(r,z)\bm{e_r} + u_z(r,z)\bm{e_z}$ <br>
    $\varepsilon$ = $\begin{bmatrix}
                        \frac{\partial u_r}{\partial r} & 0 & \frac{1}{2}(\frac{\partial u_r}{\partial z} + \frac{\partial u_z}{\partial r}) \\
                        0 & \frac{u_r}{r} & 0 \\
                        \frac{1}{2}(\frac{\partial u_r}{\partial z} + \frac{\partial u_z}{\partial r}) & 0 & \frac{\partial u_z}{\partial z}
                    \end{bmatrix}$
- __Parameters__
    - ___v___: _list_,
    displacement vector [$u_r$, $u_z$]
- __Returns__
    - ___eps___: _UFL ListTensor object_,
    $\begin{bmatrix}
        \varepsilon_{rr} & \varepsilon_{r\theta} & \varepsilon_{rz} \\
        \varepsilon_{r\theta} & \varepsilon_{\theta\theta} & \varepsilon_{\theta z} \\
        \varepsilon_{rz} & \varepsilon_{\theta z} & \varepsilon_{zz}
    \end{bmatrix}$

#### eps_to_vector()
- __Overview__
     calculate strain components from displacement for the axisymmetric problem: 
    $\bm{u} = u_r(r,z)\bm{e_r} + u_z(r,z)\bm{e_z}$ <br>
    $\varepsilon$ = $\begin{bmatrix}
                        \frac{\partial u_r}{\partial r} & 0 & \frac{1}{2}(\frac{\partial u_r}{\partial z} + \frac{\partial u_z}{\partial r}) \\
                        0 & \frac{u_r}{r} & 0 \\
                        \frac{1}{2}(\frac{\partial u_r}{\partial z} + \frac{\partial u_z}{\partial r}) & 0 & \frac{\partial u_z}{\partial z}
                    \end{bmatrix}$
- __Parameters__
    - ___v___: _list_,
    displacement vector [$u_r$, $u_z$]
- __Returns__
    - ___eps___: _UFL ListTensor object_,
    [$\varepsilon_{rr}$, $\varepsilon_{\theta\theta}$, $\varepsilon_{zz}$, $\varepsilon_{rz}$]

#### sigma()
- __Overview__
     calculate stress using linear thermoelastic constitutive relationship: 
    $\bm{\sigma} = \big[\lambda \text{tr}(\varepsilon) - (3\lambda+2\mu)\alpha \Delta T\big] \bm{I} + 2\mu \bm{\varepsilon} $
- __Parameters__
    - ___v___: _list_,
    displacement vector [$u_r$, $u_z$]
    - ___dT___: _dolfin Function object_,
    $\Delta T$
- __Returns__
    - ___sigma___: _UFL ListTensor object_,
    $\bm{\sigma}$?????????

####  sigma_tang()
- __Overview__
      calculate $\Delta \bm{\sigma}$ using given $\Delta \bm{u}$ and consistent tangent modulus in Newton-Raphson alogrithm.
      $\Delta \bm{\sigma} = \mathbb{C}: \Delta \bm{\varepsilon} = \lambda \,\mathrm{tr}(\Delta\bm{\varepsilon}) \mathbf{I} + 2\mu \Delta\bm{\varepsilon} - 3\mu \left(\frac{3\mu}{3\mu + H} - \beta \right) \left( \bm{N}_{\mathrm{elas}} : \Delta\bm{\varepsilon} \right) \, \bm{N}_{\mathrm{elas}} - 2\mu \beta \, \Delta \bm{\varepsilon}_{dev})$
- __Parameters__
    - ___$\Delta \bm{u}$___: _UFL object_,
    trial function
- __Returns__
    - ___$\Delta \bm{\sigma}$___: _UFL ListTensor object_,

### Line-by-line Explanation
```python
if not verbose:
    set_log_level(LogLevel.ERROR)  # SHOW ONLY ERROR MESSAGE
else:
    set_log_level(LogLevel.DEBUG). # SHOW DEBUG MESSAGE
```

```python
# GET MATERIAL PROPERTIES
rho, cp, kappa, E, sig0, nu, alpha_V = properties
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2.0 / (1 + nu)
Et = E / 1e5
H = E * Et / (E - Et)
```
Lame constants $\lambda = \frac{E\nu}{(1+\nu)(1+2\nu)}$, $\mu = \frac{E}{2(1+\nu)}$ 
Tangent modulus $E_t = E\times10^{-5}$. Need to have non-zero $E_t$, haven't found a better way.
Hardening modulus $H = \frac{E E_t}{E-E_t}$


```python
# DEFINE THERMAL PROBLEM WITH BACKWARD EULER DISCRETIZATION
V = FunctionSpace(mesh, "P", 1).   # TEMPERATURE FUNCTION SPACE
v = TestFunction(V)                # TEST FUNCTION
x = SpatialCoordinate(mesh)        # COORDIANTES
T_initial = Constant(0.0)          # INITIAL TEMPERATURE
T_pre = interpolate(T_initial, V)  # LAST THERMAL STEP TEMPERATURE T^(n+1)
T_old = interpolate(T_initial, V)  # LAST MECHANICAL STEP TEMPERATURE
T_crt = TrialFunction(V)           # CURRENT THERMAL STEP TEMPERATURE T^n
dt = Constant(1.0)                 # THERMAL TIME STEP
F_thermal = (rho * cp * (T_crt - T_pre) / dt) * v * x[0] * dx + kappa * dot(
    grad(T_crt), grad(v)
) * x[0] * dx                      # WEAK FORM 
a_thermal, L_thermal = lhs(F_thermal), rhs(F_thermal)
```
Weak form of axisymmetric transient thermal problem <br>
$\int_0^R \rho C_p \frac{T^{n+1} - T^n}{\Delta t} \, v \, r \, dr + \int_0^R k \frac{d T^{n+1}}{dr} \frac{dv}{dr} \, r \, dr = 0 \\$
$\implies$ $\int_0^R \rho C_p \frac{T^{n+1}}{\Delta t}  v \, r \, dr + \int_0^R k \frac{d T^{n+1}}{dr} \frac{dv}{dr} \, r \, dr = \int_0^R \rho C_p \frac{T^n}{\Delta t}  v \, r \, dr$
   
```python
# DEFINE MECHANICAL PROBLEM
U = VectorFunctionSpace(mesh, "CG", 1)  # DISPLACEMENT FUNCTION SPACE ON NODES
We = VectorElement(
    "Quadrature", mesh.ufl_cell(), degree=1, dim=4, quad_scheme="default"
)                                       # DEFINE 4-COMPONENT VECTOR QUADRATURE ELEMENT
W = FunctionSpace(mesh, We)             # DEFINE FUNCTION SPACE ON 4-COMPONENT VECTOR QUADRATURE ELEMENT
W0e = FiniteElement("Quadrature", mesh.ufl_cell(), degree=1, quad_scheme="default")  # DEFINE SCALAR QUADRATURE ELEMENT
W0 = FunctionSpace(mesh, W0e)           # DEFINE FUNCTION SPACE ON SCALAR QUADRATURE ELEMENT

metadata = {"quadrature_degree": 1, "quadrature_scheme": "default"}  # DEFINE GAUSSIAN QUADRATURE SCHEME
dxm = dx(metadata=metadata)  # APPLY THE QUADRATURE SCHEME TO THE MEASURE DX 

# OBTAIN VARIABLES FROM FUNCTION SPACES
sig = Function(W)         # CURRENT STEP STRESS
sig_old = Function(W)     # LAST STEP STRESS  
n_elas = Function(W)      # FLOW DIRECTIONS
strain = Function(W, name="Strain vector")          # CURRENT STRAIN

beta = Function(W0)       # PLASTIC CORRECTION FACTOR
p = Function(W0, name="Cumulative plastic strain")  # CUMULATIVE PLASTIC STRAIN (PEEQ IN ABAQUS)

u = Function(U, name="Total displacement")          # TOTAL DISPLACEMENT
du = Function(U, name="Iteration correction")       # RETURN MAPPING CORRETION DISPLACEMENT
Du = Function(U, name="Current increment")          # STEP INCREMENTAL DISPLACEMENT
v_ = TrialFunction(U)     # UNKNOWN VARIABLE TO CONSTRUCT LINEAR SYSTEM
u_ = TestFunction(U)      # TEST FUNCTION IN WEAK FORM
```

```python
# DEFINE INTERNAL FUNCTIONS
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
```

```python
omega_ = Constant(omega)    # TURN OMEGA VALUE INTO FENICS CONSTANT
F_int = rho * omega_**2 * x[0] * u_[0]    # CENTRIFUGAL LOAD IN WEAK FORM

a_Newton = inner(eps(v_), sigma_tang(u_)) * x[0] * dxm  # 
res = -inner(eps(u_), as_3D_tensor(sig)) * x[0] * dxm + F_int * x[0] * dxm
```
$a_{Newton} = \int_{\Omega}  \Delta \bm{\sigma}:\delta\bm{\varepsilon} \, d\Omega $
$Res =  - \int_{\Omega}  \bm{\sigma}:\delta\bm{\varepsilon} \, d\Omega  + \int_{\Omega}  \bm{f}_{\omega} \cdot \delta\bm{u} \, d\Omega$ 

Notes from Shen -- it should be `a_Newton = inner(eps(u_), sigma_tang(v_)) * x[0] * dxm`, however given the symmtry of consistent tangent modulus $\mathbb{C}$ for isotropic hardening used here, `a_Newton = inner(eps(v_), sigma_tang(u_)) * x[0] * dxm` also works mathematically.

```python
## DEFINE BOUNDARY CONDITIONS
boundary_left, boundary_right, boundary_bot, _ = get_boundary(mesh)  # GET MODEL BOUNDARYS
T_L = Constant(0.0)  # UFL CONSTANT FOR THE LEFT BOUNDARY TEMPERATURE
T_R = Constant(0.0)  # UFL CONSTANT FOR THE RIGHT BOUNDARY TEMPERATURE
T_bc_L = DirichletBC(V, T_L, boundary_left)   # ASSIGN THE LEFT BOUNDARY OF TEMPERATURE FUNCTION SPACE WITH DIRICHLET BC 
T_bc_R = DirichletBC(V, T_R, boundary_right)  # ASSIGN THE RIGHT BOUNDARY OF TEMPERATURE FUNCTION SPACE WITH DIRICHLET BC 
Thermal_BC = [T_bc_L, T_bc_R]

U_bc_B = DirichletBC(U.sub(1), 0.0, boundary_bot)   # ASSIGN THE BOTTOM BOUNDARY OF DISPLACEMENT FUNCTION SPACE WITH DIRICHLET BC (FIX Uz DIRECTION DISPLACEMENT)
# U_bc_L = DirichletBC(U, Constant((0.,0.)), boundary_left)
U_bc_L = DirichletBC(U.sub(0), 0.0, boundary_left)  # ASSIGN THE BOTTOM BOUNDARY OF DISPLACEMENT FUNCTION SPACE WITH DIRICHLET BC (FIX Ur DIRECTION DISPLACEMENT)
if Displacement_BC == "fix-free":
    Mech_BC = [U_bc_B, U_bc_L]
else:
    Mech_BC = [U_bc_B]
```

```python
P0 = FunctionSpace(mesh, "DG", 0)   # NON-CONTINUOUS FUNCTION SPACE ON NODES
p_avg = Function(P0)                # ACCUMULATIVE EQUIVALENT PLASTIC STRAIN
T_crt = Function(V)                 # CURRENT STEP TEMPERATURE
dT = Function(V)                    # ????????????????

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
```
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

### Example


### Notes
N/A

```python
hi
```