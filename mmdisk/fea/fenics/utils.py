import numpy as np
from dolfin import (
    PETScMatrix,
    PETScVector,
    as_backend_type,
    as_tensor,
    near,
)
from scipy.sparse import csr_matrix


def get_boundary(mesh):
    r1 = mesh.coordinates()[:, 0].min()
    r2 = mesh.coordinates()[:, 0].max()
    h1 = mesh.coordinates()[:, 1].min()
    h2 = mesh.coordinates()[:, 1].max()

    # BCs
    def bound_left(x):
        return near(x[0], r1)

    def bound_right(x):
        return near(x[0], r2)

    def bound_bot(x):
        return near(x[1], h1)

    def bound_top(x):
        return near(x[1], h2)

    return bound_left, bound_right, bound_bot, bound_right


def get_loadfunc(T_load, cycle):
    cycle_time = np.sum(cycle)

    t_times = np.zeros(len(cycle) + 1)
    t_times[1:] = np.cumsum(cycle)

    def load(t):
        t = t % cycle_time
        return T_load * np.interp(
            t,
            t_times,
            [0.0, 1.0, 1.0, 0.0, 0.0],
        )

    return load


def get_time_list(t_cycle, t_fine, time_step, time_step_coarse, n_cyc):
    t_list_fine = np.linspace(0, t_fine, int(t_fine / time_step) + 1)
    t_list_coarse = np.linspace(
        t_fine, t_cycle, np.ceil((t_cycle - t_fine) / time_step_coarse).astype(int) + 1
    )
    t_list_cycle = np.concatenate((t_list_fine, t_list_coarse[1:]))
    t_list = t_list_cycle
    for n in range(1, n_cyc):
        t_list = np.concatenate((t_list, t_list_cycle[1:] + n * t_cycle))
    return t_list


def get_time_list_alt(time_intervals, step_list, n_cyc, skip_cold=False):
    t_start = 0.0
    t_cycle = (np.array(time_intervals)).sum()
    t_list_cycle = np.array([0.0])
    for t, step in zip(time_intervals, step_list):
        t_list_local = np.linspace(
            t_start, t_start + t, np.ceil(t / step).astype(int) + 1
        )
        if skip_cold and t == time_intervals[-1]:  # skip the cold dwell
            t_list_local = t_list_local[[0, 1, -1]]  # remove points
        t_list_cycle = np.concatenate((t_list_cycle, t_list_local[1:]))
        t_start = t_start + t
    t_list = t_list_cycle
    for n in range(1, n_cyc):
        t_list = np.concatenate((t_list, t_list_cycle[1:] + n * t_cycle))
    return t_list


def as_3D_tensor(X):
    return as_tensor([[X[0], 0, X[3]], [0, X[1], 0], [X[3], 0, X[2]]])


def ppos(x):
    """Positive part of x."""
    return (x + abs(x)) / 2.0


def to_scipy_csr(A):
    A_petsc = as_backend_type(A).mat()
    ai, aj, av = A_petsc.getValuesCSR()
    return csr_matrix((av, aj, ai), shape=(A.size(0), A.size(1)))


def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:, col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:, mask]
    else:
        return mat


class AbPETScContext:
    def __init__(self):
        self._A = None
        self._b = None

    def __enter__(self):
        self._A = PETScMatrix()
        self._b = PETScVector()
        return self._A, self._b

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self._A is not None:
            self._A.mat().destroy()
            self._A = None
        if self._b is not None:
            self._b.vec().destroy()
            self._b = None
