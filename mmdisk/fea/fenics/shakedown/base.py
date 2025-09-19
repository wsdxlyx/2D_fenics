import numpy as np
from scipy.sparse import block_diag

from mmdisk.fea.fenics.utils import delete_from_csr


class SDProblem:
    def __init__(self, args, prob_type="3D", **kwargs):
        self.type = prob_type
        self.Ni = args["Ni"]
        self.Nj = args["Nj"]
        self.H = args["H"]
        if self.type == "3D":
            self.Ntau = 5
            self.Nsig = 6
        if self.type == "Axisymmetric":
            self.Ntau = 3
            self.Nsig = 4

        self.problem = None
        self.r = None
        self.ct = None
        self.f_ext = None

        self.params = kwargs

    def to_solver_sparse(self, A):
        return A

    def update_matrix(self, Y):
        if self.type == "3D":
            Q_local = np.array(
                [
                    [2**0.5 / 2.0, 2**0.5 / 2.0, 2**0.5 / 2.0, 0.0, 0.0, 0.0],
                    [1 / 2.0, -1, 1 / 2.0, 0.0, 0.0, 0.0],
                    [-(3**0.5) / 2.0, 0.0, 3**0.5 / 2.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 3**0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 3**0.5, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 3**0.5],
                ]
            )
        if self.type == "Axisymmetric":
            Q_local = np.array(
                [
                    [2**0.5 / 2.0, 2**0.5 / 2.0, 2**0.5 / 2.0, 0.0],
                    [1 / 2.0, -1, 1 / 2.0, 0.0],
                    [-(3**0.5) / 2.0, 0.0, 3**0.5 / 2.0, 0.0],
                    [0.0, 0.0, 0.0, 3**0.5],
                ]
            )
        Q_inv_local = np.linalg.inv(Q_local)
        self.Q = block_diag([Q_local / Y[j] for j in range(self.Nj)])
        self.Q_inv = block_diag([Q_inv_local * Y[j] for j in range(self.Nj)])

        self.A = self.H @ self.Q_inv
        Ap = self.A[:, :: self.Nsig]
        At = delete_from_csr(
            self.A,
            row_indices=[],
            col_indices=list(np.arange(0, self.Nj * self.Nsig, self.Nsig)),
        )
        self.At = self.to_solver_sparse(At)
        self.Ap = self.to_solver_sparse(Ap)
        self.build_system()

    def build_system(self):
        raise NotImplementedError(
            "This method should be implemented in a derived class."
        )

    def _update_c(self, sig):
        f_ext = self.A @ self.Q @ sig[0]
        c = np.array([self.Q @ sig[i] for i in range(self.Ni)])
        ct = np.delete(c, np.arange(0, self.Nj * self.Nsig, self.Nsig), axis=1)
        return f_ext, ct

    def update_c(self, sig):
        raise NotImplementedError(
            "This method should be implemented in a derived class."
        )

    def optimize_r_EL(self):
        ct = self.ct_array.reshape(self.Ni, self.Nj, self.Ntau)
        r = ((ct[:, :, 0] ** 2 + ct[:, :, 1] ** 2 + ct[:, :, 2] ** 2) ** 0.5).max()
        return r

    def optimize_r(self) -> float:
        raise NotImplementedError(
            "This method should be implemented in a derived class."
        )
