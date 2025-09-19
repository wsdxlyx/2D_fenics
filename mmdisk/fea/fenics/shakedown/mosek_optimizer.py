import sys

import mosek.fusion.pythonic  # noqa: F401
from mosek.fusion import Domain, Model, ObjectiveSense, Var, Matrix

from .base import SDProblem


class MosekSDProblem(SDProblem):
    def __del__(self):
        if self.problem is not None:
            self.r = None
            self.f_ext = None
            self.ct = None
            self.problem.dispose()
            self.problem = None

    def to_solver_sparse(self, A):
        A_coo = A.tocoo()
        A_mosek = Matrix.sparse(
            A_coo.shape[0], A_coo.shape[1], A_coo.row, A_coo.col, A_coo.data
        )
        return A_mosek

    def build_system(self):
        self.problem = M = Model("Shakedown")
        M.setSolverParam("intpntCoTolDfeas", 1.0e-6)

        if self.params["verbose"]:
            M.setLogHandler(sys.stdout)

        self.r = M.variable("r", 1, Domain.greaterThan(0.0))
        y_t = M.variable("y_t", [self.Ni * self.Nj, self.Ntau], Domain.unbounded())
        y_1p = M.variable("y_p", self.Nj, Domain.unbounded())

        cone_expr = Var.hstack(Var.repeat(self.r, self.Ni * self.Nj), y_t)
        M.constraint("Cone", cone_expr, Domain.inQCone())

        self.ct = M.parameter("ct", [self.Ni, self.Nj * self.Ntau])
        for i in range(1, self.Ni):
            lhs = y_t[self.Nj * i : self.Nj * (i + 1), :] - y_t[: self.Nj, :]
            rhs = self.ct[i, :].reshape([self.Nj, self.Ntau]) - self.ct[0, :].reshape(
                [self.Nj, self.Ntau]
            )
            M.constraint("EQ_{}".format(i), lhs == rhs)

        self.f_ext = M.parameter("f_ext", [self.At.numRows()])
        M.constraint(
            "EQ",
            self.At @ y_t[0 : self.Nj, :].reshape(self.Nj * self.Ntau)
            + (self.Ap @ y_1p)
            == self.f_ext,
        )
        M.objective("obj", ObjectiveSense.Minimize, self.r)

    @property
    def ct_array(self):
        return self.ct.getValue()

    def update_c(self, sig):
        f_ext, ct = self._update_c(sig)
        self.f_ext.setValue(f_ext)
        self.ct.setValue(ct)

    def optimize_r(self) -> float:
        self.problem.solve()
        out = self.r.level()[0]
        return out
