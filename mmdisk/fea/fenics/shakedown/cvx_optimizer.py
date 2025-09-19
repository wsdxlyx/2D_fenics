import cvxpy as cp
import numpy as np

from .base import SDProblem


class CVXSDProblem(SDProblem):

    def build_system(self):
        self.r = cp.Variable(1, "r", nonneg=True)
        y_t = cp.Variable([self.Ni * self.Nj, self.Ntau], "y_t")
        y_1p = cp.Variable(self.Nj, "y_p")

        self.rhs = cp.Parameter([(self.Ni - 1) * self.Nj, self.Ntau])
        self.f_ext = cp.Parameter([self.At.shape[0]])

        constraints = [
            cp.SOC(self.r[np.zeros(self.Ni * self.Nj, dtype=np.int32)], y_t, axis=1),
            self.At @ y_t[0 : self.Nj, :].reshape(self.Nj * self.Ntau, order="C")
            + (self.Ap @ y_1p)
            == self.f_ext,
        ]

        # rhs = (self.ct[1:] - self.ct[np.zeros(self.Ni - 1, dtype=np.int32)]).reshape(
        #     (-1, self.Ntau), order="C"
        # )
        y_t_base = y_t[: self.Nj]
        indices = np.tile(np.arange(y_t_base.shape[0]), int(self.Ni - 1))
        lhs = y_t[self.Nj :] - y_t_base[indices]

        constraints.append(lhs == self.rhs)

        self.problem = cp.Problem(cp.Minimize(self.r), constraints)

    # @property
    # def ct_array(self):
    #     return self.ct.value

    def update_c(self, sig):
        f_ext, ct = self._update_c(sig)
        self.f_ext.project_and_assign(f_ext)
        self.ct_array = ct
        rhs = (ct[1:] - ct[0]).reshape((-1, self.Ntau))
        self.rhs.project_and_assign(rhs)

    def optimize_r(self):
        parameters = {
            "ignore_dpp": True,
            **self.params,
        }

        self.problem.solve(**parameters)
        return self.r.value[0]
