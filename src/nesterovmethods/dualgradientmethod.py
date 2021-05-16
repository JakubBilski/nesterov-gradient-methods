from .gradientiteration import gradient_iteration
from .threecases import three_cases

class DualGradientMethod:
    def __init__(self, penalty, gamma_u, gamma_d, L_0, v_0, psi, f, gradient_f) -> None:
        self.penalty = penalty
        self.gamma_u = gamma_u
        self.gamma_d = gamma_d
        self.L_0 = L_0
        self.L = L_0
        self.v_0 = v_0
        self.v = v_0
        self.psi = psi
        self.f = f
        self.gradient_f = gradient_f
        self.A = 0
        self.B = 0

    def compute_steps(self, no_steps):
        for _ in range(no_steps):
            self.y, M = gradient_iteration(
                self.penalty,
                self.gamma_u,
                self.v,
                self.L,
                self.psi,
                self.f,
                self.gradient_f)
            self.L = max(self.L_0, M/self.gamma_d)
            self.A = self.A + self.gradient_f(self.v) / M
            self.B = self.B + 1 / M
            changes = self.A - self.v_0
            self.v = three_cases(changes, self.penalty*self.B, 1)
