from .gradientiteration import gradient_iteration

class BasicMethod:
    def __init__(self, penalty, gamma_u, gamma_d, L_0, y_0, psi, f, gradient_f) -> None:
        self.penalty = penalty
        self.gamma_u = gamma_u
        self.gamma_d = gamma_d
        self.L_0 = L_0
        self.L = L_0
        self.y = y_0
        self.psi = psi
        self.f = f
        self.gradient_f = gradient_f

    def compute_steps(self, no_steps):
        for _ in range(no_steps):
            self.y, M = gradient_iteration(
                self.penalty,
                self.gamma_u,
                self.y,
                self.L,
                self.psi,
                self.f,
                self.gradient_f)
            self.L = max(self.L_0, M/self.gamma_d)
