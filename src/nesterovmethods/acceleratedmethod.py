from .threecases import three_cases
import math
import numpy as np


class DebugInfo:
    NUM_PERFORMED_ITERATIONS = 0

class AcceleratedMethod:
    def __init__(self, L_0, mi, x_0, gamma_u, gamma_d, f, gradient_f, penalty) -> None:
        self.L = L_0
        self.mi = mi
        self.x_0 = x_0
        self.x = x_0
        self.gamma_u = gamma_u
        self.gamma_d = gamma_d
        self.A = 0
        self.penalty = penalty
        self.f = f
        self.gradient_f = gradient_f
        self.C = 0
        self.v = x_0

    def accelerated_gradient_iteration(self):
        L = self.L
        mi = self.mi
        DebugInfo.NUM_PERFORMED_ITERATIONS = 0

        while True:
            DebugInfo.NUM_PERFORMED_ITERATIONS += 1
            a = 2 + 2*mi*self.A + math.sqrt((2+2*mi*self.A)*(2+2*mi*self.A) + 8*L*self.A*(1+mi*self.A))
            y = (self.A*self.x*2*L + a*self.v) / (self.A*2*L + a)
            T = three_cases(self.gradient_f(y) - L*y, self.penalty, L)
            diff = self.gradient_f(y) - self.gradient_f(T)
            if np.dot(diff, y-T) >= (np.linalg.norm(diff, ord=2)**2)/L:
                break
            L = L*self.gamma_u

        M = L
        self.L = M/self.gamma_d
        self.x = three_cases(self.gradient_f(y) - M*y, self.penalty, M)
        self.C = self.C + a * self.gradient_f(self.x) / (2 * L)
        self.A = self.A + a / (2 * L)
        self.v = three_cases(self.C - self.x_0, self.penalty*self.A, 1)
        self.y = y

    def compute_steps(self, no_steps):
        for _ in range(no_steps):
            self.accelerated_gradient_iteration()
