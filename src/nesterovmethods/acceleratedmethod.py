from .threecases import three_cases
import math
import numpy as np


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

        while True:
            a = 2 + 2*mi*self.A + math.sqrt((2+2*mi*self.A)*(2+2*mi*self.A) + 8*L*self.A*(1+mi*self.A))
            a = a/(2*L)
            y = (self.A*self.x + a*self.v) / (self.A + a)
            if np.isnan(self.A + a):
                return
            T = three_cases(self.gradient_f(y) - L*y, self.penalty, L)
            if np.dot(self.gradient_f(T), y-T)*L >= np.dot(self.gradient_f(T), self.gradient_f(T)):
                break
            L = L*self.gamma_u

        M = L
        self.L = M/self.gamma_d
        self.x = three_cases(self.gradient_f(y) - M*y, self.penalty, M)
        self.C = self.C + a * self.gradient_f(self.x)
        self.A = self.A + a
        self.v = three_cases(self.C - self.x_0, self.penalty*self.A, 1)
        self.y = y

    def compute_steps(self, no_steps):
        for _ in range(no_steps):
            self.accelerated_gradient_iteration()
