from .threecases import three_cases
import math
import numpy as np


class AcceleratedMethod:
    def __init__(self, L_0, mi, x_0, gamma_u, f, gradient_f, penalty) -> None:
        self.L_k = L_0
        self.mi = mi
        self.x_0 = x_0
        self.x_k = x_0
        self.gamma_u = gamma_u
        self.A_k = 0
        self.penalty = penalty
        self.f = f
        self.gradient_f = gradient_f
        self.C_k = 0
        self.b_k = 0
        self.v_k = x_0  # idk...

    def accelerated_gradient_iteration(self):
        L = self.L_k
        mi = self.mi
        A = self.A_k
        x = self.x_k
        v = self.v_k
        x_0 = self.x_0
        penalty = self.penalty
  
        while True:
            a = 2 + 2*mi*A + math.sqrt((2+2*mi*A)*(2+2*mi*A) + 8*L*A*(1+mi*A))
            a = a/(2*L)
            y = (A*x + a*v) / (A + a)

            changes = self.gradient_f(y) - L*y
            T = three_cases(changes, penalty, L)

            if np.dot(self.gradient_f(T), y-T) >= np.linalg.norm(self.gradient_f(T))*np.linalg.norm(self.gradient_f(T))/L:
                break

            L = L*self.gamma_u

        y_k = y
        M_k = L
        a_k1 = a
        L_k1 = M_k/self.gamma_u
        changes = self.gradient_f(y_k) - M_k*y_k
        x_k1 = three_cases(changes, self.penalty, M_k)
        A_k1 = A + a_k1

        C_k1 = self.C_k + a_k1 * self.gradient_f(x_k1)
        b_k1 = self.b_k + a_k1

        v_k1 = (np.less_equal(C_k1 - x_0, -penalty*b_k1)*(x_0 - C_k1 - penalty*b_k1) + np.greater_equal(C_k1 - x_0, penalty*b_k1)*(x_0 - C_k1 + penalty*b_k1))

        self.L_k = L_k1
        self.x_k = x_k1
        self.A_k = A_k1
        self.C_k = C_k1
        self.b_k = b_k1
        self.v_k = v_k1

        self.y = y_k
        self.M = M_k

    def compute_steps(self, no_steps):
        for _ in range(no_steps):
            self.accelerated_gradient_iteration()
