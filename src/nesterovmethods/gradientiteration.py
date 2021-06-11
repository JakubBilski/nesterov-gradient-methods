import numpy as np

from .threecases import three_cases

class DebugInfo:
    NUM_PERFORMED_ITERATIONS = 0

def gradient_iteration(penalty, gamma_u, x, M, psi, f, gradient_f):
    L = M
    DebugInfo.NUM_PERFORMED_ITERATIONS = 0
    while True:
        DebugInfo.NUM_PERFORMED_ITERATIONS += 1
        changes = gradient_f(x) - L*x
        T = three_cases(changes, penalty, L)
        if not psi(T) > np.matmul(gradient_f(T), (x - T)) + np.square(x - T).sum()*L/2 + psi(x):
            break
        L = L * gamma_u
    # S_L = np.linalg.norm(gradient_f(T) - gradient_f(x)) / np.linalg.norm(T - x)
    return T, L
