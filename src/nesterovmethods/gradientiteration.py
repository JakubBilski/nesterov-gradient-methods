import numpy as np

def gradient_iteration(penalty, gamma_u, x, M, psi, f, gradient_f):
    L = M
    while True:
        changes = gradient_f(x) - L*x
        T = (np.less_equal(changes, -penalty)*(-changes - penalty) + np.greater_equal(changes, penalty)*(-changes + penalty)) / L
        if not psi(T) > np.matmul(gradient_f(T), (x - T)) + np.square(x - T).sum()*L/2 + psi(x):
            break
        L = L * gamma_u
    # S_L = np.linalg.norm(gradient_f(T) - gradient_f(x)) / np.linalg.norm(T - x)
    return T, L