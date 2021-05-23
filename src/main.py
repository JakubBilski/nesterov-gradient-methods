import matplotlib.pyplot as plt
import numpy as np

import nesterovmethods as nvms
import dataloading.generated


def get_problem_functions(y, X, penalty):
    def psi(x): return np.abs(x).sum()*penalty
    def f(x): return np.square(np.linalg.norm(y-np.matmul(X, x)))/2
    def gradient_f(x): return np.matmul((np.matmul(X, x)-y), X)
    return psi, f, gradient_f


if __name__ == '__main__':
    n = 100
    p = 2
    no_classes = 2
    penalty = 1

    X, y = dataloading.generated.get_data(p, n, no_classes)

    # fig, ax = plt.subplots()
    # ax.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    psi, f, gradient_f = get_problem_functions(y, X, penalty)

    np.random.seed(1)
    y_0 = np.random.random(size=p)
    gamma_u = 1.01
    gamma_d = 2
    L_0 = 1
    mi = 0

    basic = nvms.BasicMethod(
        penalty=penalty,
        gamma_u=gamma_u,
        gamma_d=gamma_d,
        L_0=L_0,
        y_0=y_0,
        psi=psi,
        f=f,
        gradient_f=gradient_f
    )

    dual = nvms.DualGradientMethod(
        penalty=penalty,
        gamma_u=gamma_u,
        gamma_d=gamma_d,
        L_0=L_0,
        v_0=y_0,
        psi=psi,
        f=f,
        gradient_f=gradient_f
    )

    accelerated = nvms.AcceleratedMethod(
        L_0=L_0,
        mi=mi,
        x_0=y_0,
        gamma_u=gamma_u,
        f=f,
        gradient_f=gradient_f,
        penalty=penalty
    )

    no_steps = 100
    basic_errors = []
    dual_errors = []
    accelerated_errors = []
    for _ in range(no_steps):
        basic.compute_steps(1)
        basic_errors.append(f(basic.y)+psi(basic.y))
        dual.compute_steps(1)
        dual_errors.append(f(dual.y)+psi(dual.y))
        accelerated.compute_steps(1)
        accelerated_errors.append(f(accelerated.y)+psi(accelerated.y))

    print("Final solution:")
    print(basic.y)
    print(dual.y)
    print(accelerated.y)

    fig, ax = plt.subplots()
    ax.plot(range(no_steps), basic_errors, label='basic')
    ax.plot(range(no_steps), dual_errors, label='dual gradient')
    ax.plot(range(no_steps), accelerated_errors, label='accelerated')
    plt.legend()
    plt.show()
