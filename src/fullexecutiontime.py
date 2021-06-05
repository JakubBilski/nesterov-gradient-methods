import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

import nesterovmethods as nvms
import dataloading.generated
import dataloading.real


def get_problem_functions(y, X, penalty):
    def psi(x): return np.abs(x).sum()*penalty
    def f(x): return np.square(np.linalg.norm(y-np.matmul(X, x), ord=2))/2
    def gradient_f(x): return np.matmul((np.matmul(X, x)-y), X)
    return psi, f, gradient_f


if __name__ == '__main__':
    use_generated_data = True

    if use_generated_data:
        X, y = dataloading.generated.get_data(1000, 100)
    else:
        X, y = dataloading.real.get_data()
    n = y.shape[0]
    p = X.shape[1]

    # fig, ax = plt.subplots()
    # ax.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    penalty = 1
    no_steps = 2000
    gamma_u = 2
    gamma_d = 2
    L_0 = 10
    mi = 0
    np.random.seed(1)
    y_0 = np.random.random(size=p)

    psi, f, gradient_f = get_problem_functions(y, X, penalty)

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
        gamma_d=gamma_d,
        f=f,
        gradient_f=gradient_f,
        penalty=penalty
    )

    start_time = time.time()
    basic.compute_steps(no_steps)
    basic_time = time.time() - start_time

    start_time = time.time()
    dual.compute_steps(no_steps)
    dual_time = time.time() - start_time

    start_time = time.time()
    accelerated.compute_steps(no_steps)
    accelerated_time = time.time() - start_time

    print(f"Basic time: {basic_time}")
    print(f"Dual time: {dual_time}")
    print(f"Accelerated time: {accelerated_time}")
