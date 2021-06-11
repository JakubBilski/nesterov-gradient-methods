import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

import nesterovmethods as nvms
import dataloading.generated
import dataloading.real
from nesterovmethods.gradientiteration import DebugInfo as GIDebugInfo
from nesterovmethods.acceleratedmethod import DebugInfo as AccDebugInfo


def get_problem_functions(y, X, penalty):
    def psi(x): return np.abs(x).sum()*penalty
    def f(x): return np.square(np.linalg.norm(y-np.matmul(X, x), ord=2))/2
    def gradient_f(x): return np.matmul((np.matmul(X, x)-y), X)
    return psi, f, gradient_f


if __name__ == '__main__':
    use_generated_data = True

    if use_generated_data:
        X, y = dataloading.generated.get_regression(30, 500)
    else:
        X, y = dataloading.real.get_data()
    n = y.shape[0]
    p = X.shape[1]

    # fig, ax = plt.subplots()
    # ax.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    penalty = 10
    no_steps = 1000
    gamma_us = [2**x for x in range(1, 5)]
    gamma_us.insert(0, 1.01)
    gamma_d = 2
    L_0 = 10
    mi = 0
    np.random.seed(1)
    y_0 = np.random.random(size=p)

    method = nvms.AcceleratedMethod
    method_name = "accelerated"

    steps_mean = 100

    psi, f, gradient_f = get_problem_functions(y, X, penalty)

    fig, ax = plt.subplots(figsize=(8,8))
    data_type = "generated" if use_generated_data else "real"
    plot2_ys = []
    plot2_gamma_us = []

    for gamma_u in gamma_us:
        if method == nvms.AcceleratedMethod:
            model = method(
                L_0=L_0,
                mi=mi,
                x_0=y_0,
                gamma_u=gamma_u,
                gamma_d=gamma_d,
                f=f,
                gradient_f=gradient_f,
                penalty=penalty
            )
        else:
            model = method(
                penalty=penalty,
                gamma_u=gamma_u,
                gamma_d=gamma_d,
                L_0=L_0,
                v_0=y_0,
                psi=psi,
                f=f,
                gradient_f=gradient_f
            )
        errors = []
        num_loop_iters = []
        for _ in tqdm(range(no_steps)):
            model.compute_steps(1)
            errors.append(f(model.y)+psi(model.y))
            if method == nvms.acceleratedmethod.AcceleratedMethod:
                num_loop_iters.append(AccDebugInfo.NUM_PERFORMED_ITERATIONS)
            else:
                num_loop_iters.append(GIDebugInfo.NUM_PERFORMED_ITERATIONS)

        mean_loop_iters = [sum(num_loop_iters[i-steps_mean:i])/steps_mean for i in range(steps_mean, len(num_loop_iters))]
        plot2_ys.append(mean_loop_iters)
        plot2_gamma_us.append(gamma_u)
        ax.plot(range(no_steps), errors, label=f'gamma_u={gamma_u}')

    ax.set_xlabel("step")
    ax.set_ylabel("loss function")
    ax.set_yscale("log")
    # ax.set_xlim(-10, 5550)
    ax.set_ylim(10**2, 2*10**2)
    plt.suptitle(f"{data_type} data: method {method_name} penalty {penalty}, gamma_d {gamma_d}, L_0 {L_0}, mi {mi}")
    ax.set_title("Loss function by number of steps")
    plt.legend()
    plt.show()
    plt.close()


    fig2, ax2 = plt.subplots(figsize=(8,8))
    for ys, gamma_u in zip(plot2_ys, plot2_gamma_us):
        ax2.plot(range(steps_mean, no_steps), ys, label=f'gamma_u={gamma_u}')

    ax2.set_xlabel("step")
    ax2.set_ylabel("number of iterations")
    ax2.set_yscale("log")
    plt.suptitle(f"{data_type} data: method {method_name} penalty {penalty}, gamma_d {gamma_d}, L_0 {L_0}, mi {mi}")
    ax2.set_title(f"Mean number of iterations in inner loop during last {steps_mean} steps")
    plt.legend()
    plt.show()
