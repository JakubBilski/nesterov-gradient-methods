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
    # ax.scatter(X[:, 0], y)
    # ax.set_xlabel("x1")
    # ax.set_ylabel("y")
    # plt.show()

    penalty = 10
    no_steps = 1000
    gamma_u = 2
    gamma_d = 2
    L_0 = 0.1
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

    basic_errors = []
    dual_errors = []
    accelerated_errors = []
    basic_Ls = []
    dual_Ls = []
    accelerated_Ls = []
    basic_times = []
    dual_times = []
    accelerated_times = []
    basic_loop_iters = []
    dual_loop_iters = []
    accelerated_loop_iters = []

    for _ in tqdm(range(no_steps)):
        start_time = time.time()
        basic.compute_steps(1)
        basic_times.append(time.time()-start_time)
        basic_errors.append(f(basic.y)+psi(basic.y))
        basic_Ls.append(basic.L)
        basic_loop_iters.append(GIDebugInfo.NUM_PERFORMED_ITERATIONS)
        start_time = time.time()
        dual.compute_steps(1)
        dual_times.append(time.time()-start_time)
        dual_errors.append(f(dual.y)+psi(dual.y))
        dual_Ls.append(dual.L)
        dual_loop_iters.append(GIDebugInfo.NUM_PERFORMED_ITERATIONS)
        start_time = time.time()
        accelerated.compute_steps(1)
        accelerated_times.append(time.time()-start_time)
        accelerated_errors.append(f(accelerated.y)+psi(accelerated.y))
        accelerated_Ls.append(accelerated.L)
        accelerated_loop_iters.append(AccDebugInfo.NUM_PERFORMED_ITERATIONS)

    # print("Final solution:")
    # print(basic.y)
    # print(dual.y)
    # print(accelerated.y)

    print("Share of zeroes after last step:")
    print(sum(basic.y == 0) / basic.y.size)
    print(sum(dual.y == 0) / dual.y.size)
    print(sum(accelerated.y == 0) / accelerated.y.size)

    figsize = (7, 7)
    data_type = "generated" if use_generated_data else "real"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(no_steps), basic_errors, label='basic')
    ax.plot(range(no_steps), dual_errors, label='dual gradient')
    ax.plot(range(no_steps), accelerated_errors, label='accelerated')
    ax.set_xlabel("step")
    ax.set_ylabel("loss function")
    ax.set_yscale("log")
    # ax.set_ylim(-1, 1.1*10**3)
    plt.suptitle(f"{data_type} data: penalty {penalty}, gamma_u {gamma_u}, gamma_d {gamma_d}, L_0 {L_0}, mi {mi}")
    ax.set_title("Loss function by number of steps")
    plt.legend()
    plt.show()


    steps_mean = 50
    mean_basic_Ls = [sum(basic_Ls[i-steps_mean:i])/steps_mean for i in range(steps_mean, len(basic_Ls))]
    mean_dual_Ls = [sum(dual_Ls[i-steps_mean:i])/steps_mean for i in range(steps_mean, len(dual_Ls))]
    mean_accelerated_Ls = [sum(accelerated_Ls[i-steps_mean:i])/steps_mean for i in range(steps_mean, len(accelerated_Ls))]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(steps_mean, no_steps), mean_basic_Ls, label='basic')
    ax.plot(range(steps_mean, no_steps), mean_dual_Ls, label='dual gradient')
    ax.plot(range(steps_mean, no_steps), mean_accelerated_Ls, label='accelerated')
    ax.set_xlabel("step")
    ax.set_ylabel(f"mean L value in last {steps_mean} steps")
    ax.set_yscale("log")
    plt.suptitle(f"{data_type} data: penalty {penalty}, gamma_u {gamma_u}, gamma_d {gamma_d}, L_0 {L_0}, mi {mi}")
    ax.set_title(f"Estimation of Lipschitz constant by number of steps")
    plt.legend()
    plt.show()

    cumulative_basic_times = np.cumsum(basic_times)
    cumulative_dual_times = np.cumsum(dual_times)
    cumulative_accelerated_times = np.cumsum(accelerated_times)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cumulative_basic_times, basic_errors, label='basic')
    ax.plot(cumulative_dual_times, dual_errors, label='dual gradient')
    ax.plot(cumulative_accelerated_times, accelerated_errors, label='accelerated')
    ax.set_xlabel("time of execution [s]")
    ax.set_ylabel("loss function")
    ax.set_yscale("log")
    plt.suptitle(f"{data_type} data: penalty {penalty}, gamma_u {gamma_u}, gamma_d {gamma_d}, L_0 {L_0}, mi {mi}")
    ax.set_title("Loss function by time of execution")
    plt.legend()
    plt.show()


    # steps_mean = 1000
    # mean_basic_loop_iters = [sum(basic_loop_iters[i-steps_mean:i])/steps_mean for i in range(steps_mean, len(basic_loop_iters))]
    # mean_dual_loop_iters = [sum(dual_loop_iters[i-steps_mean:i])/steps_mean for i in range(steps_mean, len(dual_loop_iters))]
    # mean_accelerated_loop_iters = [sum(accelerated_loop_iters[i-steps_mean:i])/steps_mean for i in range(steps_mean, len(accelerated_loop_iters))]

    # fig, ax = plt.subplots(figsize=figsize)
    # ax.plot(range(steps_mean, no_steps), mean_basic_loop_iters, label='basic')
    # ax.plot(range(steps_mean, no_steps), mean_dual_loop_iters, label='dual gradient')
    # ax.plot(range(steps_mean, no_steps), mean_accelerated_loop_iters, label='accelerated')
    # ax.set_xlabel("step")
    # ax.set_ylabel("number of iterations")
    # plt.suptitle(f"{data_type} data: penalty {penalty}, gamma_u {gamma_u}, gamma_d {gamma_d}, L_0 {L_0}, mi {mi}")
    # ax.set_title(f"Mean number of iterations in inner loop during last {steps_mean} steps")
    # plt.legend()
    # plt.show()
