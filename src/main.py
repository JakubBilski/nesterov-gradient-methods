import matplotlib.pyplot as plt
import numpy as np

import nesterovmethods as nvms
import dataloading.generated

def get_problem_functions(y, X, penalty):
    psi = lambda x: np.abs(x).sum()*penalty
    f = lambda x: np.square(np.linalg.norm(y-np.matmul(X, x)))/2
    gradient_f = lambda x: (np.transpose(np.transpose(x)*X) - np.transpose(y)).sum(axis=1)
    return psi, f, gradient_f

if __name__ == '__main__':
    n = 1000
    p = 2
    no_classes = 3
    penalty = 1

    X, y = dataloading.generated.get_data(p, n, no_classes)

    # fig, ax = plt.subplots()
    # ax.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    psi, f, gradient_f = get_problem_functions(y, X, penalty)

    np.random.seed(1)
    y_0 = np.random.random(size=p)

    basic = nvms.BasicMethod(
        penalty=penalty,
        gamma_u=1.001,
        gamma_d=1,
        L_0=1,
        y_0=y_0,
        psi=psi,
        f=f,
        gradient_f=gradient_f
    )

    no_steps = 100
    errors = []
    for _ in range(no_steps):
        basic.compute_steps(1)
        errors.append(f(basic.y)+psi(basic.y))
    
    print("Final solution:")
    print(basic.y)
    
    fig, ax = plt.subplots()
    ax.plot(range(no_steps), errors, label='basic')
    plt.legend()
    plt.show()