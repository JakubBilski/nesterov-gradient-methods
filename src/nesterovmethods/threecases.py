import numpy as np

def three_cases(changes, penalty, denominator):
    return (np.less_equal(changes, -penalty)*(-changes - penalty) + np.greater_equal(changes, penalty)*(-changes + penalty)) / denominator