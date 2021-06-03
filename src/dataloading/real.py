import sklearn.datasets 


def get_data():
    return sklearn.datasets.load_breast_cancer(return_X_y=True)