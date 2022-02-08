import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C




def classificationModels():
    models = {}

    ######################################
    # Logistic Regression
    ######################################
    from sklearn.gaussian_process import GaussianProcessRegressor
    models['Gaussian Process'] = {}
    models['Gaussian Process'] = GaussianProcessRegressor()

    return models

    ######################################
