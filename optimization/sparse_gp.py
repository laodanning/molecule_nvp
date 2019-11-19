import GPy
import numpy as np

# x = np.random.rand(10,1)
# y = np.exp(x)
# a = GPy.models.SparseGPRegression(x, y, num_inducing=5)
# print(a)
# a.optimize("lbfgs")
# print(a)

# x_new = np.random.rand(5, 1)
# y_new = np.exp(x_new)
# y_mean, y_var = a.predict(x_new)
# print(y_mean.shape)
# y_pred = np.random.multivariate_normal(y_mean.reshape(-1), y_var.reshape(-1) * np.eye(5), size=(1)).reshape(5, 1)
# print(y_new.shape)
# print(y_pred.shape)
# print("err: {}".format(np.sum((y_new - y_pred) ** 2)))

class SparseGP(object):
    pass

class BayesionOptimization(object):
    pass
