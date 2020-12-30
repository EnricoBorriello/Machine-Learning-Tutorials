### Linear Regression package

import numpy as np

# Normal equation
def best_fit_parameters(X,Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y) 

# Cost function
def objective_function(X,Y,theta):
    return (X.dot(theta)-Y).T.dot(X.dot(theta)-Y)[0][0]/2./len(Y)