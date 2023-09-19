import numpy as np
from math import sqrt, sin, cos
import random
import plotly.graph_objects as go
import itertools

def noise(mean_matrix, covariance_matrix):
    return np.random.multivariate_normal(mean_matrix, covariance_matrix).reshape(-1, 1)

def action_upd(X_t, A_t, B_t, u_t, mean_epsilon, R):
    return np.dot(A_t, X_t) + np.dot(B_t, u_t) + noise(mean_epsilon, R)

def obsv_upd(X_t, C_t, mean_delta, Q):
    return np.dot(C_t, X_t) + noise(mean_delta, Q)


## Makes the concious choice of selecting apt sensor measurement for the right path

            


   
        
    



