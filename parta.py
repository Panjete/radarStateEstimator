import plotly.express as px
import plotly.graph_objects as go
import numpy as np


covar_init = np.diag(np.array([])) ## prior_belief



## For modelling the noise that gets added after each update step
def noise(mean_matrix, covariance_matrix):
    return np.random.multivariate_normal(mean_matrix, covariance_matrix).reshape(-1, 1)

def action_upd(X_t, A_t, B_t, u_t, mean_epsilon, R):
    # print("S1 = ", np.dot(A_t, X_t).shape)
    # print("S2 = ", np.dot(B_t, u_t).shape)
    # print("S3 = ", noise(mean_epsilon, R).shape)
    # print("S Sum = ", (np.dot(A_t, X_t) + np.dot(B_t, u_t) + noise(mean_epsilon, R)).shape)
    return np.dot(A_t, X_t) + np.dot(B_t, u_t) + noise(mean_epsilon, R)

def obsv_upd(X_t, C_t, mean_delta, Q):
    # print(X_t.shape)
    # print(C_t.shape)
    # print(np.dot(C_t, X_t).shape)
    return np.dot(C_t, X_t) + noise(mean_delta, Q)

def kalman_update(mu_tm1, sigma_tm1, u_t, z_t, A_t, B_t, C_t, Q_t, R_t):
    n_shape = mu_tm1.shape[0]
    C_t_transpose = np.transpose(C_t)
    A_t_transpose = np.transpose(A_t)
    mu_t_bar = np.dot(A_t, mu_tm1) + B_t*u_t
    sigma_t_bar = np.dot(np.dot(A_t, sigma_tm1), A_t_transpose) + R_t
    K_t = np.dot(np.dot(sigma_t_bar, C_t_transpose),np.linalg.inv(np.dot(np.dot(C_t, mu_t_bar),C_t_transpose)+Q_t))
    mu_t = mu_t_bar + np.dot(K_t, (z_t-np.dot(C_t, mu_t_bar)))
    sigma_t = np.dot((np.identity(n_shape)-np.dot(K_t, C_t)), sigma_t_bar)
    return mu_t, sigma_t

def part_a():
    u_t = np.array([[0], [0], [0]]) ## Control Inputs are all 0
    X_init = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]]) ## start from (0,0,0) with vel (1,1,1)
    observed_vals = np.zeros((300, 3))
    actual_vals = np.zeros((300, 3))
    print("ut shape = ",u_t.shape)
    print("xt shape = ", X_init.shape)
    deltaT = 1.0

    ## Action Model Parameters
    A_t = np.array([[1.0,0,0,deltaT,0,0],
                    [0,1.0,0,0,deltaT,0],
                    [0,0,1.0,0,0,deltaT],
                    [0,0,0,1.0,0,0],
                    [0,0,0,0,1.0,0],
                    [0,0,0,0,0,1.0]])

    B_t = np.array([[0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [1.0,0,0],
                    [0,1.0,0],
                    [0,0,1.0]])
    sigma_ri = 1.0
    sigma_ridot = 0.008
    mean_epsilon = np.zeros(6)
    R = np.diag(np.array([sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot]))

    ## Observation Model Parameters
    C_t = np.array([[1.0,0,0,0,0,0],
                    [0,1.0,0,0,0,0],
                    [0,0,1.0,0,0,0]])
    sigma_s = 8
    Q = sigma_s*sigma_s*np.identity(3)
    mean_delta = np.zeros(3)

    print("At shape = ", A_t.shape)
    print("B_t shape =", B_t.shape)
    print("C_t shape =", C_t.shape)
    print("R shape = ", R.shape)
    print("mean epsilon shape = ", mean_epsilon.shape)
    print("Qshape = ", Q.shape)
    print("mean_delta shape = ", mean_delta.shape)
    print("CT.Xt shape",  np.dot(C_t, X_init).shape)
    print("noise shape = ", noise(mean_delta, Q).shape)
    for i in range(300):
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        # print("Upd shape =", X_init.shape)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        observed_vals[i] = z_t.squeeze()
        actual_vals[i] = X_init[0:3].squeeze()

    # Create a 3D scatter plot for the first dataset
    fig = go.Figure(data=[go.Scatter3d(
        x=observed_vals[:, 0],
        y=observed_vals[:, 1],
        z=observed_vals[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Observed Trajectory'
    )])

    # Add points from the second dataset to the same plot
    fig.add_trace(go.Scatter3d(
        x=actual_vals[:, 0],
        y=actual_vals[:, 1],
        z=actual_vals[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory'
    ))

    # Update the layout if needed
    fig.update_layout(
        title='3D Line Plots of Actual Vs Observed Trajectories',
        scene=dict(aspectmode='data')
    )

    # Show the plot
    fig.show()

    return 0

# print("A_t = ", A_t)
# print("B_t = ", B_t)
# print("R = ", R)
# print("C_t = ", C_t)
# print("Q = ", Q)
# print("hello!")

part_a()