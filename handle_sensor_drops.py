import plotly.graph_objects as go
import numpy as np
from math import cos, sin


## For modelling the noise that gets added after each update step
def noise(mean_matrix, covariance_matrix):
    return np.random.multivariate_normal(mean_matrix, covariance_matrix).reshape(-1, 1)

def action_upd(X_t, A_t, B_t, u_t, mean_epsilon, R):
    return np.dot(A_t, X_t) + np.dot(B_t, u_t) + noise(mean_epsilon, R)

def obsv_upd(X_t, C_t, mean_delta, Q):
    return np.dot(C_t, X_t) + noise(mean_delta, Q)

## Extract 100 points depaicting the uncertainity ellipses for a distribution
def uncer_ellipse_params(mu_t, sigma_t):
    xy = mu_t[0:2].reshape(-1, 1)
    xy_cov = sigma_t[:2, :2]
    eigen_values, eigen_vectors = np.linalg.eig(xy_cov)

    # Determine the major and minor axes lengths (standard deviations)
    std_dev_major = np.sqrt(eigen_values[0])
    std_dev_minor = np.sqrt(eigen_values[1])

    rotation_angle = np.arctan2(eigen_vectors[1, 0], eigen_vectors[0, 0])

    theta = np.linspace(0, 2 * np.pi, 100)

    x_rotated = mu_t[0] + (std_dev_major * np.cos(theta)) * np.cos(rotation_angle) - (std_dev_minor * np.sin(theta)) * np.sin(rotation_angle)
    y_rotated = mu_t[1] + (std_dev_major * np.cos(theta)) * np.sin(rotation_angle) + (std_dev_minor * np.sin(theta)) * np.cos(rotation_angle)

    return x_rotated, y_rotated

## Kalman Filter Update Equations
def kalman_update(mu_tm1, sigma_tm1, u_t, z_t, A_t, B_t, C_t, Q_t, R_t):
    n_shape = mu_tm1.shape[0]
    C_t_transpose = np.transpose(C_t)
    A_t_transpose = np.transpose(A_t)
    mu_t_bar = np.dot(A_t, mu_tm1) + np.dot(B_t, u_t)
    sigma_t_bar = np.dot(np.dot(A_t, sigma_tm1), A_t_transpose) + R_t
    K_t = np.dot(np.dot(sigma_t_bar, C_t_transpose),np.linalg.inv(np.dot(np.dot(C_t, sigma_t_bar),C_t_transpose)+Q_t))
    mu_t = mu_t_bar + np.dot(K_t, (z_t-np.dot(C_t, mu_t_bar)))
    sigma_t = np.dot((np.identity(n_shape)-np.dot(K_t, C_t)), sigma_t_bar)
    return mu_t, sigma_t

## To handle no measurements being recieved
def kalman_update_pure_evolution(mu_tm1, sigma_tm1, u_t, A_t, B_t, R_t):
    A_t_transpose = np.transpose(A_t)
    mu_t_bar = np.dot(A_t, mu_tm1) + np.dot(B_t, u_t)
    sigma_t_bar = np.dot(np.dot(A_t, sigma_tm1), A_t_transpose) + R_t
    return mu_tm1, sigma_tm1
    
## Measurements are not recieved for two periods within the simulation time
def all_sensors_drop_off():
    X_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    mu_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    sigma_init = (0.008)*(0.008)*np.identity(6)
    observed_traj = np.zeros((300, 3))
    actual_traj = np.zeros((300, 3))
    estimated_traj = np.zeros((300, 3))

    uncertainity_ellipse_values = np.zeros((300, 100, 2))

    ## Action Model Parameters
    deltaT = 1.0
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

    for i in range(300):
        tt_now = i
        u_t = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        if ((i < 50) or (i > 80 and i < 200) or (i >230)):
            mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)
        else:
            mu_init, sigma_init = kalman_update_pure_evolution(mu_init, sigma_init, u_t, A_t, B_t, R)

        observed_traj[i] = z_t.squeeze()
        actual_traj[i] = X_init[0:3].squeeze()
        estimated_traj[i] = mu_init[0:3].squeeze()

        x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        uncertainity_ellipse_values[i] = np.column_stack((x_uncert_points, y_uncert_points))
        

    fig = go.Figure(data=[go.Scatter3d(x=observed_traj[:, 0],y=observed_traj[:, 1],z=observed_traj[:, 2],mode='lines',name='Noisy Observations')])
    fig.add_trace(go.Scatter3d(x=actual_traj[:, 0],y=actual_traj[:, 1],z=actual_traj[:, 2],mode='lines',name='Actual Trajectory'))
    fig.add_trace(go.Scatter3d(x=estimated_traj[:, 0],y=estimated_traj[:, 1],z=estimated_traj[:, 2],mode='lines',name='Kalman Estimation'))
    fig.update_layout(title='3D Line Plots of Actual vs Estimated Trajectory, and Noisy Observations and Sensor Off Periods',scene=dict(aspectmode='data'))

    fig2 = go.Figure(data=[go.Scatter(x=estimated_traj[:, 0],y=estimated_traj[:, 1],mode='lines',name='Trajectory Points',)])
    for i in range(300):
        fig2.add_trace(go.Scatter(x=uncertainity_ellipse_values[i][:,0],y=uncertainity_ellipse_values[i][:,1],mode='lines',line=dict(color='red', width=1),showlegend=False))

    fig2.update_layout(title='Projection of Estimated Traj into X-Y plane',scene=dict(aspectmode='data'))
    
    fig.show()
    fig2.show()

    fig.write_html("allSensorOff.html")
    fig2.write_html("ue_allSensorsOff.html")
    return 0

def kalman_update_yz(mu_tm1, sigma_tm1, u_t, z_t_red, A_t, B_t, C_t_red, Q_t_red, R_t):
    n_shape = mu_tm1.shape[0]
    C_t_transpose = np.transpose(C_t_red)
    A_t_transpose = np.transpose(A_t)
    mu_t_bar = np.dot(A_t, mu_tm1) + np.dot(B_t, u_t)
    sigma_t_bar = np.dot(np.dot(A_t, sigma_tm1), A_t_transpose) + R_t
    K_t = np.dot(np.dot(sigma_t_bar, C_t_transpose),np.linalg.inv(np.dot(np.dot(C_t_red, sigma_t_bar),C_t_transpose)+Q_t_red))
    mu_t = mu_t_bar + np.dot(K_t, (z_t_red-np.dot(C_t_red, mu_t_bar)))
    sigma_t = np.dot((np.identity(n_shape)-np.dot(K_t, C_t_red)), sigma_t_bar)
    return mu_t, sigma_t

## Only the X sensor drops off
def sensor_X_drops():
    X_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    mu_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    sigma_init = (0.008)*(0.008)*np.identity(6)
    observed_traj = np.zeros((300, 3))
    actual_traj = np.zeros((300, 3))
    estimated_traj = np.zeros((300, 3))

    uncertainity_ellipse_values = np.zeros((300, 100, 2))

    ## Action Model Parameters
    deltaT = 1.0
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
    mean_delta_red = np.zeros(2)

    C_t_red = np.array([[0,1.0,0,0,0,0],
                        [0,0,1.0,0,0,0]])
    Q_red = sigma_s*sigma_s*np.identity(2)

    for i in range(300):
        tt_now = i 
        u_t = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)
        z_t_red = obsv_upd(X_init, C_t_red, mean_delta_red, Q_red)

        if (i < 100):
            mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)
        else:
            mu_init, sigma_init = kalman_update_yz(mu_init, sigma_init, u_t, z_t_red, A_t, B_t, C_t_red, Q_red, R)

        observed_traj[i] = z_t.squeeze()
        actual_traj[i] = X_init[0:3].squeeze()
        estimated_traj[i] = mu_init[0:3].squeeze()

        x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        uncertainity_ellipse_values[i] = np.column_stack((x_uncert_points, y_uncert_points))

    # Create a 3D scatter plot for the first dataset
    fig = go.Figure(data=[go.Scatter3d(x=observed_traj[:, 0],y=observed_traj[:, 1],z=observed_traj[:, 2],mode='lines',name='Noisy Observations')])

   
    fig.add_trace(go.Scatter3d(x=actual_traj[:, 0],y=actual_traj[:, 1],z=actual_traj[:, 2],mode='lines',name='Actual Trajectory'))
    fig.add_trace(go.Scatter3d(x=estimated_traj[:, 0],y=estimated_traj[:, 1],z=estimated_traj[:, 2],mode='lines',name='Kalman Estimation'))

    fig.update_layout(title='3D Line Plots of Actual vs Estimated Trajectory, and Noisy Observations and Sensor Off Periods',scene=dict(aspectmode='data'))

    fig2 = go.Figure(data=[go.Scatter(x=estimated_traj[:, 0],y=estimated_traj[:, 1],mode='markers',marker=dict(size=3),name='Trajectory Points',)])

    for i in range(300):
        fig2.add_trace(go.Scatter(x=uncertainity_ellipse_values[i][:,0],y=uncertainity_ellipse_values[i][:,1],mode='lines',line=dict(color='red', width=1),showlegend=False))

    fig2.update_layout(title='Projection of Estimated Traj into X-Y plane',scene=dict(aspectmode='data'))
    
    fig.show()
    fig2.show()

    fig.write_html("XsensorOff.html")
    fig2.write_html("ue_XsensorOff.html")
    return 0