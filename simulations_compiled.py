import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from math import cos, sin, sqrt
import random
import itertools


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

def part_a():
    u_t = np.array([[0], [0], [0]]) ## Control Inputs are all 0
    X_init = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]]) ## start from (0,0,0) with vel (1,1,1)
    observed_vals = np.zeros((300, 3))
    actual_vals = np.zeros((300, 3))
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

    for i in range(300):
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        observed_vals[i] = z_t.squeeze()
        actual_vals[i] = X_init[0:3].squeeze()

    fig = go.Figure(data=[go.Scatter3d(
        x=observed_vals[:, 0],
        y=observed_vals[:, 1],
        z=observed_vals[:, 2],
        mode='lines',
        name='Observed Trajectory'
    )])

    fig.add_trace(go.Scatter3d(
        x=actual_vals[:, 0],
        y=actual_vals[:, 1],
        z=actual_vals[:, 2],
        mode='lines',
        name='Actual Trajectory'
    ))

    fig.update_layout(
        title='3D Line Plots of Actual Vs Observed Trajectories',
        scene=dict(aspectmode='data')
    )

    fig.show()
    return 0

## Part B solved for a general Kalman Filter Update
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

## initialise parameters, estimate and plot
def part_c():
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
        tt_now = i * 1.0 ## Update this to a different timestamp if alternate readings reqd
        u_t = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)

        observed_traj[i] = z_t.squeeze()
        actual_traj[i] = X_init[0:3].squeeze()
        estimated_traj[i] = mu_init[0:3].squeeze()

        x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        uncertainity_ellipse_values[i] = np.column_stack((x_uncert_points, y_uncert_points))

    fig = go.Figure(data=[go.Scatter3d(
        x=observed_traj[:, 0],
        y=observed_traj[:, 1],
        z=observed_traj[:, 2],
        mode='lines',
        name='Noisy Observations'
    )])

    fig.add_trace(go.Scatter3d(
        x=actual_traj[:, 0],
        y=actual_traj[:, 1],
        z=actual_traj[:, 2],
        mode='lines',
        name='Actual Trajectory'
    ))

    fig.add_trace(go.Scatter3d(
        x=estimated_traj[:, 0],
        y=estimated_traj[:, 1],
        z=estimated_traj[:, 2],
        mode='lines',
        name='Kalman Estimation'
    ))

    fig.update_layout(
        title='3D Line Plots of Actual vs Estimated Trajectory, and Noisy Observations',
        scene=dict(aspectmode='data')
    )

    fig2 = go.Figure(data=[go.Scatter(
        x=estimated_traj[:, 0],
        y=estimated_traj[:, 1],
        mode='lines',
        name='Trajectory Points',
    )])

    for i in range(300):
        fig2.add_trace(go.Scatter(
            x=uncertainity_ellipse_values[i][:,0],
            y=uncertainity_ellipse_values[i][:,1],
            mode='lines',
            line=dict(color='red', width=1),
            showlegend=False
        ))

    fig2.update_layout(
        title='Projection of Estimated Traj into X-Y plane',
        scene=dict(aspectmode='data')
    )

    fig.show()
    fig2.show()

    fig.write_html("part1c_trajs.html")
    fig2.write_html("part1c_ue.html")
    return 0

def part_d():
    observed_traj_normal = np.zeros((300, 3))
    actual_traj_normal = np.zeros((300, 3))
    estimated_traj_normal = np.zeros((300, 3))

    observed_traj_pos_noise_enhanced = np.zeros((300, 3))
    actual_traj_pos_noise_enhanced = np.zeros((300, 3))
    estimated_traj_pos_noise_enhanced = np.zeros((300, 3))
    
    observed_traj_vel_noise_enhanced = np.zeros((300, 3))
    actual_traj_vel_noise_enhanced = np.zeros((300, 3))
    estimated_traj_vel_noise_enhanced = np.zeros((300, 3))

    observed_traj_sensor_noise_enhanced = np.zeros((300, 3))
    actual_traj_sensor_noise_enhanced = np.zeros((300, 3))
    estimated_traj_sensor_noise_enhanced = np.zeros((300, 3))
    
    uncertainity_ellipse_values_normal = np.zeros((300, 100, 2))
    uncertainity_ellipse_values_pos_noise_enhanced = np.zeros((300, 100, 2))
    uncertainity_ellipse_values_vel_noise_enhanced = np.zeros((300, 100, 2))
    uncertainity_ellipse_values_sensor_noise_enhanced = np.zeros((300, 100, 2))

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
    

    ## Observation Model Parameters
    C_t = np.array([[1.0,0,0,0,0,0],
                    [0,1.0,0,0,0,0],
                    [0,0,1.0,0,0,0]])
    
    ##### Normal case 1/4
    X_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    mu_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    sigma_init = (0.008)*(0.008)*np.identity(6)
    sigma_ri = 1.0
    sigma_ridot = 0.008
    mean_epsilon = np.zeros(6)
    R = np.diag(np.array([sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot]))
    sigma_s = 8
    Q = sigma_s*sigma_s*np.identity(3)
    mean_delta = np.zeros(3)

    for i in range(300):
        tt_now = i
        u_t = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)

        observed_traj_normal[i] = z_t.squeeze()
        actual_traj_normal[i] = X_init[0:3].squeeze()
        estimated_traj_normal[i] = mu_init[0:3].squeeze()

        x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        uncertainity_ellipse_values_normal[i] = np.column_stack((x_uncert_points, y_uncert_points))

    ##### Position Noise Enhanced 2/4
    X_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    mu_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    sigma_init = (0.008)*(0.008)*np.identity(6)
    sigma_ri = 3.0
    sigma_ridot = 0.008
    mean_epsilon = np.zeros(6)
    R = np.diag(np.array([sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot]))
    sigma_s = 8
    Q = sigma_s*sigma_s*np.identity(3)
    mean_delta = np.zeros(3)

    for i in range(300):
        tt_now = i 
        u_t = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)

        observed_traj_pos_noise_enhanced[i] = z_t.squeeze()
        actual_traj_pos_noise_enhanced[i] = X_init[0:3].squeeze()
        estimated_traj_pos_noise_enhanced[i] = mu_init[0:3].squeeze()

        x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        uncertainity_ellipse_values_pos_noise_enhanced[i] = np.column_stack((x_uncert_points, y_uncert_points))

    ##### Velocity Noise Enhanced 3/4
    sigma_ri = 1.0
    sigma_ridot = 0.012
    mean_epsilon = np.zeros(6)
    R = np.diag(np.array([sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot]))
    sigma_s = 8
    Q = sigma_s*sigma_s*np.identity(3)
    mean_delta = np.zeros(3)
    X_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    mu_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    sigma_init = (0.008)*(0.008)*np.identity(6)

    for i in range(300):
        tt_now = i
        u_t = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        # print("Upd shape =", X_init.shape)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)

        observed_traj_vel_noise_enhanced[i] = z_t.squeeze()
        actual_traj_vel_noise_enhanced[i] = X_init[0:3].squeeze()
        estimated_traj_vel_noise_enhanced[i] = mu_init[0:3].squeeze()

        x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        uncertainity_ellipse_values_vel_noise_enhanced[i] = np.column_stack((x_uncert_points, y_uncert_points))

    ##### Sensor Noise enhanced 4/4
    sigma_ri = 1.0
    sigma_ridot = 0.008
    mean_epsilon = np.zeros(6)
    R = np.diag(np.array([sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot]))
    sigma_s = 16
    Q = sigma_s*sigma_s*np.identity(3)
    mean_delta = np.zeros(3)
    X_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    mu_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    sigma_init = (0.008)*(0.008)*np.identity(6)

    for i in range(300):
        tt_now = i
        u_t = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)

        observed_traj_sensor_noise_enhanced[i] = z_t.squeeze()
        actual_traj_sensor_noise_enhanced[i] = X_init[0:3].squeeze()
        estimated_traj_sensor_noise_enhanced[i] = mu_init[0:3].squeeze()

        x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        uncertainity_ellipse_values_sensor_noise_enhanced[i] = np.column_stack((x_uncert_points, y_uncert_points))

    fig1 = go.Figure(data = [go.Scatter3d(x=observed_traj_normal[:, 0],y=observed_traj_normal[:, 1],z=observed_traj_normal[:, 2],mode='lines',name='Noisy Observations Normal')])
    fig1.add_trace(go.Scatter3d(x=actual_traj_normal[:, 0],y=actual_traj_normal[:, 1],z=actual_traj_normal[:, 2],mode='lines',name='Actual Trajectory Normal'))
    fig1.add_trace(go.Scatter3d(x=estimated_traj_normal[:, 0],y=estimated_traj_normal[:, 1],z=estimated_traj_normal[:, 2],mode='lines',name='Kalman Estimation Normal'))
    fig1.add_trace(go.Scatter3d(x=observed_traj_pos_noise_enhanced[:, 0],y=observed_traj_pos_noise_enhanced[:, 1],z=observed_traj_pos_noise_enhanced[:, 2],mode='lines',name='Noisy Observations Position Noise Enhanced'))
    fig1.add_trace(go.Scatter3d(x=actual_traj_pos_noise_enhanced[:, 0],y=actual_traj_pos_noise_enhanced[:, 1],z=actual_traj_pos_noise_enhanced[:, 2],mode='lines',name='Actual Trajectory Position Noise Enhanced'))
    fig1.add_trace(go.Scatter3d(x=estimated_traj_pos_noise_enhanced[:, 0],y=estimated_traj_pos_noise_enhanced[:, 1],z=estimated_traj_pos_noise_enhanced[:, 2],mode='lines',name='Kalman Estimation Position Noise Enhanced'))

    fig2 = go.Figure(data=[go.Scatter3d(x=observed_traj_normal[:, 0],y=observed_traj_normal[:, 1],z=observed_traj_normal[:, 2],mode='lines', name='Noisy Observations Normal')])
    fig2.add_trace(go.Scatter3d(x=actual_traj_normal[:, 0],y=actual_traj_normal[:, 1],z=actual_traj_normal[:, 2],mode='lines',name='Actual Trajectory Normal'))
    fig2.add_trace(go.Scatter3d(x=estimated_traj_normal[:, 0],y=estimated_traj_normal[:, 1],z=estimated_traj_normal[:, 2],mode='lines',name='Kalman Estimation Normal'))
    fig2.add_trace(go.Scatter3d(x=observed_traj_vel_noise_enhanced[:, 0],y=observed_traj_vel_noise_enhanced[:, 1],z=observed_traj_vel_noise_enhanced[:, 2],mode='lines',name='Noisy Observations Velocity Noise Enhanced'))
    fig2.add_trace(go.Scatter3d(x=actual_traj_vel_noise_enhanced[:, 0],y=actual_traj_vel_noise_enhanced[:, 1],z=actual_traj_vel_noise_enhanced[:, 2],mode='lines',name='Actual Trajectory Velocity Noise Enhanced'))
    fig2.add_trace(go.Scatter3d(x=estimated_traj_vel_noise_enhanced[:, 0],y=estimated_traj_vel_noise_enhanced[:, 1],z=estimated_traj_vel_noise_enhanced[:, 2],mode='lines',name='Kalman Estimation Velocity Noise Enhanced'))
    
    fig3 = go.Figure(data=[go.Scatter3d(x=observed_traj_normal[:, 0],y=observed_traj_normal[:, 1],z=observed_traj_normal[:, 2],mode='lines',name='Noisy Observations Normal')])
    fig3.add_trace(go.Scatter3d(x=actual_traj_normal[:, 0],y=actual_traj_normal[:, 1],z=actual_traj_normal[:, 2],mode='lines',name='Actual Trajectory Normal'))
    fig3.add_trace(go.Scatter3d(x=estimated_traj_normal[:, 0],y=estimated_traj_normal[:, 1],z=estimated_traj_normal[:, 2],mode='lines',name='Kalman Estimation Normal'))
    fig3.add_trace(go.Scatter3d(x=observed_traj_sensor_noise_enhanced[:, 0],y=observed_traj_sensor_noise_enhanced[:, 1],z=observed_traj_sensor_noise_enhanced[:, 2],mode='lines',name='Noisy Observations Sensor Noise Enhanced'))
    fig3.add_trace(go.Scatter3d(x=actual_traj_sensor_noise_enhanced[:, 0],y=actual_traj_sensor_noise_enhanced[:, 1],z=actual_traj_sensor_noise_enhanced[:, 2],mode='lines',name='Actual Trajectory Sensor Noise Enhanced'))
    fig3.add_trace(go.Scatter3d(x=estimated_traj_sensor_noise_enhanced[:, 0],y=estimated_traj_sensor_noise_enhanced[:, 1],z=estimated_traj_sensor_noise_enhanced[:, 2], mode='lines',name='Kalman Estimation Sensor Noise Enhanced'))

    fig4 = go.Figure(data=[go.Scatter(x=estimated_traj_normal[:, 0],y=estimated_traj_normal[:, 1],mode='lines',name='Trajectory Points Normal')])
    fig4.add_trace(go.Scatter(x=estimated_traj_pos_noise_enhanced[:, 0],y=estimated_traj_pos_noise_enhanced[:, 1],mode='lines',name='Trajectory Points with Position Noise'))
    fig4.add_trace(go.Scatter(x=estimated_traj_vel_noise_enhanced[:, 0],y=estimated_traj_vel_noise_enhanced[:, 1],mode='lines',name='Trajectory Points with Velocity Noise'))
    fig4.add_trace(go.Scatter(x=estimated_traj_sensor_noise_enhanced[:, 0],y=estimated_traj_sensor_noise_enhanced[:, 1],mode='lines',name='Trajectory Points with Sensor Noise'))

    for i in range(0, 300): # do range(0,300, 5) for selective ellipses
        fig4.add_trace(go.Scatter(x=uncertainity_ellipse_values_normal[i][:,0],y=uncertainity_ellipse_values_normal[i][:,1],mode='lines', line=dict(color='red', width=1),showlegend=False))
        fig4.add_trace(go.Scatter(x=uncertainity_ellipse_values_pos_noise_enhanced[i][:,0],y=uncertainity_ellipse_values_pos_noise_enhanced[i][:,1],mode='lines', line=dict(color='red', width=1),showlegend=False))
        fig4.add_trace(go.Scatter(x=uncertainity_ellipse_values_vel_noise_enhanced[i][:,0],y=uncertainity_ellipse_values_vel_noise_enhanced[i][:,1],mode='lines', line=dict(color='red', width=1),showlegend=False))
        fig4.add_trace(go.Scatter(x=uncertainity_ellipse_values_sensor_noise_enhanced[i][:,0],y=uncertainity_ellipse_values_sensor_noise_enhanced[i][:,1],mode='lines', line=dict(color='red', width=1),showlegend=False))

    fig4.update_layout(
        title='Uncertainity Ellipses and Projection',
        scene=dict(aspectmode='data')
    )

    fig5 = go.Figure(data = [go.Scatter3d(x=actual_traj_normal[:, 0],y=actual_traj_normal[:, 1],z=actual_traj_normal[:, 2],mode='lines',name='Actual Trajectory Normal')])
    fig5.add_trace(go.Scatter3d(x=actual_traj_pos_noise_enhanced[:, 0],y=actual_traj_pos_noise_enhanced[:, 1],z=actual_traj_pos_noise_enhanced[:, 2],mode='lines',name='Actual Trajectory Pos Noise'))
    fig5.add_trace(go.Scatter3d(x=actual_traj_vel_noise_enhanced[:, 0],y=actual_traj_vel_noise_enhanced[:, 1],z=actual_traj_vel_noise_enhanced[:, 2],mode='lines',name='Actual Trajectory Vel Noise'))
    fig5.add_trace(go.Scatter3d(x=actual_traj_sensor_noise_enhanced[:, 0],y=actual_traj_sensor_noise_enhanced[:, 1],z=actual_traj_sensor_noise_enhanced[:, 2],mode='lines',name='Actual Trajectory Sensor Noise'))

    fig5.add_trace(go.Scatter3d(x=estimated_traj_normal[:, 0],y=estimated_traj_normal[:, 1],z=estimated_traj_normal[:, 2],mode='lines',name='Estimated Trajectory Normal'))
    fig5.add_trace(go.Scatter3d(x=estimated_traj_pos_noise_enhanced[:, 0],y=estimated_traj_pos_noise_enhanced[:, 1],z=estimated_traj_pos_noise_enhanced[:, 2],mode='lines',name='Estimated Trajectory Pos Noise'))
    fig5.add_trace(go.Scatter3d(x=estimated_traj_vel_noise_enhanced[:, 0],y=estimated_traj_vel_noise_enhanced[:, 1],z=estimated_traj_vel_noise_enhanced[:, 2],mode='lines',name='Estimated Trajectory Vel Noise'))
    fig5.add_trace(go.Scatter3d(x=estimated_traj_sensor_noise_enhanced[:, 0],y=estimated_traj_sensor_noise_enhanced[:, 1],z=estimated_traj_sensor_noise_enhanced[:, 2],mode='lines',name='Estimated Trajectory Sensor Noise'))

    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()
    fig5.show()

    fig1.write_html("part1d_posNoiseComp.html")
    fig2.write_html("part1d_velNoiseComp.html")
    fig3.write_html("part1d_sensorNoiseComp.html")
    fig4.write_html("part1d_ue.html")
    fig5.write_html("part1d_allTrajs.html")
    return 0


def kalman_update_pure_evolution(mu_tm1, sigma_tm1, u_t, A_t, B_t, R_t):
    A_t_transpose = np.transpose(A_t)
    mu_t_bar = np.dot(A_t, mu_tm1) + np.dot(B_t, u_t)
    sigma_t_bar = np.dot(np.dot(A_t, sigma_tm1), A_t_transpose) + R_t
    return mu_tm1, sigma_tm1
    #return mu_t_bar, sigma_t_bar

def part_e():
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
        

    # Create a 3D scatter plot for the first dataset
    fig = go.Figure(data=[go.Scatter3d(
        x=observed_traj[:, 0],
        y=observed_traj[:, 1],
        z=observed_traj[:, 2],
        mode='lines',
        name='Noisy Observations'
    )])

    # Add points from the second dataset to the same plot
    fig.add_trace(go.Scatter3d(
        x=actual_traj[:, 0],
        y=actual_traj[:, 1],
        z=actual_traj[:, 2],
        mode='lines',
        name='Actual Trajectory'
    ))

    fig.add_trace(go.Scatter3d(
        x=estimated_traj[:, 0],
        y=estimated_traj[:, 1],
        z=estimated_traj[:, 2],
        mode='lines',
        name='Kalman Estimation'
    ))

    fig.update_layout(
        title='3D Line Plots of Actual vs Estimated Trajectory, and Noisy Observations and Sensor Off Periods',
        scene=dict(aspectmode='data')
    )

    fig2 = go.Figure(data=[go.Scatter(
        x=estimated_traj[:, 0],
        y=estimated_traj[:, 1],
        mode='lines',
        name='Trajectory Points',
    )])

    for i in range(300):
        fig2.add_trace(go.Scatter(x=uncertainity_ellipse_values[i][:,0],y=uncertainity_ellipse_values[i][:,1],mode='lines',line=dict(color='red', width=1),showlegend=False))

    fig2.update_layout(
        title='Projection of Estimated Traj into X-Y plane',
        scene=dict(aspectmode='data')
    )
    
    fig.show()
    fig2.show()

    fig.write_html("part1e_SensorOff.html")
    fig2.write_html("part1e_ue.html")
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

def part_f():
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
        # print("Upd shape =", X_init.shape)
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
    fig = go.Figure(data=[go.Scatter3d(
        x=observed_traj[:, 0],
        y=observed_traj[:, 1],
        z=observed_traj[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Noisy Observations'
    )])

    # Add points from the second dataset to the same plot
    fig.add_trace(go.Scatter3d(
        x=actual_traj[:, 0],
        y=actual_traj[:, 1],
        z=actual_traj[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory'
    ))

    fig.add_trace(go.Scatter3d(
        x=estimated_traj[:, 0],
        y=estimated_traj[:, 1],
        z=estimated_traj[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Kalman Estimation'
    ))

    # Update the layout if needed
    fig.update_layout(
        title='3D Line Plots of Actual vs Estimated Trajectory, and Noisy Observations and Sensor Off Periods',
        scene=dict(aspectmode='data')
    )

    fig2 = go.Figure(data=[go.Scatter(
        x=estimated_traj[:, 0],
        y=estimated_traj[:, 1],
        mode='markers',
        marker=dict(size=3),
        name='Trajectory Points',
    )])

    for i in range(300):
        fig2.add_trace(go.Scatter(x=uncertainity_ellipse_values[i][:,0],y=uncertainity_ellipse_values[i][:,1],mode='lines',line=dict(color='red', width=1),showlegend=False))

    fig2.update_layout(
        title='Projection of Estimated Traj into X-Y plane',
        scene=dict(aspectmode='data')
    )
    
    # Show the plot
    #fig.show()
    fig.show()
    fig2.show()

    fig.write_html("part1f_XsensorOff.html")
    fig2.write_html("part1f_ue.html")
    return 0

def part_g():
    X_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    mu_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    sigma_init = (0.008)*(0.008)*np.identity(6)

    actual_vel = np.zeros((300, 3))
    estimated_vel = np.zeros((300, 3))

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

        mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)

        actual_vel[i] = X_init[3:6].squeeze()
        estimated_vel[i] = mu_init[3:6].squeeze()

    # Create a 3D scatter plot for the first dataset
    fig = go.Figure(data=[go.Scatter3d(
        x=actual_vel[:, 0],
        y=actual_vel[:, 1],
        z=actual_vel[:, 2],
        mode='lines',
        name='Actual Velocity Values'
    )])

    # Add points from the second dataset to the same plot
    fig.add_trace(go.Scatter3d(
        x=estimated_vel[:, 0],
        y=estimated_vel[:, 1],
        z=estimated_vel[:, 2],
        mode='lines',
        name='Estimated Velocity Velocity'
    ))

    # Update the layout if needed
    fig.update_layout(
        title='3D Line Plots of Actual vs Estimated Velocity Points',
        scene=dict(aspectmode='data')
    )

    fig.show()
    fig.write_html("part1g_velPlots.html")
    return 0


## Data Association Problem starts
def kalman_update_evolve(mu_tm1, sigma_tm1, u_t, A_t, B_t, R_t):
    A_t_transpose = np.transpose(A_t)
    mu_t_bar = np.dot(A_t, mu_tm1) + np.dot(B_t, u_t)
    sigma_t_bar = np.dot(np.dot(A_t, sigma_tm1), A_t_transpose) + R_t
    return mu_t_bar, sigma_t_bar

def kalman_update_correct(mu_t_bar, sigma_t_bar,  z_t, C_t, Q_t):
    n_shape = mu_t_bar.shape[0]
    C_t_transpose = np.transpose(C_t)
    K_t = np.dot(np.dot(sigma_t_bar, C_t_transpose),np.linalg.inv(np.dot(np.dot(C_t, sigma_t_bar),C_t_transpose)+Q_t))
    mu_t = mu_t_bar + np.dot(K_t, (z_t-np.dot(C_t, mu_t_bar)))
    sigma_t = np.dot((np.identity(n_shape)-np.dot(K_t, C_t)), sigma_t_bar)
    return mu_t, sigma_t

def euclidean_distance(point, state):
    return np.linalg.norm(point - state[0:3].squeeze())

def manhattan_distance(point, state):
    return np.sum(np.abs(point - state[0:3].squeeze()))

def mahalanobis_distance(point, mean_state, cov_state):
    cov_rel = cov_state[0:3, 0:3]
    mean_state_rel = mean_state[0:3].reshape(-1,1)
    dist_err = point - mean_state_rel
    dis_err_transpose = np.transpose(dist_err)
    return sqrt(np.dot(np.dot(dis_err_transpose, np.linalg.inv(cov_rel)), dist_err))

def choose_measurements(sensor_measures, mu_tm1, mu_tm2, sigma_tm1, sigma_tm2, u1_t, u2_t, A, B, C, R1, R2, Q1, Q2):
    mu_tb1, sigma_tb1 = kalman_update_evolve(mu_tm1, sigma_tm1, u1_t, A, B, R1)
    mu_tb2, sigma_tb2 = kalman_update_evolve(mu_tm2, sigma_tm2, u2_t, A, B, R2)

    if(len(sensor_measures) == 0):
        return mu_tb1, sigma_tb1, mu_tb2, sigma_tb2
    elif(len(sensor_measures) == 1):
        meas_we_have = sensor_measures[0]
        dist_1 = euclidean_distance(meas_we_have, mu_tb1)
        dist_2 = euclidean_distance(meas_we_have, mu_tb2)
        if(dist_1 < dist_2):
            mu1_t, sigma1_t = kalman_update_correct(mu_tb1, sigma_tb1, meas_we_have, C, Q1)
            return mu1_t, sigma1_t, mu_tb2, sigma_tb2
        else:
            mu2_t, sigma2_t = kalman_update_correct(mu_tb2, sigma_tb2, meas_we_have, C, Q2)
            return mu_tb1, sigma_tb1, mu2_t, sigma2_t
        
    else:
        ma = sensor_measures[0]
        mb = sensor_measures[1]

        dist_1_a = euclidean_distance(ma, mu_tb1)
        dist_1_b = euclidean_distance(mb, mu_tb1)
        dist_2_a = euclidean_distance(ma, mu_tb2)
        dist_2_b = euclidean_distance(mb, mu_tb2)

        if(dist_1_a < dist_1_b and dist_2_b < dist_2_a):
            # point a with 1, b with 2
            mu1_t, sigma1_t = kalman_update_correct(mu_tb1, sigma_tb1, ma, C, Q1)
            mu2_t, sigma2_t = kalman_update_correct(mu_tb2, sigma_tb2, mb, C, Q2)
            return mu1_t, sigma1_t, mu2_t, sigma2_t
        
        elif(dist_1_b < dist_1_a and dist_2_a < dist_2_b):
            # b with 1 , a with 2
            mu1_t, sigma1_t = kalman_update_correct(mu_tb1, sigma_tb1, mb, C, Q1)
            mu2_t, sigma2_t = kalman_update_correct(mu_tb2, sigma_tb2, ma, C, Q2)
            return mu1_t, sigma1_t, mu2_t, sigma2_t

        elif(dist_1_a < dist_1_b and dist_2_a < dist_2_b):
            # a is closer to both 1 and 2 than b
            # now compare dist 1 and 2 with a
            # allot closer to a, and farther to b

            if(dist_1_a < dist_2_a):
                # 1 goes with a, 2 goes with b
                mu1_t, sigma1_t = kalman_update_correct(mu_tb1, sigma_tb1, ma, C, Q1)
                mu2_t, sigma2_t = kalman_update_correct(mu_tb2, sigma_tb2, mb, C, Q2)
                return mu1_t, sigma1_t, mu2_t, sigma2_t
            else:
                # 2 goes with a, 1 goes with b
                mu1_t, sigma1_t = kalman_update_correct(mu_tb1, sigma_tb1, mb, C, Q1)
                mu2_t, sigma2_t = kalman_update_correct(mu_tb2, sigma_tb2, ma, C, Q2)
                return mu1_t, sigma1_t, mu2_t, sigma2_t
            
        else:
            # b is closer to both 1 and 2 than a
            # now compare dist 1 and 2 with b
            # allot closer to b, and farther to a
            if(dist_1_b < dist_2_b):
                # 1 goes with b, 2 goes with a
                mu1_t, sigma1_t = kalman_update_correct(mu_tb1, sigma_tb1, mb, C, Q1)
                mu2_t, sigma2_t = kalman_update_correct(mu_tb2, sigma_tb2, ma, C, Q2)
                return mu1_t, sigma1_t, mu2_t, sigma2_t
            
            else:
                # 2 goes with b, 1 goes with a
                mu1_t, sigma1_t = kalman_update_correct(mu_tb1, sigma_tb1, ma, C, Q1)
                mu2_t, sigma2_t = kalman_update_correct(mu_tb2, sigma_tb2, mb, C, Q2)
                return mu1_t, sigma1_t, mu2_t, sigma2_t
## Makes the concious choice of selecting apt sensor measurement for the right path, for multiple sensors
def choose_measurements_multiple(sensor_measures, mus, sigmas, u_ts, A, B, C, Rs, Qs):
    mu_tb_s = []
    sigma_tb_s = []
    for i in range(len(mus)):
        mu_iiii, sigma_iiii = kalman_update_evolve(mus[i], sigmas[i], u_ts[i], A, B, Rs[i])
        mu_tb_s.append(mu_iiii)
        sigma_tb_s.append(sigma_iiii)
    
    ## Perm approach
    all_permutations = list(itertools.permutations(sensor_measures))
    #print(len(all_permutations))

    min_dis = float('inf')
    min_dis_perm = all_permutations[0] #Rand initialise
    for perm in all_permutations:
        curprod = 1.0
        for j in range(len(mus)):
            curprod *= mahalanobis_distance(perm[j], mus[j], sigmas[j])/100.0
            #print("M d =", mahalanobis_distance(perm[j], mus[j], sigmas[j]))
        #print("current currprod = ", curprod)
        if(curprod < min_dis):
                min_dis_perm = perm
                min_dis = curprod
    
    mus_t = []
    sigmas_t = []
    for i in range(len(mus)):
        mus_iiiiii, sigma_iiiiii = kalman_update_correct(mu_tb_s[i], sigma_tb_s[i], min_dis_perm[i], C, Qs[i])
        mus_t.append(mus_iiiiii)
        sigmas_t.append(sigma_iiiiii)

    ## Greedy approach working code, an approximation
    # set_sensor_measure = []
    # for sm in sensor_measures:
    #     set_sensor_measure.append(sm)

    # measures_ordered = []
    # for j in range(len(mus)):
    #     min_dis = float('inf')
    #     min_dis_at = sensor_measures[0] #Rand initialise
    #     min_dist_ind = 0
    #     for i in range(len(set_sensor_measure)):
    #         sm = set_sensor_measure[i]
    #         if(mahalanobis_distance(sm, mus[j], sigmas[j]) < min_dis):
    #             min_dis_at = sm
    #             min_dist_ind = i
    #             min_dis = mahalanobis_distance(sm, mus[j], sigmas[j])
        
    #     measures_ordered.append(min_dis_at)

    #     sm_copy = []
    #     for i in range(len(set_sensor_measure)):
    #         if (i!= min_dist_ind):
    #             sm_copy.append(set_sensor_measure[i])
        
    #     set_sensor_measure = sm_copy
    #     #print(mus[j], "chooses point = ", min_dis_at)

    # mus_t = []
    # sigmas_t = []
    # for i in range(len(mus)):
    #     mus_iiiiii, sigma_iiiiii = kalman_update_correct(mu_tb_s[i], sigma_tb_s[i], measures_ordered[i], C, Qs[i])
    #     mus_t.append(mus_iiiiii)
    #     sigmas_t.append(sigma_iiiiii)
    

    return mus_t, sigmas_t
    
def part_h():
    X_init_a = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    mu_init_a = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)

    X_init_b = np.array([[100.0], [100.0], [100.0], [0.0], [0.0], [0.0]]) ## start from (100,100,100) with vel (0,0,0)
    mu_init_b = np.array([[100.0], [100.0], [100.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)

    sigma_init_a = (0.008)*(0.008)*np.identity(6)
    sigma_init_b = (0.08)*(0.008)*np.identity(6)

    actual_traj_a = np.zeros((300, 3))
    estimated_traj_a = np.zeros((300, 3))

    actual_traj_b = np.zeros((300, 3))
    estimated_traj_b = np.zeros((300, 3))
    
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
    R1 = np.diag(np.array([sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot]))
    R2 = np.diag(np.array([4*sigma_ri*sigma_ri, 8*sigma_ri*sigma_ri, sigma_ri*sigma_ri, 7*sigma_ridot*sigma_ridot,9*sigma_ridot*sigma_ridot,2*sigma_ridot*sigma_ridot]))
    ## Observation Model Parameters
    C_t = np.array([[1.0,0,0,0,0,0],
                    [0,1.0,0,0,0,0],
                    [0,0,1.0,0,0,0]])
    sigma_s = 8
    Q1 = sigma_s*sigma_s*np.identity(3)
    Q2 = sigma_s*sigma_s*np.identity(3)
    mean_delta = np.zeros(3)

    for i in range(300):
        tt_now = i * deltaT *0.1
        u_t_a = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        u_t_b = np.array([[sin(tt_now)], [cos(tt_now)], [cos(tt_now)]]) ## Control Inputs are sinusoidal functions

        X_init_a = action_upd(X_init_a, A_t, B_t, u_t_a, mean_epsilon, R1)
        X_init_b = action_upd(X_init_b, A_t, B_t, u_t_b, mean_epsilon, R2)

        # print("Upd shape =", X_init.shape)
        z1_t = obsv_upd(X_init_a, C_t, mean_delta, Q1)
        z2_t = obsv_upd(X_init_b, C_t, mean_delta, Q2)

        boo = random.choice([True, False])
        if boo:
            sensors = [z1_t, z2_t]
        else:
            sensors = [z2_t, z1_t]
        
        mu_init_a, sigma_init_a, mu_init_b, sigma_init_b = choose_measurements(sensors, mu_init_a, mu_init_b, sigma_init_a, sigma_init_b, u_t_a, u_t_b, A_t, B_t, C_t, R1, R2, Q1, Q2)

        actual_traj_a[i] = X_init_a[0:3].squeeze()
        actual_traj_b[i] = X_init_b[0:3].squeeze()

        estimated_traj_a[i] = mu_init_a[0:3].squeeze()
        estimated_traj_b[i] = mu_init_b[0:3].squeeze()

    fig = go.Figure(data=[go.Scatter3d(
        x=actual_traj_a[:, 0],
        y=actual_traj_a[:, 1],
        z=actual_traj_a[:, 2],
        mode='lines',
        name='Actual Trajectory AeroPlane A'
    )])

    fig.add_trace(go.Scatter3d(
        x=actual_traj_b[:, 0],
        y=actual_traj_b[:, 1],
        z=actual_traj_b[:, 2],
        mode='lines',
        name='Actual Trajectory AeroPlane B'
    ))

    fig.add_trace(go.Scatter3d(
        x=estimated_traj_a[:, 0],
        y=estimated_traj_a[:, 1],
        z=estimated_traj_a[:, 2],
        mode='lines',
        name='Kalman Estimation of AeroPlane A'
    ))

    fig.add_trace(go.Scatter3d(
        x=estimated_traj_b[:, 0],
        y=estimated_traj_b[:, 1],
        z=estimated_traj_b[:, 2],
        mode='lines',
        name='Kalman Estimation of AeroPlane B'
    ))

    fig.update_layout(
        title='3D Line Plots of Actual, Estimated Trajectory with Data Association',
        scene=dict(aspectmode='data')
    )

    fig.show()
    fig.write_html("part1h_landmarks.html")
    return 0

def part_i():
    X_init_a = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    mu_init_a = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)

    X_init_b = np.array([[1000.0], [1000.0], [1000.0], [0.0], [0.0], [0.0]]) ## start from (100,100,100) with vel (0,0,0)
    mu_init_b = np.array([[1000.0], [1000.0], [1000.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)

    X_init_c = np.array([[1000.0], [1000.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (100,100,0) with vel (0,0,0)
    mu_init_c = np.array([[1000.0], [1000.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (100,100,0) with vel (0,0,0)

    X_init_d = np.array([[500.0], [500.0], [500.0], [0.0], [0.0], [0.0]]) ## start from (50,50,50) with vel (0,0,0)
    mu_init_d = np.array([[500.0], [500.0], [500.0], [0.0], [0.0], [0.0]]) ## start from (50,50,50) with vel (0,0,0)

    sigma_init_a = (0.008)*(0.008)*np.identity(6)
    sigma_init_b = (0.08)*(0.08)*np.identity(6)
    sigma_init_c = (0.02)*(0.02)*np.identity(6)
    sigma_init_d = (0.01)*(0.8)*np.identity(6)

    mus  = [mu_init_a, mu_init_b, mu_init_c, mu_init_d]
    sigmas = [sigma_init_a, sigma_init_b, sigma_init_c, sigma_init_d]

    actual_traj_a = np.zeros((300, 3))
    estimated_traj_a = np.zeros((300, 3))

    actual_traj_b = np.zeros((300, 3))
    estimated_traj_b = np.zeros((300, 3))

    actual_traj_c = np.zeros((300, 3))
    estimated_traj_c = np.zeros((300, 3))

    actual_traj_d = np.zeros((300, 3))
    estimated_traj_d = np.zeros((300, 3))


    #uncertainity_ellipse_values = np.zeros((300, 100, 2))
    
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
    sigma_ri = 0.10
    sigma_ridot = 0.002
    mean_epsilon = np.zeros(6)

    R1 = np.diag(np.array([sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot]))
    R2 = np.diag(np.array([4*sigma_ri*sigma_ri, 8*sigma_ri*sigma_ri, sigma_ri*sigma_ri, 7*sigma_ridot*sigma_ridot,9*sigma_ridot*sigma_ridot,2*sigma_ridot*sigma_ridot]))
    R3 = np.diag(np.array([7*sigma_ri*sigma_ri, 5*sigma_ri*sigma_ri, 4*sigma_ri*sigma_ri, 30*sigma_ridot*sigma_ridot, 10*sigma_ridot*sigma_ridot, 23*sigma_ridot*sigma_ridot]))
    R4 = np.diag(np.array([4*sigma_ri*sigma_ri, 1*sigma_ri*sigma_ri, 6*sigma_ri*sigma_ri, 7*sigma_ridot*sigma_ridot,90*sigma_ridot*sigma_ridot,21*sigma_ridot*sigma_ridot]))

    Rs = [R1, R2, R3, R4]
    ## Observation Model Parameters
    C_t = np.array([[1.0,0,0,0,0,0],
                    [0,1.0,0,0,0,0],
                    [0,0,1.0,0,0,0]])
    sigma_s = 8
    Q1 = sigma_s*sigma_s*np.identity(3)
    Q2 = 0.2*sigma_s*sigma_s*np.identity(3)
    Q3 = 0.4*sigma_s*sigma_s*np.identity(3)
    Q4 = 0.7*sigma_s*sigma_s*np.identity(3)
    Qs = [Q1, Q2, Q3, Q4]
    mean_delta = np.zeros(3)

    for i in range(300):
        tt_now = i * deltaT *0.1
       
        u_t_a = np.array([[cos(tt_now)], [sin(tt_now)], [cos(tt_now)]]) ## Control Inputs are sinusoidal functions
        u_t_b = np.array([[i/800.0], [i/900.0], [cos(tt_now)]]) ## Control Inputs are sinusoidal functions
        u_t_c = np.array([[sin(tt_now)], [-cos(tt_now)], [2*sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        u_t_d = np.array([[0.02], [0.07], [0.04]]) ## Control Inputs are sinusoidal functions

        uts = [u_t_a, u_t_b, u_t_c, u_t_d]

        X_init_a = action_upd(X_init_a, A_t, B_t, u_t_a, mean_epsilon, R1)
        X_init_b = action_upd(X_init_b, A_t, B_t, u_t_b, mean_epsilon, R2)
        X_init_c = action_upd(X_init_c, A_t, B_t, u_t_c, mean_epsilon, R3)
        X_init_d = action_upd(X_init_d, A_t, B_t, u_t_d, mean_epsilon, R4)

        # print("Upd shape =", X_init.shape)
        z1_t = obsv_upd(X_init_a, C_t, mean_delta, Q1)
        z2_t = obsv_upd(X_init_b, C_t, mean_delta, Q2)
        z3_t = obsv_upd(X_init_c, C_t, mean_delta, Q3)
        z4_t = obsv_upd(X_init_d, C_t, mean_delta, Q4)

        sensors = [z1_t, z2_t, z3_t, z4_t] 
        random.shuffle(sensors) ### Pass Random Permutation into the choosing mechanism

        mus, sigmas = choose_measurements_multiple(sensors, mus, sigmas, uts, A_t, B_t, C_t, Rs, Qs)

        #mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)

        #observed_traj[i] = z_t.squeeze()
        actual_traj_a[i] = X_init_a[0:3].squeeze()
        actual_traj_b[i] = X_init_b[0:3].squeeze()
        actual_traj_c[i] = X_init_c[0:3].squeeze()
        actual_traj_d[i] = X_init_d[0:3].squeeze()

        estimated_traj_a[i] = mus[0][0:3].squeeze()
        estimated_traj_b[i] = mus[1][0:3].squeeze()
        estimated_traj_c[i] = mus[2][0:3].squeeze()
        estimated_traj_d[i] = mus[3][0:3].squeeze()


        #x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        #uncertainity_ellipse_values[i] = np.column_stack((x_uncert_points, y_uncert_points))

    # Create a 3D scatter plot for the first dataset
    fig = go.Figure(data=[go.Scatter3d(
        x=actual_traj_a[:, 0],
        y=actual_traj_a[:, 1],
        z=actual_traj_a[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory AeroPlane A'
    )])

    # Add points from the second dataset to the same plot
    fig.add_trace(go.Scatter3d(
        x=actual_traj_b[:, 0],
        y=actual_traj_b[:, 1],
        z=actual_traj_b[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory AeroPlane B'
    ))

    fig.add_trace(go.Scatter3d(
        x=actual_traj_c[:, 0],
        y=actual_traj_c[:, 1],
        z=actual_traj_c[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory AeroPlane C'
    ))

    fig.add_trace(go.Scatter3d(
        x=actual_traj_d[:, 0],
        y=actual_traj_d[:, 1],
        z=actual_traj_d[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory AeroPlane D'
    ))

    fig.add_trace(go.Scatter3d(
        x=estimated_traj_a[:, 0],
        y=estimated_traj_a[:, 1],
        z=estimated_traj_a[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Kalman Estimation of AeroPlane A'
    ))

    fig.add_trace(go.Scatter3d(
        x=estimated_traj_b[:, 0],
        y=estimated_traj_b[:, 1],
        z=estimated_traj_b[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Kalman Estimation of AeroPlane B'
    ))

    fig.add_trace(go.Scatter3d(
        x=estimated_traj_c[:, 0],
        y=estimated_traj_c[:, 1],
        z=estimated_traj_c[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Kalman Estimation of AeroPlane C'
    ))

    fig.add_trace(go.Scatter3d(
        x=estimated_traj_d[:, 0],
        y=estimated_traj_d[:, 1],
        z=estimated_traj_d[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Kalman Estimation of AeroPlane D'
    ))

    # Update the layout if needed
    fig.update_layout(
        title='3D Line Plots of Actual, Estimated Trajectory, 4 agents with Data Association',
        scene=dict(aspectmode='data')
    )

    fig.write_html("part1i_multiLand.html")
    fig.show()
    return 0

def obsv_upd_land(X_t, mean_delta, Q, landmark_num):
    x = X_t[0][0]
    y = X_t[1][0]
    if(landmark_num == 1):
        h_mu_t = np.array([[x], [y], [sqrt((x-150.0)*(x-150.0) + y*y)]])
    elif(landmark_num == 2):
        h_mu_t = np.array([[x], [y], [sqrt((x+150.0)*(x+150.0) + y*y)]])
    elif(landmark_num == 3):
        h_mu_t = np.array([[x], [y], [sqrt(x*x + (y-150.0)*(y-150.0))]])
    elif(landmark_num == 4):
        h_mu_t = np.array([[x], [y], [sqrt(x*x + (y+150.0)*(y+150.0))]])
    elif(landmark_num == 5):
        h_mu_t = np.array([[x], [y], [sqrt((x-25.0)*(x-25.0) + y*y)]])
    else:
        h_mu_t = np.array([[x], [y], [sqrt((x-25.0)*(x-25.0) + (y-400.0)*(y-400.0))]])

    return h_mu_t + noise(mean_delta, Q)


def jacobian_h_l1(mu):
    x = mu[0][0]
    y = mu[1][0]
    v1 = (x - 150.0)/(sqrt((x-150.0)*(x-150.0) + y*y))
    v2 = y/(sqrt((x-150.0)*(x-150.0) + y*y))
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [v1, v2, 0.0, 0.0]])
    return H

def jacobian_h_l2(mu):
    x = mu[0][0]
    y = mu[1][0]
    v1 = (x + 150.0)/(sqrt((x+150.0)*(x-150.0) + y*y))
    v2 = y/(sqrt((x+150.0)*(x+150.0) + y*y))
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [v1, v2, 0.0, 0.0]])
    return H

def jacobian_h_l3(mu):
    x = mu[0][0]
    y = mu[1][0]
    v1 = (x)/(sqrt(x*x + (y-150.0)*(y-150.0)))
    v2 = (y-150.0)/(sqrt(x*x + (y-150.0)*(y-150.0)))

    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [v1, v2, 0.0, 0.0]])
    return H

def jacobian_h_l4(mu):
    x = mu[0][0]
    y = mu[1][0]
    v1 = (x)/(sqrt(x*x + (y+150.0)*(y+150.0)))
    v2 = (y+150.0)/(sqrt(x*x + (y+150.0)*(y+150.0)))
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [v1, v2, 0.0, 0.0]])
    return H

def jacobian_h_l5(mu):
    x = mu[0][0]
    y = mu[1][0]
    v1 = (x - 25.0)/(sqrt((x-25.0)*(x-25.0) + y*y))
    v2 = y/(sqrt((x-25.0)*(x-25.0) + y*y))
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [v1, v2, 0.0, 0.0]])
    return H

def jacobian_h_l6(mu):
    x = mu[0][0]
    y = mu[1][0]
    v1 = (x - 25.0)/(sqrt((x-25.0)*(x-25.0) + (y-400.0)*(y-400.0)))
    v2 = y/(sqrt((x-25.0)*(x-25.0) + (y-400.0)*(y-400.0)))
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [v1, v2, 0.0, 0.0]])
    return H

def ext_kalman_update(mu_tm1, sigma_tm1, u_t, z_t, A_t, B_t, Q_t, R_t, landmark_num):
    n_shape = mu_tm1.shape[0]
    A_t_transpose = np.transpose(A_t)
    mu_t_bar = np.dot(A_t, mu_tm1) + np.dot(B_t, u_t)
    sigma_t_bar = np.dot(np.dot(A_t, sigma_tm1), A_t_transpose) + R_t # R_t is the motion model noise

    x_mu_t_bar = mu_t_bar[0][0]
    y_mu_t_bar = mu_t_bar[1][0]
    if(landmark_num == 1):
        H_t = jacobian_h_l1(mu_t_bar)
        h_mu_t = np.array([[x_mu_t_bar], [y_mu_t_bar], [sqrt((x_mu_t_bar-150.0)*(x_mu_t_bar-150.0) + y_mu_t_bar*y_mu_t_bar)]])
    elif(landmark_num == 2):
        H_t = jacobian_h_l2(mu_t_bar)
        h_mu_t = np.array([[x_mu_t_bar], [y_mu_t_bar], [sqrt((x_mu_t_bar+150.0)*(x_mu_t_bar+150.0) + y_mu_t_bar*y_mu_t_bar)]])
    elif(landmark_num == 3):
        H_t = jacobian_h_l3(mu_t_bar)
        h_mu_t = np.array([[x_mu_t_bar], [y_mu_t_bar], [sqrt(x_mu_t_bar*x_mu_t_bar + (y_mu_t_bar-150.0)*(y_mu_t_bar-150.0))]])
    elif(landmark_num == 4):
        H_t = jacobian_h_l4(mu_t_bar)
        h_mu_t = np.array([[x_mu_t_bar], [y_mu_t_bar], [sqrt(x_mu_t_bar*x_mu_t_bar + (y_mu_t_bar+150.0)*(y_mu_t_bar+150.0))]])
    elif(landmark_num == 5):
        H_t = jacobian_h_l5(mu_t_bar)
        h_mu_t = np.array([[x_mu_t_bar], [y_mu_t_bar], [sqrt((x_mu_t_bar-25.0)*(x_mu_t_bar-25.0) + y_mu_t_bar*y_mu_t_bar)]])
    else:
        H_t = jacobian_h_l6(mu_t_bar)
        h_mu_t = np.array([[x_mu_t_bar], [y_mu_t_bar], [sqrt((x_mu_t_bar-25.0)*(x_mu_t_bar-25.0) + (y_mu_t_bar-400)*(y_mu_t_bar-400))]])

    H_t_transpose = np.transpose(H_t)
    K_t = np.dot(np.dot(sigma_t_bar, H_t_transpose),np.linalg.inv(np.dot(np.dot(H_t, sigma_t_bar),H_t_transpose)+Q_t))
    mu_t = mu_t_bar + np.dot(K_t, (z_t-h_mu_t))
    sigma_t = np.dot((np.identity(n_shape)-np.dot(K_t, H_t)), sigma_t_bar)

    return mu_t, sigma_t


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

def part_2b():
    X_init = np.array([[0.0], [0.0], [0.0], [0.0]]) ## start from (0,0) with vel (0,0)
    mu_init = np.array([[0.0], [0.0], [0.0], [0.0]]) ## start from (0,0) with vel (0,0)
    sigma_init = (0.008)*(0.008)*np.identity(4) ## Prior belief covariance

    actual_traj = np.zeros((1000, 2))
    estimated_traj = np.zeros((1000, 2))
    uncertainity_ellipse_values = np.zeros((1000, 100, 2))

    L1 = np.array([[150.0], [0.0]])
    L2 = np.array([[-150.0], [0.0]])
    L3 = np.array([[0.0], [150.0]])
    L4 = np.array([[0.0], [-150.0]])
    L5 = np.array([[25.0], [0.0]])

    landmark_range = 50.0

    uncertainity_ellipse_values = np.zeros((1000, 100, 2))
    print("xt shape = ", X_init.shape)

    ## Action Model Parameters
    deltaT = 1.0
    A_t = np.array([[1.0,0,deltaT,0],
                    [0,1.0,0,deltaT],
                    [0,0,1.0,0],
                    [0,0,0,1.0]])

    B_t = np.array([[0,0],
                    [0,0],
                    [1.0,0],
                    [0,1.0]])

    mean_epsilon = np.zeros(4)
    R = sigma_init = (0.01)*(0.01)*np.identity(4) ## isotropic Gaussian Noise

    ## Observation Model Parameters
    C_t = np.array([[1.0,0,0,0],
                    [0,1.0,0,0]])
    
    sigma_gps = 10.0
    sigma_lm = 1.0
    Q_gps_only = sigma_gps*sigma_gps*np.identity(2)
    Q_gps_landmark = np.diag(np.array([sigma_gps*sigma_gps, sigma_gps*sigma_gps, sigma_lm*sigma_lm]))
    mean_delta_2 = np.zeros(2)
    mean_delta_3 = np.zeros(3)

    for i in range(1000):
        tt_now = i * deltaT *0.1
        u_t = np.array([[cos(tt_now)/15.0], [sin(tt_now)/15.0]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)

        if(np.linalg.norm(L1 - X_init[0:2].squeeze()) < landmark_range):
            #In L1's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 1)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 1)
    
        elif(np.linalg.norm(L2 - X_init[0:2].squeeze()) < landmark_range):
            #In L2's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 2)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 2)
        
        elif(np.linalg.norm(L3 - X_init[0:2].squeeze()) < landmark_range):
            #In L3's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 3)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 3)

        elif(np.linalg.norm(L4 - X_init[0:2].squeeze()) < landmark_range):
            #In L4's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 4)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 4)

        elif(np.linalg.norm(L5 - X_init[0:2].squeeze()) < landmark_range):
            #In L5's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 5)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 5)

        else:
            #In nobody's range, just do a simple update
            z_t = obsv_upd(X_init, C_t, mean_delta_2, Q_gps_only)
            mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q_gps_only, R)

        #observed_traj[i] = z_t.squeeze()
        actual_traj[i] = X_init[0:2].squeeze()
        estimated_traj[i] = mu_init[0:2].squeeze()

        x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        uncertainity_ellipse_values[i] = np.column_stack((x_uncert_points, y_uncert_points))

        #x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        #uncertainity_ellipse_values[i] = np.column_stack((x_uncert_points, y_uncert_points))

    fig = go.Figure(data=[go.Scatter(
        x=actual_traj[:, 0],
        y=actual_traj[:, 1],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory'
    )])

    fig.add_trace(go.Scatter(
        x=estimated_traj[:, 0],
        y=estimated_traj[:, 1],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Extended Kalman Estimation'
    ))

    # Update the layout if needed
    fig.update_layout(
        title='2D Line Plots of Actual vs Estimated Trajectory, with 5 landmarks',
        scene=dict(aspectmode='data')
    )

    for i in range(1000):
        fig.add_trace(go.Scatter(
            x=uncertainity_ellipse_values[i][:,0],
            y=uncertainity_ellipse_values[i][:,1],
            mode='lines',
            line=dict(color='gold', width=1),
            showlegend=False
        ))



    theta = np.linspace(0, 2 * np.pi, 100)
    x_rotated_1 = 150.0 + (landmark_range * np.cos(theta))   
    x_rotated_2 = -150.0 + (landmark_range * np.cos(theta))   
    x_rotated_3 = 25.0 + (landmark_range * np.cos(theta))   
    x_rotated_4 = 0.0 + (landmark_range * np.cos(theta))   

    y_rotated_1 = 0.0 + (landmark_range * np.sin(theta))
    y_rotated_2 = 150.0 + (landmark_range * np.sin(theta))
    y_rotated_3 = -150.0 + (landmark_range * np.sin(theta))

    
    fig.add_trace(go.Scatter(x=x_rotated_1,y=y_rotated_1,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_2,y=y_rotated_1,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_4,y=y_rotated_2,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_4,y=y_rotated_3,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_3,y=y_rotated_1,mode='lines',line=dict(color='black', width=1),showlegend=False))

    fig.show()

    fig.write_html("part2b_traj.html")
    return 0


def part_2c():
    X_init = np.array([[-200.0], [-50.0], [4.0*np.cos(0.35)], [4.0 * np.sin(0.35)]]) ## start from (0,0) with vel (0,0)
    mu_init = np.array([[-200.0], [-50.0], [4.0*np.cos(0.35)], [4.0 * np.sin(0.35)]]) ## start from (0,0) with vel (0,0)
    sigma_init = (0.008)*(0.008)*np.identity(4) ## Prior belief covariance

    #observed_traj = np.zeros((300, 2))
    num_t_steps = 1000
    actual_traj = np.zeros((num_t_steps, 2))
    estimated_traj = np.zeros((num_t_steps, 2))
    uncertainity_ellipse_values = np.zeros((num_t_steps, 100, 2))

    L1 = np.array([[150.0], [0.0]])
    L2 = np.array([[-150.0], [0.0]])
    L3 = np.array([[0.0], [150.0]])
    L4 = np.array([[0.0], [-150.0]])
    L5 = np.array([[25.0], [0.0]])

    landmark_range = 50.0

    uncertainity_ellipse_values = np.zeros((num_t_steps, 100, 2))
    print("xt shape = ", X_init.shape)

    ## Action Model Parameters
    deltaT = 1.0
    A_t = np.array([[1.0,0,deltaT,0],
                    [0,1.0,0,deltaT],
                    [0,0,1.0,0],
                    [0,0,0,1.0]])

    B_t = np.array([[0,0],
                    [0,0],
                    [1.0,0],
                    [0,1.0]])

    mean_epsilon = np.zeros(4)
    R = sigma_init = (0.01)*(0.01)*np.identity(4) ## isotropic Gaussian Noise

    ## Observation Model Parameters
    C_t = np.array([[1.0,0,0,0],
                    [0,1.0,0,0]])
    
    sigma_gps = 10.0
    sigma_lm = 1.0
    Q_gps_only = sigma_gps*sigma_gps*np.identity(2)
    Q_gps_landmark = np.diag(np.array([sigma_gps*sigma_gps, sigma_gps*sigma_gps, sigma_lm*sigma_lm]))
    mean_delta_2 = np.zeros(2)
    mean_delta_3 = np.zeros(3)

    for i in range(num_t_steps):
        tt_now = i * deltaT *0.1
        u_t = np.array([[cos(i)], [sin(i)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)

        if(np.linalg.norm(L1 - X_init[0:2].squeeze()) < landmark_range):
            #In L1's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 1)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 1)
    
        elif(np.linalg.norm(L2 - X_init[0:2].squeeze()) < landmark_range):
            #In L2's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 2)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 2)
        
        elif(np.linalg.norm(L3 - X_init[0:2].squeeze()) < landmark_range):
            #In L3's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 3)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 3)

        elif(np.linalg.norm(L4 - X_init[0:2].squeeze()) < landmark_range):
            #In L4's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 4)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 4)

        elif(np.linalg.norm(L5 - X_init[0:2].squeeze()) < landmark_range):
            #In L5's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 5)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 5)

        else:
            #In nobody's range, just do a simple update
            z_t = obsv_upd(X_init, C_t, mean_delta_2, Q_gps_only)
            mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q_gps_only, R)

        #observed_traj[i] = z_t.squeeze()
        actual_traj[i] = X_init[0:2].squeeze()
        estimated_traj[i] = mu_init[0:2].squeeze()

        x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        uncertainity_ellipse_values[i] = np.column_stack((x_uncert_points, y_uncert_points))


    fig = go.Figure(data=[go.Scatter(
        x=actual_traj[:, 0],
        y=actual_traj[:, 1],
        mode='lines',
        name='Actual Trajectory'
    )])

    fig.add_trace(go.Scatter(
        x=estimated_traj[:, 0],
        y=estimated_traj[:, 1],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Extended Kalman Estimation'
    ))

    # Update the layout if needed
    fig.update_layout(
        title='2D Line Plots of Actual vs Estimated Trajectory, with 5 landmarks',
        scene=dict(aspectmode='data')
    )

    for i in range(num_t_steps):
        fig.add_trace(go.Scatter(
            x=uncertainity_ellipse_values[i][:,0],
            y=uncertainity_ellipse_values[i][:,1],
            mode='lines',
            line=dict(color='gold', width=1),
            showlegend=False
        ))



    theta = np.linspace(0, 2 * np.pi, 100)
    x_rotated_1 = 150.0 + (landmark_range * np.cos(theta))   
    x_rotated_2 = -150.0 + (landmark_range * np.cos(theta))
    x_rotated_3 = 25.0 + (landmark_range * np.cos(theta))  
    x_rotated_4 = 0.0 + (landmark_range * np.cos(theta)) 

    y_rotated_1 = 0.0 + (landmark_range * np.sin(theta)) 
    y_rotated_2 = 150.0 + (landmark_range * np.sin(theta)) 
    y_rotated_3 = -150.0 + (landmark_range * np.sin(theta)) 

    
    fig.add_trace(go.Scatter(x=x_rotated_1,y=y_rotated_1,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_2,y=y_rotated_1,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_4,y=y_rotated_2,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_4,y=y_rotated_3,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_3,y=y_rotated_1,mode='lines',line=dict(color='black', width=1),showlegend=False))

    fig.show()

    fig.write_html("part2c_pathExample.html")
    return 0

def part_2d():
    X_init = np.array([[-200.0], [-50.0], [4.0*np.cos(0.35)], [4.0 * np.sin(0.35)]]) 
    mu_init = np.array([[-200.0], [-50.0], [4.0*np.cos(0.35)], [4.0 * np.sin(0.35)]]) 
    sigma_init = (0.008)*(0.008)*np.identity(4) ## Prior belief covariance

    #observed_traj = np.zeros((300, 2))
    num_t_steps = 1000
    actual_traj = np.zeros((num_t_steps, 2))
    estimated_traj = np.zeros((num_t_steps, 2))
    uncertainity_ellipse_values = np.zeros((num_t_steps, 100, 2))

    L1 = np.array([[150.0], [0.0]])
    L2 = np.array([[-150.0], [0.0]])
    L3 = np.array([[0.0], [150.0]])
    L4 = np.array([[0.0], [-150.0]])
    L5 = np.array([[25.0], [0.0]])

    landmark_range = 50.0

    uncertainity_ellipse_values = np.zeros((num_t_steps, 100, 2))
    print("xt shape = ", X_init.shape)

    ## Action Model Parameters
    deltaT = 1.0
    A_t = np.array([[1.0,0,deltaT,0],
                    [0,1.0,0,deltaT],
                    [0,0,1.0,0],
                    [0,0,0,1.0]])

    B_t = np.array([[0,0],
                    [0,0],
                    [1.0,0],
                    [0,1.0]])

    mean_epsilon = np.zeros(4)
    R = sigma_init = (0.01)*(0.01)*np.identity(4) ## isotropic Gaussian Noise

    ## Observation Model Parameters
    C_t = np.array([[1.0,0,0,0],
                    [0,1.0,0,0]])
    
    sigma_gps = 10.0
    sigma_lm = 20.0
    Q_gps_only = sigma_gps*sigma_gps*np.identity(2)
    Q_gps_landmark = np.diag(np.array([sigma_gps*sigma_gps, sigma_gps*sigma_gps, sigma_lm*sigma_lm]))
    mean_delta_2 = np.zeros(2)
    mean_delta_3 = np.zeros(3)

    for i in range(num_t_steps):
        tt_now = i * deltaT *0.1
        u_t = np.array([[cos(i)], [sin(i)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)

        if(np.linalg.norm(L1 - X_init[0:2].squeeze()) < landmark_range):
            #In L1's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 1)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 1)
    
        elif(np.linalg.norm(L2 - X_init[0:2].squeeze()) < landmark_range):
            #In L2's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 2)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 2)
        
        elif(np.linalg.norm(L3 - X_init[0:2].squeeze()) < landmark_range):
            #In L3's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 3)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 3)

        elif(np.linalg.norm(L4 - X_init[0:2].squeeze()) < landmark_range):
            #In L4's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 4)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 4)

        elif(np.linalg.norm(L5 - X_init[0:2].squeeze()) < landmark_range):
            #In L5's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 5)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 5)

        else:
            #In nobody's range, just do a simple update
            z_t = obsv_upd(X_init, C_t, mean_delta_2, Q_gps_only)
            mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q_gps_only, R)

        #observed_traj[i] = z_t.squeeze()
        actual_traj[i] = X_init[0:2].squeeze()
        estimated_traj[i] = mu_init[0:2].squeeze()

        x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        uncertainity_ellipse_values[i] = np.column_stack((x_uncert_points, y_uncert_points))

        #x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        #uncertainity_ellipse_values[i] = np.column_stack((x_uncert_points, y_uncert_points))

    fig = go.Figure(data=[go.Scatter(
        x=actual_traj[:, 0],
        y=actual_traj[:, 1],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory'
    )])

    fig.add_trace(go.Scatter(
        x=estimated_traj[:, 0],
        y=estimated_traj[:, 1],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Extended Kalman Estimation'
    ))

    # Update the layout if needed
    fig.update_layout(
        title='2D Line Plots of Actual vs Estimated Trajectory, with 5 landmarks',
        scene=dict(aspectmode='data')
    )

    for i in range(num_t_steps):
        fig.add_trace(go.Scatter(
            x=uncertainity_ellipse_values[i][:,0],
            y=uncertainity_ellipse_values[i][:,1],
            mode='lines',
            line=dict(color='gold', width=1),
            showlegend=False
        ))



    theta = np.linspace(0, 2 * np.pi, 100)
    x_rotated_1 = 150.0 + (landmark_range * np.cos(theta))   
    x_rotated_2 = -150.0 + (landmark_range * np.cos(theta))
    x_rotated_3 = 25.0 + (landmark_range * np.cos(theta))  
    x_rotated_4 = 0.0 + (landmark_range * np.cos(theta)) 

    y_rotated_1 = 0.0 + (landmark_range * np.sin(theta)) 
    y_rotated_2 = 150.0 + (landmark_range * np.sin(theta)) 
    y_rotated_3 = -150.0 + (landmark_range * np.sin(theta)) 

    
    fig.add_trace(go.Scatter(x=x_rotated_1,y=y_rotated_1,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_2,y=y_rotated_1,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_4,y=y_rotated_2,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_4,y=y_rotated_3,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_3,y=y_rotated_1,mode='lines',line=dict(color='black', width=1),showlegend=False))

    fig.show()

    fig.write_html("part2d_increasedNoise.html")
    return 0

def part_2e():
    X_init = np.array([[0.0], [0.0], [0.0], [0.0]]) ## start from (0,0) with vel (0,0)
    mu_init = np.array([[0.0], [0.0], [0.0], [0.0]]) ## start from (0,0) with vel (0,0)
    sigma_init = (0.008)*(0.008)*np.identity(4) ## Prior belief covariance

    #observed_traj = np.zeros((300, 2))
    num_t_steps = 1000
    actual_traj = np.zeros((num_t_steps, 2))
    estimated_traj = np.zeros((num_t_steps, 2))
    uncertainity_ellipse_values = np.zeros((num_t_steps, 100, 2))

    L1 = np.array([[150.0], [0.0]])
    L2 = np.array([[-150.0], [0.0]])
    L3 = np.array([[0.0], [150.0]])
    L4 = np.array([[0.0], [-150.0]])
    L5 = np.array([[25.0], [0.0]])
    L6 = np.array([[25.0], [400.0]])
    landmark_range = 50.0

    uncertainity_ellipse_values = np.zeros((num_t_steps, 100, 2))
    print("xt shape = ", X_init.shape)

    ## Action Model Parameters
    deltaT = 1.0
    A_t = np.array([[1.0,0,deltaT,0],
                    [0,1.0,0,deltaT],
                    [0,0,1.0,0],
                    [0,0,0,1.0]])

    B_t = np.array([[0,0],
                    [0,0],
                    [1.0,0],
                    [0,1.0]])

    mean_epsilon = np.zeros(4)
    R = sigma_init = (0.01)*(0.01)*np.identity(4) ## isotropic Gaussian Noise

    ## Observation Model Parameters
    C_t = np.array([[1.0,0,0,0],
                    [0,1.0,0,0]])
    
    sigma_gps = 10.0
    sigma_lm = 1.0
    Q_gps_only = sigma_gps*sigma_gps*np.identity(2)
    Q_gps_landmark = np.diag(np.array([sigma_gps*sigma_gps, sigma_gps*sigma_gps, sigma_lm*sigma_lm]))
    mean_delta_2 = np.zeros(2)
    mean_delta_3 = np.zeros(3)

    for i in range(num_t_steps):
        tt_now = i * deltaT *0.1
        u_t = np.array([[cos(tt_now)/15.0], [sin(tt_now)/15.0]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)

        if(np.linalg.norm(L1 - X_init[0:2].squeeze()) < landmark_range):
            #In L1's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 1)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 1)
    
        elif(np.linalg.norm(L2 - X_init[0:2].squeeze()) < landmark_range):
            #In L2's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 2)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 2)
        
        elif(np.linalg.norm(L3 - X_init[0:2].squeeze()) < landmark_range):
            #In L3's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 3)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 3)

        elif(np.linalg.norm(L4 - X_init[0:2].squeeze()) < landmark_range):
            #In L4's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 4)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 4)

        elif(np.linalg.norm(L5 - X_init[0:2].squeeze()) < landmark_range):
            #In L5's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 5)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 5)
        elif(np.linalg.norm(L6 - X_init[0:2].squeeze()) < landmark_range):
            #In L5's range
            z_t = obsv_upd_land(X_init, mean_delta_3, Q_gps_landmark, 6)
            mu_init, sigma_init = ext_kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, Q_gps_landmark, R, 6)
        else:
            #In nobody's range, just do a simple update
            z_t = obsv_upd(X_init, C_t, mean_delta_2, Q_gps_only)
            mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q_gps_only, R)

        #observed_traj[i] = z_t.squeeze()
        actual_traj[i] = X_init[0:2].squeeze()
        estimated_traj[i] = mu_init[0:2].squeeze()

        x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        uncertainity_ellipse_values[i] = np.column_stack((x_uncert_points, y_uncert_points))

        #x_uncert_points, y_uncert_points = uncer_ellipse_params(mu_init, sigma_init)
        #uncertainity_ellipse_values[i] = np.column_stack((x_uncert_points, y_uncert_points))

    fig = go.Figure(data=[go.Scatter(
        x=actual_traj[:, 0],
        y=actual_traj[:, 1],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory'
    )])

    fig.add_trace(go.Scatter(
        x=estimated_traj[:, 0],
        y=estimated_traj[:, 1],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Extended Kalman Estimation'
    ))

    # Update the layout if needed
    fig.update_layout(
        title='2D Line Plots of Actual vs Estimated Trajectory, with 5 landmarks',
        scene=dict(aspectmode='data')
    )

    for i in range(num_t_steps):
        fig.add_trace(go.Scatter(
            x=uncertainity_ellipse_values[i][:,0],
            y=uncertainity_ellipse_values[i][:,1],
            mode='lines',
            line=dict(color='gold', width=1),
            showlegend=False
        ))



    theta = np.linspace(0, 2 * np.pi, 100)
    x_rotated_1 = 150.0 + (landmark_range * np.cos(theta))   
    x_rotated_2 = -150.0 + (landmark_range * np.cos(theta))
    x_rotated_3 = 25.0 + (landmark_range * np.cos(theta))  
    x_rotated_4 = 0.0 + (landmark_range * np.cos(theta)) 

    y_rotated_1 = 0.0 + (landmark_range * np.sin(theta)) 
    y_rotated_2 = 150.0 + (landmark_range * np.sin(theta)) 
    y_rotated_3 = -150.0 + (landmark_range * np.sin(theta))
    y_rotated_4 = 400.0 + (landmark_range * np.sin(theta)) 

    
    fig.add_trace(go.Scatter(x=x_rotated_1,y=y_rotated_1,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_2,y=y_rotated_1,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_4,y=y_rotated_2,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_4,y=y_rotated_3,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_3,y=y_rotated_1,mode='lines',line=dict(color='black', width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=x_rotated_3,y=y_rotated_4,mode='lines',line=dict(color='black', width=1),showlegend=False))

    fig.show()
    fig.write_html("part2e_extralandmark.html")
    return 0


## To Run, simply uncomment the appropriate part

# part_a()
# part_c()
# part_d()
# part_e()
# part_f()
part_g()
# part_h()
# part_i()
# part_2b()
# part_2c()
# part_2d()
# part_2e()

