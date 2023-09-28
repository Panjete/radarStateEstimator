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

def actual_state_variables():
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

    fig = go.Figure(data=[go.Scatter3d(x=observed_vals[:, 0],y=observed_vals[:, 1],z=observed_vals[:, 2],mode='lines', name='Observed Trajectory')])
    fig.add_trace(go.Scatter3d(x=actual_vals[:, 0],y=actual_vals[:, 1],z=actual_vals[:, 2],mode='lines',name='Actual Trajectory'))
    fig.update_layout(title='3D Line Plots of Actual Vs Observed Trajectories',scene=dict(aspectmode='data'))
    fig.show()

    return 0

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
def simulate_and_estimate():
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

    fig = go.Figure(data=[go.Scatter3d(x=observed_traj[:, 0],y=observed_traj[:, 1],z=observed_traj[:, 2],mode='lines',name='Noisy Observations')])

    fig.add_trace(go.Scatter3d(x=actual_traj[:, 0],y=actual_traj[:, 1],z=actual_traj[:, 2],mode='lines',name='Actual Trajectory'))
    fig.add_trace(go.Scatter3d(x=estimated_traj[:, 0],y=estimated_traj[:, 1],z=estimated_traj[:, 2],mode='lines',name='Kalman Estimation'))
    fig.update_layout(title='3D Line Plots of Actual vs Estimated Trajectory, and Noisy Observations',scene=dict(aspectmode='data'))
    fig2 = go.Figure(data=[go.Scatter(x=estimated_traj[:, 0],y=estimated_traj[:, 1],mode='lines',name='Trajectory Points',)])

    for i in range(300):
        fig2.add_trace(go.Scatter(x=uncertainity_ellipse_values[i][:,0],y=uncertainity_ellipse_values[i][:,1],mode='lines',line=dict(color='red', width=1),showlegend=False))

    fig2.update_layout(title='Projection of Estimated Traj into X-Y plane',scene=dict(aspectmode='data'))

    fig.show()
    fig2.show()

    fig.write_html("trajs.html")
    fig2.write_html("ue.html")
    return 0

## To analyse estimation accuracy of variables not monitored by the sensors
def check_velocity_estimation():
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


    fig = go.Figure(data=[go.Scatter3d(x=actual_vel[:, 0],y=actual_vel[:, 1],z=actual_vel[:, 2],mode='lines',name='Actual Velocity Values')])
    fig.add_trace(go.Scatter3d(x=estimated_vel[:, 0],y=estimated_vel[:, 1],z=estimated_vel[:, 2],mode='lines',name='Estimated Velocity Velocity'))

    # Update the layout if needed
    fig.update_layout(title='3D Line Plots of Actual vs Estimated Velocity Points',scene=dict(aspectmode='data'))

    fig.show()
    fig.write_html("velPlot.html")
    return 0


## Analyse effect of noise vairations on estimates and trajectories
def noise_variations_analysis():
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

    fig1.write_html("posNoiseComp.html")
    fig2.write_html("velNoiseComp.html")
    fig3.write_html("sensorNoiseComp.html")
    fig4.write_html("ue_noiseVariations.html")
    fig5.write_html("allTrajs_noisy.html")
    return 0