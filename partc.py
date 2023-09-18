import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from math import cos, sin, sqrt

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
    return np.dot(C_t, X_t) + noise(mean_delta, Q)

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

    #observed_traj = np.zeros((300, 2))
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
    return 0

def part_2d():
    X_init = np.array([[-200.0], [-50.0], [0.0], [0.0]]) ## start from (0,0) with vel (0,0)
    mu_init = np.array([[-200.0], [-50.0], [0.0], [0.0]]) ## start from (0,0) with vel (0,0)
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
    return 0

part_2c()