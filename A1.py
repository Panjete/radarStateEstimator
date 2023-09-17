import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from math import cos, sin

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

def part_c():
    X_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    mu_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    sigma_init = (0.008)*(0.008)*np.identity(6)
    observed_traj = np.zeros((300, 3))
    actual_traj = np.zeros((300, 3))
    estimated_traj = np.zeros((300, 3))
    print("xt shape = ", X_init.shape)

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
        tt_now = i * deltaT *0.1
        u_t = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        # print("Upd shape =", X_init.shape)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)

        observed_traj[i] = z_t.squeeze()
        actual_traj[i] = X_init[0:3].squeeze()
        estimated_traj[i] = mu_init[0:3].squeeze()

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
        title='3D Line Plots of Actual vs Estimated Trajectory, and Noisy Observations',
        scene=dict(aspectmode='data')
    )

    fig2 = go.Figure(data=[go.Scatter(
        x=estimated_traj[:, 0],
        y=estimated_traj[:, 1],
        mode='markers',
        marker=dict(size=3),
        name='Trajectory Points',
    )])

    fig2.update_layout(
        title='Projection of Estimated Traj into X-Y plane',
        scene=dict(aspectmode='data')
    )
    
    # Show the plot
    #fig.show()
    fig2.show()
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
        tt_now = i * deltaT *0.1
        u_t = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        # print("Upd shape =", X_init.shape)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)

        observed_traj_normal[i] = z_t.squeeze()
        actual_traj_normal[i] = X_init[0:3].squeeze()
        estimated_traj_normal[i] = mu_init[0:3].squeeze()

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
        tt_now = i * deltaT *0.1
        u_t = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        # print("Upd shape =", X_init.shape)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)

        observed_traj_pos_noise_enhanced[i] = z_t.squeeze()
        actual_traj_pos_noise_enhanced[i] = X_init[0:3].squeeze()
        estimated_traj_pos_noise_enhanced[i] = mu_init[0:3].squeeze()

    ##### Velocity Noise Enhanced 3/4
    sigma_ri = 1.0
    sigma_ridot = 0.08
    mean_epsilon = np.zeros(6)
    R = np.diag(np.array([sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ri*sigma_ri, sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot,sigma_ridot*sigma_ridot]))
    sigma_s = 8
    Q = sigma_s*sigma_s*np.identity(3)
    mean_delta = np.zeros(3)
    X_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    mu_init = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) ## start from (0,0,0) with vel (0,0,0)
    sigma_init = (0.008)*(0.008)*np.identity(6)

    for i in range(300):
        tt_now = i * deltaT *0.1
        u_t = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        # print("Upd shape =", X_init.shape)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)

        observed_traj_vel_noise_enhanced[i] = z_t.squeeze()
        actual_traj_vel_noise_enhanced[i] = X_init[0:3].squeeze()
        estimated_traj_vel_noise_enhanced[i] = mu_init[0:3].squeeze()

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
        tt_now = i * deltaT *0.1
        u_t = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        # print("Upd shape =", X_init.shape)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)

        observed_traj_sensor_noise_enhanced[i] = z_t.squeeze()
        actual_traj_sensor_noise_enhanced[i] = X_init[0:3].squeeze()
        estimated_traj_sensor_noise_enhanced[i] = mu_init[0:3].squeeze()



    # Create a 3D scatter plot for the first dataset
    fig1 = go.Figure(data = [go.Scatter3d(
        x=observed_traj_normal[:, 0],
        y=observed_traj_normal[:, 1],
        z=observed_traj_normal[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Noisy Observations Normal'
    )])

    # Add points from the second dataset to the same plot
    fig1.add_trace(go.Scatter3d(
        x=actual_traj_normal[:, 0],
        y=actual_traj_normal[:, 1],
        z=actual_traj_normal[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory Normal'
    ))

    fig1.add_trace(go.Scatter3d(
        x=estimated_traj_normal[:, 0],
        y=estimated_traj_normal[:, 1],
        z=estimated_traj_normal[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Kalman Estimation Normal'
    ))

    fig1.add_trace(go.Scatter3d(
        x=observed_traj_pos_noise_enhanced[:, 0],
        y=observed_traj_pos_noise_enhanced[:, 1],
        z=observed_traj_pos_noise_enhanced[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Noisy Observations Position Noise Enhanced'
    ))

    # Add points from the second dataset to the same plot
    fig1.add_trace(go.Scatter3d(
        x=actual_traj_pos_noise_enhanced[:, 0],
        y=actual_traj_pos_noise_enhanced[:, 1],
        z=actual_traj_pos_noise_enhanced[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory Position Noise Enhanced'
    ))

    fig1.add_trace(go.Scatter3d(
        x=estimated_traj_pos_noise_enhanced[:, 0],
        y=estimated_traj_pos_noise_enhanced[:, 1],
        z=estimated_traj_pos_noise_enhanced[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Kalman Estimation Position Noise Enhanced'
    ))

    fig2 = go.Figure(data=[go.Scatter3d(
        x=observed_traj_normal[:, 0],
        y=observed_traj_normal[:, 1],
        z=observed_traj_normal[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Noisy Observations Normal'
    )])

    # Add points from the second dataset to the same plot
    fig2.add_trace(go.Scatter3d(
        x=actual_traj_normal[:, 0],
        y=actual_traj_normal[:, 1],
        z=actual_traj_normal[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory Normal'
    ))

    fig2.add_trace(go.Scatter3d(
        x=estimated_traj_normal[:, 0],
        y=estimated_traj_normal[:, 1],
        z=estimated_traj_normal[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Kalman Estimation Normal'
    ))

    fig2.add_trace(go.Scatter3d(
        x=observed_traj_vel_noise_enhanced[:, 0],
        y=observed_traj_vel_noise_enhanced[:, 1],
        z=observed_traj_vel_noise_enhanced[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Noisy Observations Velocity Noise Enhanced'
    ))

    # Add points from the second dataset to the same plot
    fig2.add_trace(go.Scatter3d(
        x=actual_traj_vel_noise_enhanced[:, 0],
        y=actual_traj_vel_noise_enhanced[:, 1],
        z=actual_traj_vel_noise_enhanced[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory Velocity Noise Enhanced'
    ))

    fig2.add_trace(go.Scatter3d(
        x=estimated_traj_vel_noise_enhanced[:, 0],
        y=estimated_traj_vel_noise_enhanced[:, 1],
        z=estimated_traj_vel_noise_enhanced[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Kalman Estimation Velocity Noise Enhanced'
    ))

    fig3 = go.Figure(data=[go.Scatter3d(
        x=observed_traj_normal[:, 0],
        y=observed_traj_normal[:, 1],
        z=observed_traj_normal[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Noisy Observations Normal'
    )])

    # Add points from the second dataset to the same plot
    fig3.add_trace(go.Scatter3d(
        x=actual_traj_normal[:, 0],
        y=actual_traj_normal[:, 1],
        z=actual_traj_normal[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory Normal'
    ))

    fig3.add_trace(go.Scatter3d(
        x=estimated_traj_normal[:, 0],
        y=estimated_traj_normal[:, 1],
        z=estimated_traj_normal[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Kalman Estimation Normal'
    ))

    fig3.add_trace(go.Scatter3d(
        x=observed_traj_sensor_noise_enhanced[:, 0],
        y=observed_traj_sensor_noise_enhanced[:, 1],
        z=observed_traj_sensor_noise_enhanced[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Noisy Observations Sensor Noise Enhanced'
    ))

    # Add points from the second dataset to the same plot
    fig3.add_trace(go.Scatter3d(
        x=actual_traj_sensor_noise_enhanced[:, 0],
        y=actual_traj_sensor_noise_enhanced[:, 1],
        z=actual_traj_sensor_noise_enhanced[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Actual Trajectory Sensor Noise Enhanced'
    ))

    fig3.add_trace(go.Scatter3d(
        x=estimated_traj_sensor_noise_enhanced[:, 0],
        y=estimated_traj_sensor_noise_enhanced[:, 1],
        z=estimated_traj_sensor_noise_enhanced[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Kalman Estimation Sensor Noise Enhanced'
    ))

    
    # fig.add_trace(go.Scatter3d(
    #     x=observed_traj_vel_noise_enhanced[:, 0],
    #     y=observed_traj_vel_noise_enhanced[:, 1],
    #     z=observed_traj_vel_noise_enhanced[:, 2],
    #     mode='lines',
    #     #marker=dict(size=3),
    #     name='Noisy Observations Velocity Noise Enhanced'
    # ))

    # # Add points from the second dataset to the same plot
    # fig.add_trace(go.Scatter3d(
    #     x=actual_traj_vel_noise_enhanced[:, 0],
    #     y=actual_traj_vel_noise_enhanced[:, 1],
    #     z=actual_traj_vel_noise_enhanced[:, 2],
    #     mode='lines',
    #     #marker=dict(size=3),
    #     name='Actual Trajectory Velocity Noise Enhanced'
    # ))

    # fig.add_trace(go.Scatter3d(
    #     x=estimated_traj_vel_noise_enhanced[:, 0],
    #     y=estimated_traj_vel_noise_enhanced[:, 1],
    #     z=estimated_traj_vel_noise_enhanced[:, 2],
    #     mode='lines',
    #     #marker=dict(size=3),
    #     name='Kalman Estimation Velocity Noise Enhanced'
    # ))

    # fig.add_trace(go.Scatter3d(
    #     x=observed_traj_sensor_noise_enhanced[:, 0],
    #     y=observed_traj_sensor_noise_enhanced[:, 1],
    #     z=observed_traj_sensor_noise_enhanced[:, 2],
    #     mode='lines',
    #     #marker=dict(size=3),
    #     name='Noisy Observations Sensor Noise Enhanced'
    # ))

    # # Add points from the second dataset to the same plot
    # fig.add_trace(go.Scatter3d(
    #     x=actual_traj_sensor_noise_enhanced[:, 0],
    #     y=actual_traj_sensor_noise_enhanced[:, 1],
    #     z=actual_traj_sensor_noise_enhanced[:, 2],
    #     mode='lines',
    #     #marker=dict(size=3),
    #     name='Actual Trajectory Sensor Noise Enhanced'
    # ))

    # fig.add_trace(go.Scatter3d(
    #     x=estimated_traj_sensor_noise_enhanced[:, 0],
    #     y=estimated_traj_sensor_noise_enhanced[:, 1],
    #     z=estimated_traj_sensor_noise_enhanced[:, 2],
    #     mode='lines',
    #     #marker=dict(size=3),
    #     name='Kalman Estimation Sensor Noise Enhanced'
    # ))

    # Update the layout if needed
    # fig.update_layout(
    #     title='3D Line Plots of Trajectories and Observations for varying noises',
    #     scene=dict(aspectmode='data')
    # )

    # fig2 = go.Figure(data=[go.Scatter(
    #     x=estimated_traj[:, 0],
    #     y=estimated_traj[:, 1],
    #     mode='markers',
    #     marker=dict(size=3),
    #     name='Trajectory Points',
    # )])

    # fig2.update_layout(
    #     title='Projection of Estimated Traj into X-Y plane',
    #     scene=dict(aspectmode='data')
    # )
    
    # Show the plot
    #fig1.show()
    fig1.show()
    fig3.show()
    fig2.show()
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
        tt_now = i * deltaT *0.1
        u_t = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        X_init = action_upd(X_init, A_t, B_t, u_t, mean_epsilon, R)
        # print("Upd shape =", X_init.shape)
        z_t = obsv_upd(X_init, C_t, mean_delta, Q)

        if ((i < 50) or (i > 80 and i < 200) or (i >230)):
            mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)
        else:
            mu_init, sigma_init = kalman_update_pure_evolution(mu_init, sigma_init, u_t, A_t, B_t, R)

        observed_traj[i] = z_t.squeeze()
        actual_traj[i] = X_init[0:3].squeeze()
        estimated_traj[i] = mu_init[0:3].squeeze()
        #print(mu_init)

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

    fig2.update_layout(
        title='Projection of Estimated Traj into X-Y plane',
        scene=dict(aspectmode='data')
    )
    
    # Show the plot
    #fig.show()
    fig.show()
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
        tt_now = i * deltaT *0.1
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
        #print(mu_init)

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

    fig2.update_layout(
        title='Projection of Estimated Traj into X-Y plane',
        scene=dict(aspectmode='data')
    )
    
    # Show the plot
    #fig.show()
    fig.show()
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
        tt_now = i * deltaT *0.1
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
        #marker=dict(size=3),
        name='Actual Velocity Values'
    )])

    # Add points from the second dataset to the same plot
    fig.add_trace(go.Scatter3d(
        x=estimated_vel[:, 0],
        y=estimated_vel[:, 1],
        z=estimated_vel[:, 2],
        mode='lines',
        #marker=dict(size=3),
        name='Estimated Velocity Velocity'
    ))

    # Update the layout if needed
    fig.update_layout(
        title='3D Line Plots of Actual vs Estimated Velocity Points',
        scene=dict(aspectmode='data')
    )

    # fig2 = go.Figure(data=[go.Scatter(
    #     x=estimated_traj[:, 0],
    #     y=estimated_traj[:, 1],
    #     mode='markers',
    #     marker=dict(size=3),
    #     name='Trajectory Points',
    # )])

    # fig2.update_layout(
    #     title='Projection of Estimated Traj into X-Y plane',
    #     scene=dict(aspectmode='data')
    # )
    
    fig.show()
    return 0


part_g()