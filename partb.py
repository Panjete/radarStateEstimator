import numpy as np
from math import sqrt, sin, cos
import random
import plotly.graph_objects as go


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
    mean_state_rel = mean_state[0:3].squeeze()
    return sqrt(np.dot(np.dot(np.transpose(point - mean_state_rel), np.linalg.inv(cov_rel)), point - mean_state_rel))

## Makes the concious choice of selecting apt sensor measurement for the right path
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
    

    #mu_tb2, sigma_tb2 = kalman_update_evolve(mu_tm2, sigma_tm2, u2_t, A, B, R2)

    set_sensor_measure = []
    for sm in sensor_measures:
        set_sensor_measure.append(sm)

    measures_ordered = []
    for j in range(len(mus)):
        min_dis = float('inf')
        min_dis_at = sensor_measures[0] #Rand initialise
        min_dist_ind = 0
        for i in range(len(set_sensor_measure)):
            sm = set_sensor_measure[i]
            if(manhattan_distance(sm, mus[j]) < min_dis):
                min_dis_at = sm
                min_dist_ind = i
        
        measures_ordered.append(min_dis_at)

        sm_copy = []
        for i in range(len(set_sensor_measure)):
            if (i!= min_dist_ind):
                sm_copy.append(set_sensor_measure[i])
        
        set_sensor_measure = sm_copy

    mus_t = []
    sigmas_t = []
    for i in range(len(mus)):
        mus_iiiiii, sigma_iiiiii = kalman_update_correct(mu_tb_s[i], sigma_tb_s[i], measures_ordered[i], C, Qs[i])
        mus_t.append(mus_iiiiii)
        sigmas_t.append(sigma_iiiiii)
    

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

        #mu_init, sigma_init = kalman_update(mu_init, sigma_init, u_t, z_t, A_t, B_t, C_t, Q, R)

        #observed_traj[i] = z_t.squeeze()
        actual_traj_a[i] = X_init_a[0:3].squeeze()
        actual_traj_b[i] = X_init_b[0:3].squeeze()

        estimated_traj_a[i] = mu_init_a[0:3].squeeze()
        estimated_traj_b[i] = mu_init_b[0:3].squeeze()


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

    # Update the layout if needed
    fig.update_layout(
        title='3D Line Plots of Actual, Estimated Trajectory with Data Association',
        scene=dict(aspectmode='data')
    )

    # fig2 = go.Figure(data=[go.Scatter(
    #     x=estimated_traj[:, 0],
    #     y=estimated_traj[:, 1],
    #     mode='lines',
    #     #marker=dict(size=3),
    #     name='Trajectory Points',
    # )])

    # for i in range(300):
    #     fig2.add_trace(go.Scatter(
    #         x=uncertainity_ellipse_values[i][:,0],
    #         y=uncertainity_ellipse_values[i][:,1],
    #         mode='lines',
    #         line=dict(color='red', width=1),
    #         showlegend=False
    #     ))

    # fig2.update_layout(
    #     title='Projection of Estimated Traj into X-Y plane',
    #     scene=dict(aspectmode='data')
    # )


    
    # Show the plot
    #fig.show()
    fig.show()
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
    sigma_ri = 1.0
    sigma_ridot = 0.008
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
    Q2 = 2*sigma_s*sigma_s*np.identity(3)
    Q3 = 4*sigma_s*sigma_s*np.identity(3)
    Q4 = 7*sigma_s*sigma_s*np.identity(3)
    Qs = [Q1, Q2, Q3, Q4]
    mean_delta = np.zeros(3)

    for i in range(300):
        tt_now = i * deltaT *0.1
        # u_t_a = np.array([[cos(tt_now)], [sin(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        # u_t_b = np.array([[sin(tt_now)], [cos(tt_now)], [cos(tt_now)]]) ## Control Inputs are sinusoidal functions
        # u_t_c = np.array([[sin(tt_now)], [cos(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions
        # u_t_d = np.array([[cos(tt_now)], [cos(tt_now)], [sin(tt_now)]]) ## Control Inputs are sinusoidal functions

        u_t_a = np.array([[1], [3], [-2]]) ## Control Inputs are sinusoidal functions
        u_t_b = np.array([[-1], [5], [1]]) ## Control Inputs are sinusoidal functions
        u_t_c = np.array([[1], [-6], [5]]) ## Control Inputs are sinusoidal functions
        u_t_d = np.array([[2], [7], [4]]) ## Control Inputs are sinusoidal functions

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
        random.shuffle(sensors)

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

    # fig2 = go.Figure(data=[go.Scatter(
    #     x=estimated_traj[:, 0],
    #     y=estimated_traj[:, 1],
    #     mode='lines',
    #     #marker=dict(size=3),
    #     name='Trajectory Points',
    # )])

    # for i in range(300):
    #     fig2.add_trace(go.Scatter(
    #         x=uncertainity_ellipse_values[i][:,0],
    #         y=uncertainity_ellipse_values[i][:,1],
    #         mode='lines',
    #         line=dict(color='red', width=1),
    #         showlegend=False
    #     ))

    # fig2.update_layout(
    #     title='Projection of Estimated Traj into X-Y plane',
    #     scene=dict(aspectmode='data')
    # )


    
    # Show the plot
    #fig.show()
    fig.show()
    return 0


part_i()