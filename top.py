## Top level file for invoking all simulations

import argparse
from kalman_estimator import simulate_and_estimate, noise_variations_analysis, check_velocity_estimation
from handle_sensor_drops import all_sensors_drop_off, sensor_X_drops
from data_association import two_airplanes_da, multi_airplanes_da
from ekf_non_lin_measmts import ekf_for_landmarks, landmarks_noise_enhanced, model_extra_landmark

aparser = argparse.ArgumentParser(description="TopLevelInputSelector")
aparser.add_argument("choose_simulation", nargs=1)

args = aparser.parse_args()
choose_simulation = args.choose_simulation[0]

match choose_simulation:
    case "simulate_kf":
        simulate_and_estimate()
    case "simulate_noise_kf":
        noise_variations_analysis()
    case "simulate_velocity_kf":
        check_velocity_estimation()
    case "simulate_all_sensors_drop":
        all_sensors_drop_off()
    case "simulate_one_sensor_drop":
        sensor_X_drops()
    case "simulate_da_2":
        two_airplanes_da()
    case "simulate_da_multi":
        multi_airplanes_da()
    case "simulate_landmarks":
        ekf_for_landmarks()
    case "simulate_landmark_noise":
        landmarks_noise_enhanced()
    case "simulate_extra_landmark":
        model_extra_landmark()