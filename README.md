## radarStateEstimator

### State Estimation using Kalman Filters

Models Noisy Airplane Observation Data, predicting it's trajectory and maintaining a belief state of it's  positional and velocity co-ordinates in 3 dimensions. 

I experiment with variational noise parameters, with an analysis of the effect of positional update, velocity update, and sensor noise. 

The motion models and observation models are implemented, and the simulation is made robust enough to propagate belief even when some sensor drops off it's observations. 


The sensor variations depict the following actual and estimated trajectories.
<img width="1184" alt="Screenshot 2023-09-26 at 3 59 43 PM" src="https://github.com/Panjete/radarStateEstimator/assets/103451209/53989f02-d839-4e54-a65a-809466ab838b">

### Data Association

When observing multi-object data, the sensors may not often be able to associate the observation with the object that produces it, in which case techniques are required to map the observations to the apt object.

This is a complicated problem, especially when the trajectories of two or more airplanes are close enough. I propagate the belief, maintain a confidence internval, try out some permutations, and agree on the assignment that makes minimizes the "distance" for all trajectory-observation pairs.

I implement the Mahalanobis distance, for measuring how associated an observation (a point) is to a trajectory (a probability distribution).

My algorithm for association is based on a greedy-permutation approach, and as observed in the plot of estimated and actual trajectories of 4 airplanes below, is able to map them correctly.

<img width="565" alt="Screenshot 2023-09-26 at 4 06 34 PM" src="https://github.com/Panjete/radarStateEstimator/assets/103451209/9eb0b6e2-9132-46ba-a6b6-b66cc3a9f0e6">
