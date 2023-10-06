## State Estimator

### How to run the Simulations

"top.py" enlists the variety of simulations possible.

In the project directory, simply call `top.py` with the relevant keyword of that simulation, enlisted in the file itself.

For example, to simulate, estimate and plot a 3D trajectory, call `python top.py simulate_kf`.

### Plots

A collection of simulations and Annotated Graphs can be accessed from [this link](https://drive.google.com/drive/folders/1bMSKxHuA2BFccCWvGDtmowpGfYLcC5Oa?usp=share_link) .

### State Estimation using Kalman Filters

Models Noisy Airplane Observation Data, predicting it's trajectory and maintaining a belief state of it's  positional and velocity co-ordinates in 3 dimensions. 

This image shows the estimated and actual trajectories (left), and the quality of estimation can be really apprectiated when viewed in contrast to the (very) noisy observations (right) I actually recieved from the sensor.

<p align="center">
<img width="330" alt="Screenshot 2023-09-26 at 4 11 35 PM" src="https://github.com/Panjete/radarStateEstimator/assets/103451209/0dd0b459-50bc-4505-a8e6-46f6ae321082">
<img width="348" alt="Screenshot 2023-09-26 at 4 11 51 PM" src="https://github.com/Panjete/radarStateEstimator/assets/103451209/185d1845-41b5-4f6f-8039-06f296ecbfd9">
</p>

I experiment with variational noise parameters, with an analysis of the effect of positional update, velocity update, and sensor noise. 

The sensor variations depict the following actual and estimated trajectories.

<p align="center">
<img width="1184" alt="Screenshot 2023-09-26 at 3 59 43 PM" src="https://github.com/Panjete/radarStateEstimator/assets/103451209/53989f02-d839-4e54-a65a-809466ab838b">
</p>

The motion models and observation models are implemented, and the simulation is made robust enough to propagate belief even when some sensor drops off it's observations.

The image below demostrates periods of sensor faults, where linear-sh stretches are introduced by the estimator to propagate belief into next timestamps without any measurements.
<p align="center">
<img width="600" alt="Screenshot 2023-09-26 at 4 14 30 PM" src="https://github.com/Panjete/radarStateEstimator/assets/103451209/9b901e3e-6305-4598-acc9-dacba1f51a3d">
</p>
  
Further, if the sensor of any 1 direction drops of, we become uncertain in that direction, but the other two are still updated accurately. This is evident from the trajectory plots (left) and the uncertainity ellipses (right, projected into XY plane) where the X-sensor develops a fault at around the 1/3rd timestamp of the simulation. The left image shows how the the estimated trajectory is fairly accurate in it’s y and z coordinates. However, it evolves the last measured update in X, which diverges with time since these is a velocity update happening in X as well.


<p align="center">
<img width="210" alt="Screenshot 2023-09-26 at 4 23 28 PM" src="https://github.com/Panjete/radarStateEstimator/assets/103451209/1f9efb97-b9ca-4129-8d04-7241600fbdde">
<img width="400" alt="Screenshot 2023-09-26 at 4 21 37 PM" src="https://github.com/Panjete/radarStateEstimator/assets/103451209/f3f80c5f-bae9-4ec0-984e-80086207db06">
</p>


### Data Association

When observing multi-object data, the sensors may not often be able to associate the observation with the object that produces it, in which case techniques are required to map the observations to the apt object.

This is a complicated problem, especially when the trajectories of two or more airplanes are close enough. I propagate the belief, maintain a confidence internval, try out some permutations, and agree on the assignment that makes minimizes the "distance" for all trajectory-observation pairs.

I implement the Mahalanobis distance, for measuring how associated an observation (a point) is to a trajectory (a probability distribution).

My algorithm for association is based on a greedy-permutation approach, and as observed in the plot of estimated and actual trajectories of 4 airplanes below, is able to map them correctly.

<p align="center">
<img width="565" alt="Screenshot 2023-09-26 at 4 06 34 PM" src="https://github.com/Panjete/radarStateEstimator/assets/103451209/9eb0b6e2-9132-46ba-a6b6-b66cc3a9f0e6">
</p>

### Incorporating Landmarks

Non-Linear Distance Measurement was obtained via 5 landmarks, and I incorporated these to make a better estimate of the state. This was done using Extended Kalman Filters, where a linearisation at the mean yields an approximate gaussian that is propagated.

Simulation shows how the extra information provided by the landmark makes the estimate accurate, and how quickly the estimated diverge when the airplane leaves the landmark range

<p align="center">
<img width="425" alt="Screenshot 2023-09-28 at 7 09 14 PM" src="https://github.com/Panjete/radarStateEstimator/assets/103451209/48089cf6-f998-4924-8fa2-2b6bada22815">
</p>

Experimenting with noise shows it's effect on estimation accuracy in the range of a Landmark

<p align="center">
<img width="541" alt="Screenshot 2023-09-28 at 7 09 30 PM" src="https://github.com/Panjete/radarStateEstimator/assets/103451209/332e43b3-64d9-4064-b67d-5c3c7c23190f">
</p>


