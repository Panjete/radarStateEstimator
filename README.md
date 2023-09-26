## radarStateEstimator

### State Estimation using Kalman Filters

Models Noisy Airplane Observation Data, predicting it's trajectory and maintaining a belief state of it's  positional and velocity co-ordinates in 3 dimensions.
I experiment with variational noise parameters, with an analysis of the effect of positional update, velocity update, and sensor noise.
The motion models and observation models are implemented, and the simulation is made robust enough to propagate belief even when some sensor drops off it's observations.

