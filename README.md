# Calibrated Model-Based Deep Reinforcement Learning



**Abstract:** Accurate estimates of predictive uncertainty are important for building effective model-based reinforcement learning agents. However, predictive uncertainties --- especially ones derived from modern neural networks --- are often inaccurate and impose a bottleneck on performance. Here, we argue that ideal model uncertainties should be calibrated, i.e. their probabilities should match empirical frequencies of predicted events. We describe a simple way to augment any model-based reinforcement learning algorithm with calibrated uncertainties and show that doing so consistently improves the accuracy of planning and also helps agents to more effectively trade off exploration and exploitation. On the HalfCheetah MuJoCo task, our system achieves state-of-the-art performance using 50% fewer samples than the current leading approach.

Our findings suggest that calibration can improve the performance and sample complexity of model-based reinforcement learning with minimal computational and implementation overhead.

## Citations

Code based heavily on https://github.com/kchua/handful-of-trials/. For running instructions and configuration arguments, see that repo.


## Running Experiments

Experiments for a particular environment can be run using:

```
python scripts/mbexp.py
    -env    ENV       (required) The name of the environment. Select from
                                 [cartpole, reacher, pusher, halfcheetah].
    -ca     CTRL_ARG  (optional) The arguments for the controller
                                 (see section below on controller arguments).
    -o      OVERRIDE  (optional) Overrides to default parameters
                                 (see section in code [repo](https://github.com/kchua/handful-of-trials/) on overrides).
    -logdir LOGDIR    (optional) Directory to which results will be logged (default: ./log)
    
    -calibrate        (optional) Enables calibration in experiment. By default, calibration is disabled.
```

Example command: `python scripts/mbexp.py -env cartpole -ca model-type PE -ca prop-type DS -calibrate`

