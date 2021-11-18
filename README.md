# CBF
Rendezvous Consensus Problem with Double Integrator Control Barrier Function Simulation


## Installing Environment
Conda is strongly recommended for easier environment management as well as better support for Apple Silicon as of Nov 2021.
```
conda create -n cbf
```

activate the environment.

```
conda activate cbf
```

install dependencies
```
conda install -y matplotlib cvxopt scipy
```

## Running the example program
```
conda activate cbf
cd robotarium_python_simulator
python obstacle_avoiding_consensus.py 