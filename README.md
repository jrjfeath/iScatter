# iScatter
Program to generate trajectory files for use in various programs.
Can be called from the command line by dropping package in site-packages.

Has the following requirements:
- python 3.6+
- numpy
- geometric
- pyscf
- pyscf-semiempirical
- pyscf-qsdopt

Program options:
- 'filename' 
- '-v','--version'
- '-he','--hessian', description='Calculates hessian values from input.xyz, to be run on its own.'
- '-g','--generate', description='Generate initial conditions from input'
- '-cs','--calcsteps', description='Run dynamics on generated trajectories with PYSCF'
- '-a','--analyse', description='Verify the generated MD data from PYSCF'
- '--traj', description='Number of trajectories (10 default)'
- '--time', description='Time step (20 fs default) for MD'
- '--steps', description='Number of steps (10 default) for MD'

Example Usage:
Calculate the hessians for water
- python -m iScatter h2om_geom.xyz -he
Calculate the hessians for hydroxyl
- python -m iScatter ohm_geom.xyz -he
Calculate 20 trajectories for the surface
- python -m iScatter input -g --traj 20
Calculate the dynamics using pyscf
- python -m iScatter input -cs --time 25 --steps 20
