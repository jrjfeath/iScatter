import argparse
import os
import sys
import warnings
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

#Supress warning from pyscf about B3lYP since we are using semi-empircal
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from iScatter.functions import iscattering
    from .mindo_pyscf_hess import calculate_hessian
    from .generate_conditions import generate_conditions
    from .rundyn import calculate_at_timesteps
    from .analyse_test import analyse_output

parser=argparse.ArgumentParser(
    prog='IScatter',
    description='''Program to generate trajectory files for use in various programs.''',
    epilog="""""")
parser.add_argument('filename', metavar='Filename', type=str, nargs=1, help='Input filename')
parser.add_argument('-v','--version', action='version', version='%(prog)s 1.0')
parser.add_argument('-he','--hessian', action="count", help='Calculates hessian values from input.xyz, to be run on its own.')
parser.add_argument('-g','--generate', action="count", help='Generate initial conditions from input')
parser.add_argument('-cs','--calcsteps', action="count", help='Run dynamics on generated trajectories with PYSCF')
parser.add_argument('-a','--analyse', action="count", help='Verify the generated MD data from PYSCF')
parser.add_argument('--traj', type=int, nargs=1, default=[10], help='Number of trajectories (10 default)')
parser.add_argument('--time', type=int, nargs=1, default=[20], help='Time step (20 fs default) for MD')
parser.add_argument('--steps', type=int, nargs=1, default=[10], help='Number of steps (10 default) for MD')
args=parser.parse_args()

#If the user is calculating hessian values
if args.hessian: 
    print('Calculating hessians, please wait.')
    with suppress_stdout():
        result = calculate_hessian(args.filename[0])
    if result == 0:
        print('Finished Calculating Hessians.')
    if result == 1:
        print('The .xyz file provided seems to contain incorrect syntax.')
    if result == 2:
        print(f'Cannot find the specified file {args.filename[0]}')
    exit()

#For anything else create scatter object
# instantiate scatter object
sc = iscattering.iscatter()
# read input file as argument ./test.py input
sc.ReadInput(args.filename[0])

if args.generate: 
    output_name = generate_conditions(sc, args.traj[0])

if args.calcsteps:
    #PYSCF must been run in the same directory as the files
    os.chdir("outputs")
    for i in range(args.traj[0]):
        if not os.path.isfile(f'{sc.fileout}_{i}.xyz'):
            print(f'Trajectory {i+1} failed, {sc.fileout}_{i}.xyz does not exist?')
            continue
        with suppress_stdout():
            calculate_at_timesteps(f'{sc.fileout}_{i}.xyz',args.time[0],args.steps[0])
        print(f'Finished calculating trajectory: {i+1} of {args.traj[0]}')
    os.chdir("..")

if args.analyse:
    analyse_output(sc, args.traj[0])