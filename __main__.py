import argparse
import os
import platform
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

# The following functions work on all platforms
from iScatter.functions import iscattering
from .generate_conditions import generate_conditions
from .analyse_test import analyse_output
# Supress warning from pyscf about B3lYP since we are using semi-empircal
if platform.system() != 'Windows':
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        from .mindo_pyscf_hess import calculate_hessian
        from .rundyn import calculate_at_timesteps

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
parser.add_argument('--seed', type=int, nargs=1, default=[-1], help='Seed to use for MD')
args=parser.parse_args()

#If the user is calculating hessian values
if args.hessian:
    if platform.system() != 'Windows':
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
    else:
        print('This function is not windows compatible!')
        exit()

#For anything else create scatter object
# instantiate scatter object
sc = iscattering.iscatter(samples = args.traj[0], seed = args.seed[0])
# read input file as argument ./test.py input
sc.ReadInput(args.filename[0])

if args.generate: 
    sc = generate_conditions(sc, args.traj[0])

if args.calcsteps:
    if platform.system() != 'Windows':
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
    else:
        print('This function is not windows compatible!')
        exit()

if args.analyse:
    analyse_output(sc, args.traj[0])