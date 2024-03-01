#!/usr/bin/env python3
import os
import numpy as np
from pyscf import gto, semiempirical
from pyscf.qsdopt.hesstools import filter_hessian
from pyscf.geomopt.geometric_solver import optimize as geometric_opt

def central_differences_hess(mol, g_scanner):
    """Evaluate numerical hessian of the energy using central differences."""
    delta = 5e-5
    fourdelta = 4 * delta
    geom = mol.atom_coords()
    nat = geom.shape[0]
    ndim = 3 * nat
    H = np.zeros([ndim, ndim])

    for iat in range(nat):
        for icoor in range(3):
            i = 3 * iat + icoor
            _geom = geom.copy()
            _geom[iat, icoor] = geom[iat, icoor] + delta
            mol.set_geom_(_geom, unit="Bohr")
            e1, g1 = g_scanner(mol)

            _geom[iat, icoor] = geom[iat, icoor] - delta
            mol.set_geom_(_geom, unit="Bohr")
            e2, g2 = g_scanner(mol)

            H[i, :] = (g1 - g2).reshape(-1)

    H = (H + H.T) / fourdelta
    return H

def calculate_hessian(filename):
    if not os.path.isfile(filename):
        return 2
    
    conv_params = { # These are the default settings
        'convergence_energy': 2e-7,  # Eh
        'convergence_grms': 1e-5,    # Eh/Bohr
        'convergence_gmax': 1.0e-4,  # Eh/Bohr
        'convergence_drms': 5.0e-4,  # Angstrom
        'convergence_dmax': 5.0e-4,  # Angstrom
    }
    amu2au = 1837.1527
    au2cm = 219474.63
    au2ang = 1.0/1.88973
    mol = gto.Mole()
    mol.basis = 'sto-3g'
    mol.atom = filename
    mol.charge = 0
    mol.spin = 1
    #mol.symmetry = False
    mol.symmetry = True
    try: mol.build()
    except SyntaxError:
        return 1
    except RuntimeError:
        mol.spin = 2
        mol.build()
    moldft = semiempirical.MINDO3(mol)
    mol2 = geometric_opt(moldft,maxsteps=1000,**conv_params)
    mol2.symmetry = True
    myscan = semiempirical.MINDO3(mol2).nuc_grad_method().as_scanner()
    energy, g0 = myscan(myscan.mol)
    g0 = g0.flatten()
    HH = central_differences_hess(mol2, myscan)
    mass = mol2.atom_mass_list()*amu2au
    atom_coords = mol2.atom_coords()
    mass_center = np.einsum('z,zx->x', mass, atom_coords) / mass.sum()
    atom_coords = atom_coords - mass_center
    natm = atom_coords.shape[0]
    sm = np.sqrt(mass)
    sm3 = np.repeat(sm, 3)
    #before applying the filter, frequencies are fairly highish
    HH = filter_hessian(myscan.mol, HH)
    HM = np.einsum("ij, i, j -> ij", HH, 1 / sm3, 1 / sm3)
    eig, ev = np.linalg.eigh(HM)
    freq = np.sqrt(abs(eig))
    for i, e in enumerate(freq):
        print(i, '{0:15.10f}'.format(e*au2cm))
        open(filename[:-4]+'_freq.txt','w').writelines([ "{:.10e}".format(e)+'\n' for e in freq])
    np.savetxt(filename[:-4]+'_hessian.txt',HM)

    for atm in mol2._atom:
        print(atm[0], atm[1])
        open(filename[:-4]+'_geom.xyz','w').writelines([str(natm)+'\n',' \n']+[atm[0] + ' ' + ''.join(['{0:15.10f}'.format(e*au2ang)+'  ' for e in atm[1]]) +'\n' for atm in mol2._atom])
    return 0