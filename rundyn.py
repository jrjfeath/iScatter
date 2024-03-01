#!/usr/bin/env python3
import numpy as np
from pyscf import gto, md, semiempirical

fmt2au, au2fmt = 41.3413745, 1.0 / 41.3413745
ang2au, au2ang = 1.88973, 1.0 / 1.88973
ev2au, au2ev = 0.0367494, 1.0 / 0.0367494

def calculate_at_timesteps(filename,time_step=20,step_count=10):
    velocs = {}
    def getvel(local):
        print('time = ', local['self'].time)
        velocs[local['self'].time] = local['self'].veloc.copy()

    mol = gto.Mole()
    mol.basis = 'sto-3g'
    mol.atom = filename
    mol.charge = 0
    mol.spin = 1
    mol.symmetry = False
    try: mol.build()
    except RuntimeError:
        mol.spin = 2
        mol.build()
    mold = semiempirical.MINDO3(mol)
    init_veloc = np.array([[float(f) for f in ln.strip().split()[1:]] for ln in open(f'{filename[:-4]}.vel','r').readlines()[2:]])*ang2au/fmt2au
    myscanner = mold.nuc_grad_method().as_scanner()
    myintegrator = md.NVE(myscanner,
        dt=time_step,
        steps=step_count,
        veloc=init_veloc,
        callback=getvel,
        data_output=f'{filename[:-4]}.md.data',
        trajectory_output=f'{filename[:-4]}.md.xyz'
    ).run()
    en0 = float(open('../ref.en','r').readline().strip())
    lins = [ln.strip() for ln in open(f'{filename[:-4]}.md.data','r').readlines()]
    out = [lins[0]+'\n']
    for ln in lins[1:]:
        lns = ln.strip().split()
        st = lns[0] 
        nn = [float(f) for f in lns[1:]]
        nn[0] += -en0
        nn[2] += -en0
        nn = np.array(nn)*au2ev
        st += ''.join([  "{0:13.9f}".format(n).rjust(20) for n in nn ]) +'\n'
        out.append(st)
    open(f'{filename[:-4]}.dat','w').writelines(out)
    na = len(mol._atom)
    el = [e[0] for e in  mol._atom] 
    out  = []
    for time in velocs.keys():
        out += [ str(na) + '\n']
        out += [ 'Time = ' + str(time) + '\n' ]
        for i, vv in enumerate(velocs[time]):
            out += [el[i] + '  ' +  ''.join(["{0:13.9f}".format(v*au2ang/au2fmt) for v in vv ]) + '\n']
    open(f'{filename[:-4]}.md.vel','w').writelines(out)
