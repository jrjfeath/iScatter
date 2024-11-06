#!/usr/bin/env python3
import sys
import numpy as np
from tqdm import tqdm

def step(xx, oe, Fx, par, domain, dl):
    """
    Perform a Metropolis-Hastings Monte Carlo step.

    Parameters:
    xx (list): Current state of the system.
    oe (float): Energy of the current state.
    Fx (function): Energy function to evaluate the energy of a state.
    par (tuple): Parameters required by the energy function Fx.
    domain (list or None): Domain constraints for each dimension, or None if no constraints.
    dl (list): Maximum displacement for each dimension during the step.

    Returns:
    list: New state of the system.
    float: New energy of the system.
    list: Mole fractions of particles moved during the step.
    """
    debug = True
    debug = False
    ne, k = 0.0, 0
    while (k < 10000 and ne <= 1.0e-15):
        mox = np.zeros(len(xx))
        nx = np.copy(xx)
        while sum(abs(mox)) <= 0.0:
            for xi in range(len(xx)):
                if np.random.random() > 0.60:
                    mox[xi] += 1
                    nx[xi] += (np.random.random() - 0.5) * dl[xi]
        nx = putwithindomain(nx, xx, dl, domain)
        ne = Fx(nx, *par)
        if debug:
            print('pnx = ', nx, (np.random.random() - 0.5) * 2 * dl[xi], 'PAR = ', par, 'NE = ', ne)
        k += 1
    if k >= 10000:
        print(oe, xx, ne, nx)
        quit('could not find positive values')
    if np.isnan(ne) or np.isinf(ne):
        print(nx, ne)
        quit('ERROR in MC: ')
    return nx, ne, mox

def putwithindomain(nx, xx, dl, domain):
    """
    Ensure that a proposed state remains within the specified domain constraints.

    Parameters:
    nx (list): Proposed state.
    xx (list): Current state.
    dl (list): Maximum displacement for each dimension during a step.
    domain (list or None): Domain constraints for each dimension, or None if no constraints.

    Returns:
    list: New state after applying domain constraints.
    """
    if not domain is None:
        for xi in range(len(nx)):
            if nx[xi] < domain[xi][0]:
                if domain[xi][2]:
                    nx[xi] = domain[xi][1] + nx[xi]
                else:
                    nx[xi] = np.random.random() * xx[xi]
            elif nx[xi] > domain[xi][1]:
                if domain[xi][2]:
                    nx[xi] = domain[xi][0] + nx[xi] - domain[xi][1]
                else:
                    nx[xi] = xx[xi] + np.random.random() * (domain[xi][1] - xx[xi])
    return nx

def MCstep(xx, Fx, par, dl, *oes, **dic):
    """
    Perform a Metropolis-Hastings Monte Carlo step for the system.

    Parameters:
    xx (list): Current state of the system.
    Fx (function): Energy function to evaluate the energy of a state.
    par (tuple): Parameters required by the energy function Fx.
    dl (list): Maximum displacement for each dimension during a step.
    oes (float): Optional initial energy value (default is evaluated from Fx).
    dic (dict): Additional parameters, including 'domains' for domain constraints.

    Returns:
    list: New state of the system.
    float: New energy of the system.
    bool: True if the step was accepted, False otherwise.
    list: Mole fractions of particles moved during the step.
    """
    if 'domains' in dic.keys():
        domain = dic['domains']
    else:
        domain = None
    if len(oes) > 0:
        oe = oes[0]
    else:
        oe = Fx(xx, *par)
    nx, ne, mox = step(xx, oe, Fx, par, domain, dl)
    if ne / (oe + 1.0e-15) > 1.0:
        pas = True
    else:
        if ne / (oe + 1.0e-15) > np.random.random():
            pas = True
        else:
            pas = False
            ne = oe
            nx = xx
    return nx, ne, pas, mox
def EstimateCorr(x0, pfr, Fx, par, dl, **dic):
    """
    Estimate the decorrelation time of a Metropolis-Hastings Markov Chain.

    Parameters:
    x0 (list): Initial state of the Markov Chain.
    pfr (float): Target success ratio.
    Fx (function): Energy function to evaluate the energy of a state.
    par (tuple): Parameters required by the energy function Fx.
    dl (float): Initial maximum displacement during a step.
    dic (dict): Additional parameters, including 'domains' for domain constraints.

    Returns:
    int: Estimated decorrelation time for the Markov Chain.
    """
    if 'domains' in dic.keys():
        domain = dic['domains']
    else:
        domain = None
    xx = x0
    dim = len(x0)
    oe = Fx(xx, *par)
    xx, oe, mox = step(xx, oe, Fx, par, domain, dl)
    nc = 5
    cn = 50
    cc = [0.0 for i in range(cn)]
    oo = np.zeros(cn)
    kk = 0
    for n in range(1, 100000):
        xx, oe, pas, mox = MCstep(xx, Fx, par, dl, oe, domains=domain)
        if n % nc == 0:
            cc.pop()
            cc = [xx] + cc
            if n / nc > cn:
                kk += 1
                for i in range(cn):
                    oo[i] += np.dot(cc[0], cc[i])
    oo = oo / float(kk)
    ndecor = 50
    if oo[0] > 0.0:
        oo = oo / oo[0]
    for i in range(cn):
        if oo[i] < 0.01:
            ndecor = i * nc
            break
    return ndecor

def EstimateDl(x0, pfr, Fx, par, idl, **dic):
    """
    Estimate the optimal step size for a Metropolis-Hastings Markov Chain.

    Parameters:
    x0 (list): Initial state of the Markov Chain.
    pfr (float): Target success ratio.
    Fx (function): Energy function to evaluate the energy of a state.
    par (tuple): Parameters required by the energy function Fx.
    idl (float): Initial maximum displacement during a step.
    dic (dict): Additional parameters, including 'domains' for domain constraints.

    Returns:
    list: Estimated step size for each dimension.
    """
    debug = True
    debug = False
    if 'domains' in dic.keys():
        domain = dic['domains']
    else:
        domain = None
    xx = x0
    dim = len(x0)
    dl = np.array([idl for i in range(dim)])
    ps, fl = [0 for i in range(dim)], [0 for i in range(dim)]
    oe = Fx(xx, *par)
    xx, oe, mox = step(xx, oe, Fx, par, domain, dl)
    n = 1
    jj = 350
    cc = 0
    while True:
        xx, oe, pas, mox = MCstep(xx, Fx, par, dl, oe, domains=domain)
        n += 1
        if pas:
            ps += mox
        else:
            fl += mox
        if n > 10000000:
            break
        if n % jj == 0:
            rr = np.zeros(dim)
            ndd = 0
            for d in range(dim):
                rr[d] = float(ps[d]) / (float(ps[d] + fl[d]))
                if rr[d] < pfr - pfr * 0.1:
                    dl[d] += -dl[d] * 0.05
                elif rr[d] > pfr + pfr * 0.1:
                    dl[d] += +dl[d] * 0.05
                else:
                    ndd += 1
            if ndd == dim:
                break
            if not domain is None:
                dobreak = True
                for d in range(dim):
                    if dl[d] < domain[d][1] * 3:
                        dobreak = False
                if dobreak:
                    break
            if n > jj * 2:
                imp = abs(pfr * np.ones(dim) - prr) - abs(pfr * np.ones(dim) - rr)
                if all([i > 0.0 for i in imp]):
                    cc += 1
                    if cc > 3:
                        dl = dl * 1.2
                        break
            prr = rr.copy()
            if debug:
                print('ps fl =', ps, fl, rr, pfr - pfr * 0.1, pfr + pfr * 0.1, 'dl = ', dl)
    if debug:
        print('DL = ', dl)
        print('ps fl = ', ps, fl, rr)
        print('PER = ', ps[0] / (ps[0] + fl[0]))
    return dl

def SampleMC(nsamp, x0, Fx, par, **dic):
    """
    Sample states using a Metropolis-Hastings Markov Chain Monte Carlo.

    Parameters:
    nsamp (int): Number of samples to generate.
    x0 (list): Initial state of the Markov Chain.
    Fx (function): Energy function to evaluate the energy of a state.
    par (tuple): Parameters required by the energy function Fx.
    dic (dict): Additional parameters, including 'domains' for domain constraints and 'idel' for initial step size.

    Returns:
    list: Sampled states.
    list: Corresponding energies of the sampled states.
    """
    debug = True
    debug = False
    if 'domains' in dic.keys():
        domain = dic['domains']
    else:
        domain = None
    if 'idel' in dic.keys():
        idel = dic['idel']
    else:
        idel = 1.0
    xx = np.copy(x0)
    dim = len(x0)
    dl = EstimateDl(xx, 0.5, Fx, par, idel, domains=domain)
    if debug:
        print('DL = ', dl)
    ndecor = EstimateCorr(xx, 0.5, Fx, par, dl, domains=domain)
    if debug:
        print('NDECOR = ', ndecor)
    niter = ndecor * (nsamp + 1)
    ff = int(niter / nsamp) - 1
    ps, fl = [0 for i in range(dim)], [0 for i in range(dim)]
    xx = np.copy(x0)
    oe = Fx(xx, *par)
    mox = np.zeros(len(xx))
    xx, oe, mox = step(xx, oe, Fx, par, domain, dl)
    xout, eout = [], []
    if niter == 0: return xout, eout
    for i in tqdm(range(1, niter), file=sys.stdout):
        xxx, ooe, pas, mox = MCstep(xx, Fx, par, dl, oe, domains=domain)
        xx, oe = xxx.copy(), ooe
        if i % ff == 0:
            xout.append(xx)
            eout.append(oe)
            if len(xout) == nsamp:
                break
        if pas:
            ps += mox
        else:
            fl += mox
    if debug:
        print('PAS = ', ps, 'FAIL = ', fl)
    return xout, eout

