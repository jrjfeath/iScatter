#!/usr/bin/env python3
import numpy as np
from .constants import kboltz

def VibPartFunc(n, w, T, izpe):
    """
    Calculate the vibrational partition function.

    Parameters:
    n (array): Array of quantum numbers for vibrational states.
    w (float): Vibrational frequency.
    T (float): Temperature.
    izpe (Bool): Flag whether or not to include ZPE.

    Returns:
    float: The vibrational partition function.
    """
    nn = np.array([int(np.round(na)) for na in n])
    for n in nn:
        if n < 0:
            return 0.0
    if izpe:
     ee = w * (nn + 0.5)
    else:
     ee = w * nn 
    return np.product(np.exp(-ee / (kboltz * T)))

def MaxwellBoltzmann(v, T, m):
    """
    Calculate the Maxwell-Boltzmann distribution.

    Parameters:
    v (float): Velocity.
    T (float): Temperature.
    m (float): Mass.

    Returns:
    float: The Maxwell-Boltzmann distribution value.
    """
    #print('v = ', v, 'm = ', m, 'T = ', T, 'kb = ', kboltz, ' RETURN = ', v**2 * exp(-m * v**2 / (2 * kboltz * T)) )
    return v**2 * np.exp(-m * v**2 / (2 * kboltz * T))

def GaussianF(x, x0, fwhm):
    """
    Gaussian function.

    Parameters:
    x (float): Input value.
    x0 (float): Mean of Gaussian
    fwhm (float): Full-width half maximum

    Returns:
    float: Gaussian function value.
    """
    sigma = fwhm/(2*np.sqrt(2.0*np.log(2.0)))
    a = 1.0/(2*sigma**2)
    return np.exp(-a * (x-x0)**2)

def IPDist(b, MaxB):
    """
    Calculate the Impact Parameter Distribution.

    Parameters:
    b (float): Impact parameter.
    MaxB (float): Maximum impact parameter.

    Returns:
    float: The Impact Parameter Distribution value.
    """
    if abs(b) > MaxB:
        return 0.0
    else:
        return abs(b)

def HarmWigner(QP, n):
    """
    Calculate the Wigner distribution for harmonic oscillators.

    Parameters:
    QP (tuple): Tuple containing Q and P coordinates.
    n (int): Quantum number.

    Returns:
    float: Wigner distribution value.
    """
    Q, P = QP
    def facfac_loop(n):
        yield 0, 1.
        r = 1.
        for m in range(1, n + 1):
            r *= float(n - m + 1) / m**2
            yield m, r
    def ana_laguerre(n, x):
        total = 0.
        for m, r in facfac_loop(n):
            entry = (-1.)**m * r * x**m
            total += entry
        return total
    if n == 0:
        return np.exp(-Q**2) * np.exp(-P**2)
    else:
        rhosquare = 2.0 * (P**2 + Q**2)
        W = (-1.0)**n * ana_laguerre(n, rhosquare) * np.exp(-rhosquare / 2.0)
        return max([W, 0.0])

def Boltzmann(E, T):
    """
    Calculate the Boltzmann distribution.

    Parameters:
    E (float): Energy.
    T (float): Temperature.

    Returns:
    float: The Boltzmann distribution value.
    """
    return np.exp(-E / (kboltz * T))

def REBoltzmann(xi, EE, JJ, T):
    """
    Calculate the Rotational Energy Boltzmann distribution.

    Parameters:
    xi (float): Quantum number.
    EE (list): List of energy levels.
    JJ (list): List of angular momentum quantum numbers.
    T (float): Temperature.

    Returns:
    float: The Rotational Energy Boltzmann distribution value.
    """
    ii = int(np.round(xi))
    if ii < 0:
        return 0.0
    elif ii >= len(EE):
        return 0.0
    else:
        E, sigma = EE[ii], JJ[ii] * 2 + 1
        return sigma * Boltzmann(E, T)

def EulerSurface(angs):
    """
    Calculate the Euler surface.

    Parameters:
    angs (list): List of angles.

    Returns:
    float: The Euler surface value.
    """
    if angs[1] < 0.0 or angs[1] > np.pi:
        return 0.0
    if angs[1] < 0.0 or angs[1] > np.pi:
        return 0.0
    for i in [0, 2]:
        if angs[i] < -np.pi or angs[i] > np.pi:
            return 0.0
        if angs[i] < -np.pi or angs[i] > np.pi:
            return 0.0
    return np.sin(angs[1])

def EulerPolarSurface(angs, alpha, beta):
    """
    Calculate the Euler Polar surface.

    Parameters:
    angs (list): List of angles.
    alpha (float): Parameter.
    beta (float): Parameter.

    Returns:
    float: The Euler Polar surface value.
    """
    if angs[1] < 0.0 or angs[1] > np.pi:
        return 0.0
    if angs[1] < 0.0 or angs[1] > np.pi:
        return 0.0
    for i in [0, 2]:
        if angs[i] < -np.pi or angs[i] > np.pi:
            return 0.0
        if angs[i] < -np.pi or angs[i] > np.pi:
            return 0.0
    return np.sin(angs[1]) * 0.5 * (1 - alpha * beta * np.cos(angs[1]))

