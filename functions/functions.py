#!/usr/bin/env python3
import numpy as np
from numpy.linalg import norm, eigh, det, svd
from numpy import reshape, matmul, array, zeros, identity
from numpy import random, cross, diag
from numpy import cos, sin, arccos, arctan2, sqrt
from math  import acos, floor, ceil
from .constants import el2m, tpi, hpi, pi, x, y, z

np.set_printoptions(suppress=True)

def rot_match_vec(r1, r2):
    """
    Rotate the vector r1 to match the direction of vector r2.

    Parameters:
    r1 (numpy.ndarray): The input vector to be rotated.
    r2 (numpy.ndarray): The target vector to match.

    Returns:
    numpy.ndarray: The rotated vector matching the direction of r2.
    numpy.ndarray: The rotation matrix used for the transformation.
    """
    U, s, V = svd(matmul(r1.T, r2))
    if det(matmul(U, V.T)) < 0:
        ON = identity(3)
        ON[2, 2] = -1
        RO = matmul(U, matmul(ON, V)).T
    else:
        RO = matmul(U, V).T
    return matmul(RO, r1.T).T, RO


# from the body frame we have : Ix w_x^2 + Iy w_y^2 + Iz w_z^2 = 2*E
# defines a ellipsoid:
#   1) w_z = sqrt(2E/Iz) cos(theta)
#   2) w_x = sqrt(2E/Ix) sin(theta) * sin(phi)
#   3) w_y = sqrt(2E/Iy) sin(theta) * cos(phi)
#   we know E and w_z = Lz/Iz, so theta = cos-1(w_z sqrt(Iz/2E)), uniform sample of phi and use 2) and 3) to calculate w_x and w_y.
# returns the classical angular momentum vector.
# note that even in the ekart frame, we are still using the energy obtained from the equlibrium geometry.
def ClassicalAngularMomentum(Ib, Lz, E, **dic):
    """
    Calculate the classical angular momentum vector for a rotating body.

    Parameters:
    Ib (numpy.ndarray): A numpy array containing the moments of inertia along the principal axes.
    Lz (float): The z-component of angular momentum.
    E (float): The total energy of the system.
    dic (dict): Additional keyword arguments, including 'phi' for specifying the azimuthal angle.

    Returns:
    numpy.ndarray: The classical angular momentum vector.
    """
    if "phi" in dic.keys():
        phi = dic["phi"]
    else:
        phi = tpi * random.random()
    Ix, Iy, Iz = Ib
    if abs(Iz) < 1.0e-8:
        wz = 0.0
        theta = hpi
    else:
        wz = Lz / Iz
        if E > 0.0:
            dm = wz * sqrt(Iz / (2 * E))
            if dm < -1.0 or dm > 1.0:
                print('WARNING, inverse cos not numerically right: is this |dm| a lot greater than 1? :',dm)
                dm = 1.0*(dm/abs(dm))
            theta = acos(dm)
        else:
            theta = 0.0
    wx = sqrt(2 * E / Ix) * sin(theta) * cos(phi)
    wy = sqrt(2 * E / Iy) * sin(theta) * sin(phi)
    return array([wx, wy, wz]) * Ib


def polar2xyz(R, psi, theta):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters:
    R (float): The radial distance.
    psi (float): The azimuthal angle.
    theta (float): The polar angle.

    Returns:
    numpy.ndarray: The Cartesian coordinates (x, y, z) corresponding to the given spherical coordinates.
    """
    zi = R * cos(theta)
    xi = R * sin(theta) * cos(psi)
    yi = R * sin(theta) * sin(psi)
    return array([xi, yi, zi])


def GetRotTransVec(xx0, ms, el):
    """
    Calculate the rotation and translation vectors for a set of coordinates.

    Parameters:
    xx0 (numpy.ndarray): The input coordinates.
    ms (numpy.ndarray): The masses of the atoms.
    el (list): The list of element symbols.

    Returns:
    numpy.ndarray: The rotation and translation vectors.
    """
    x0 = xx0 - COM(xx0, ms).T
    # need to work in ekart frame
    RR, Ibm, Is = iI(iX(x0), iX(x0), ms)
    x0 = matmul(RR.T, x0.T).T
    na = x0.shape[0]
    ve = []
    for j, d in enumerate([x, y, z]):
        rr = cross(d, x0)
        ve.append(rr)
    for j, d in enumerate([x, y, z]):
        tt = zeros((na, 3)) + d
        ve.append(tt)
    vee = []
    for v in ve:
        # move back to origical frame
        vi = matmul(RR, v.T).T
        # mass scale and normalize
        vi = reshape(mscale(vi, ms, 1), (na * 3,))
        if norm(vi) > 0.0001:
            vi = vi / norm(vi)
            vee.append(vi.tolist())
    ve = vee
    debug = False
    # debug = True
    if debug:
        out = []
        for v in ve:
            v1 = mscale(reshape(array(v), (na, 3)), ms, -1)
            v1 = 4 * v1 / norm(v1)
            out += XYZlist(el, np.concatenate((xx0.T, v1.T)).T)
        open("6mod.xyz", "w").writelines(out)
    ve = array(ve)
    # debug=True
    if debug:
        print("OVER = ")
        print(matmul(ve, ve.T))
    return ve


# to go from UN-mass-scaled to frequency mass scaled normal modes...
def ScaleTransform(x2n, n2x, freq, mass, dirr, fscale):
    """
    Scale transformation matrices for mass-scaled normal modes.

    Parameters:
    x2n (numpy.ndarray): Transformation matrix from Cartesian to normal mode coordinates.
    n2x (numpy.ndarray): Transformation matrix from normal mode to Cartesian coordinates.
    freq (list): List of frequencies.
    mass (list): List of atomic masses.
    dirr (bool): Direction of scaling (True for scaling up, False for scaling down).
    fscale (float): Scaling factor.

    Returns:
    numpy.ndarray: Scaled transformation matrix for Cartesian to normal mode coordinates.
    numpy.ndarray: Scaled transformation matrix for normal mode to Cartesian coordinates.
    """
    natoms = len(mass)
    nm     = n2x.shape[0]
    x2n2 = x2n.copy()
    n2x2 = n2x.copy()
    ws = abs(array(freq))
    for i in range(len(ws)):
        if abs(ws[i]) < 1.0e-8:
            ws[i] = 1.0
    if dirr:
        for i in range(nm):
            for j in range(natoms * 3):
                x2n2[j, i] = x2n2[j, i] * (sqrt(ws[i] ** fscale * mass[int(j / 3)]))
                n2x2[i, j] = n2x2[i, j] / (sqrt(ws[i] ** fscale * mass[int(j / 3)]))
    else:
        for i in range(nm):
            for j in range(natoms * 3):
                x2n2[j, i] = x2n2[j, i] / (sqrt(ws[i] ** fscale * mass[int(j / 3)]))
                n2x2[i, j] = n2x2[i, j] * (sqrt(ws[i] ** fscale * mass[int(j / 3)]))
    return x2n2, n2x2


def el2Mass(el):
    """
    Map element symbols to atomic masses.

    Parameters:
    el (list): List of element symbols.

    Returns:
    numpy.ndarray: List of atomic masses corresponding to the element symbols.
    """
    return array([el2m[e] for e in el])


def COM(r, mass):
    """
    Calculate the center of mass of a system.

    Parameters:
    r (numpy.ndarray): The array of coordinates.
    mass (numpy.ndarray): The masses of the atoms.

    Returns:
    numpy.ndarray: The center of mass coordinates.
    """
    return np.sum(r.T * mass, axis=1, keepdims=True) / sum(mass)


def ReadXYZ(filn):
    """
    Read atomic coordinates from an XYZ file.

    Parameters:
    filn (str): The path to the XYZ file.

    Returns:
    list: List of element symbols.
    numpy.ndarray: Array of atomic coordinates.
    """
    dat = open(filn, "r").readlines()
    na = int(dat[0].strip())
    xcoo = []
    el = []
    for i in range(2, na + 2):
        lns = dat[i].strip().split()
        el.append(lns[0])
        xcoo.append([float(f) for f in lns[1:]])
    return el, array(xcoo)


def ReadXYZs(filn):
    """
    Read XYZ coordinate data from a file and return element symbols and atomic positions.

    Parameters:
    filn (str): The path to the XYZ file.

    Returns:
    tuple: A tuple containing two elements:
        - el (list): A list of element symbols in the same order as the atomic positions.
        - xcoos (list): A list of NumPy arrays, each containing the atomic coordinates for a molecule.
        - mess (list): A list of messages in the description part of the data

    Example:
    el, xcoos, mess = read_xyzs('molecule.xyz')
    """
    dat = open(filn, 'r').readlines()
    na = int(dat[0].strip())
    xcoos, xx, ell = [], [], []
    mess = []
    for ln in [la.strip() for la in dat]:
        lns = ln.split()
        if len(lns) == 1:
            if len(xx) > 0:
                if len(xcoos) == 0:
                    el = ell.copy()
                xcoos.append(np.array(xx))
            xx = []
        elif len(lns) == 4:
            xx.append([float(ff) for ff in lns[1:]])
            ell.append(lns[0])
        else:
            mess.append(ln)
    if len(xx) > 0:
        xcoos.append(np.array(xx))

    return el, xcoos, mess

def XYZlist(el, xcoo, **dic):
    """
    Generate an XYZ format file from element symbols and atomic coordinates.

    Parameters:
    el (list): List of element symbols.
    xcoo (numpy.ndarray): Array of atomic coordinates.
    dic (dict): Additional keyword arguments, including 'mess' for a message in the file.

    Returns:
    list: List of strings containing the XYZ file content.
    """
    if "mess" in dic.keys():
        message = dic["mess"]
    else:
        message = " "
    na, ni = xcoo.shape
    xout = [str(na) + "\n"]
    xout.append(message + "\n")
    xout += [
        el[i]
        + " "
        + "".join(["{0:14.9f}".format(xcoo[i, j]) + " " for j in range(ni)])
        + "\n"
        for i in range(na)
    ]
    return xout


def mscale(mat, mass, dr):
    """
    Mass-scale the given matrix.

    Parameters:
    mat (numpy.ndarray): The input matrix.
    mass (numpy.ndarray): The masses of the atoms.
    dr (int): Scaling direction (1 for scaling up, -1 for scaling down).

    Returns:
    numpy.ndarray: The mass-scaled matrix.
    """
    na = mat.shape[0]
    mo = mat.copy()
    for i in range(na):
        if dr == 1:
            mo[i, :] = mo[i, :] * sqrt(mass[i])
        else:
            mo[i, :] = mo[i, :] / sqrt(mass[i])
    return mo


# mass scale the coordinates as 1d tensor.
def mscale2(vi, mass, dr):
    """
    Mass-scale a 1D tensor of coordinates.

    Parameters:
    vi (numpy.ndarray): The 1D tensor of coordinates.
    mass (numpy.ndarray): The masses of the corresponding atoms.
    dr (int): Scaling direction (1 for scaling up, -1 for scaling down).

    Returns:
    numpy.ndarray: The mass-scaled 1D tensor of coordinates.
    """
    mo = vi.copy()
    if len(list(vi.shape)) == 1:
        nd = vi.shape[0]
        for k, i in enumerate(range(0, nd, 3)):
            if dr == 1:
                mo[i : i + 3] = mo[i : i + 3] * sqrt(mass[k])
            else:
                mo[i : i + 3] = mo[i : i + 3] / sqrt(mass[k])
    else:
        nd = vi.shape[1]
        for j in range(vi.shape[0]):
            for k, i in enumerate(range(0, nd, 3)):
                if dr == 1:
                    mo[j, i : i + 3] = mo[j, i : i + 3] * sqrt(mass[k])
                else:
                    mo[j, i : i + 3] = mo[j, i : i + 3] / sqrt(mass[k])
    return mo


# XX matrix, cross product like matrix that represents the coordinate in the matrix=>vector mapping
def iX(r):
    """
    Generate the XX matrix for a given set of coordinates.

    Parameters:
    r (numpy.ndarray): The input coordinates.

    Returns:
    numpy.ndarray: The XX matrix.
    """
    na = r.shape[0]
    xo = zeros((na, 3, 3))
    for i in range(na):
        xo[i, 0, 1], xo[i, 0, 2], xo[i, 1, 2] = -r[i, 2], +r[i, 1], -r[i, 0]
        xo[i, :, :] = xo[i, :, :] - xo[i, :, :].T
    return xo


# uses the quaterion approach to eckart frame as discussed in
# stepanov et al, jcp, 140, 2014
# note that this is, at least most of the time, the same as doing:
# mx, me = mscale(self.sxx,self.mass,1), mscale(self.xxe,self.mass,1)
# U4 = rot_match_vec(mx,me)[1]
def EckartFrameTrans(xxe, sxx, mass):
    """
    Transform coordinates into the Eckart frame.

    Parameters:
    xxe (numpy.ndarray): Mass-scaled Cartesian coordinates.
    sxx (numpy.ndarray): Vectorial coordinates.
    mass (numpy.ndarray): The masses of the atoms.

    Returns:
    numpy.ndarray: Transformed coordinates in the Eckart frame.
    """
    C = zeros((4, 4))
    xp = xxe + sxx
    xm = xxe - sxx
    m = mass
    C[0, 1] = sum(m * (xp[:, 1] * xm[:, 2] - xm[:, 1] * xp[:, 2]))
    C[0, 2] = sum(m * (xm[:, 0] * xp[:, 2] - xp[:, 0] * xm[:, 2]))
    C[0, 3] = sum(m * (xp[:, 0] * xm[:, 1] - xm[:, 0] * xp[:, 1]))
    C[1, 2] = sum(m * (xm[:, 0] * xm[:, 1] - xp[:, 0] * xp[:, 1]))
    C[1, 3] = sum(m * (xm[:, 0] * xm[:, 2] - xp[:, 0] * xp[:, 2]))
    C[2, 3] = sum(m * (xm[:, 1] * xm[:, 2] - xp[:, 1] * xp[:, 2]))
    C += C.T
    C[0, 0] = sum(m * (xm[:, 0] ** 2 + xm[:, 1] ** 2 + xm[:, 2] ** 2))
    C[1, 1] = sum(m * (xm[:, 0] ** 2 + xp[:, 1] ** 2 + xp[:, 2] ** 2))
    C[2, 2] = sum(m * (xp[:, 0] ** 2 + xm[:, 1] ** 2 + xp[:, 2] ** 2))
    C[3, 3] = sum(m * (xp[:, 0] ** 2 + xp[:, 1] ** 2 + xm[:, 2] ** 2))
    ee, VV = eigh(C)
    # Find Orientat closest to equlibrium
    q = VV[:, 0]
    U = zeros((3, 3))
    U[0, 0] = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    U[1, 1] = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    U[2, 2] = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
    U[0, 1] = 2 * (q[1] * q[2] + q[0] * q[3])
    U[1, 0] = 2 * (q[1] * q[2] - q[0] * q[3])
    U[0, 2] = 2 * (q[1] * q[3] - q[0] * q[2])
    U[2, 0] = 2 * (q[1] * q[3] + q[0] * q[2])
    U[1, 2] = 2 * (q[2] * q[3] + q[0] * q[1])
    U[2, 1] = 2 * (q[2] * q[3] - q[0] * q[1])
    if False:
        sxx = matmul(U, sxx.T).T
        print("U  =, angles = ", iR2q(U))
        print(U)
    return U


# get inertia matrix body-fixed frame 1) transformation, 2) I in body fixed fram (I diagonal), 3) in space fixed frame
def iI(XX1, XX2, mass):
    """
    Calculate the inertia matrix and eigenvectors for a set of coordinates.

    Parameters:
    XX1 (numpy.ndarray): XX matrix for coordinates.
    XX2 (numpy.ndarray): XX matrix for transformed coordinates.
    mass (numpy.ndarray): The masses of the atoms.

    Returns:
    numpy.ndarray: Eigenvectors of the inertia matrix.
    numpy.ndarray: Eigenvalues of the inertia matrix.
    numpy.ndarray: The inertia matrix.
    """
    debug = False
    # debug = True
    na = len(mass)
    II = zeros((3, 3))
    for i in range(na):
        II += mass[i] * matmul(XX1[i, :, :].T, XX2[i, :, :])
    evl, evc = eigh(II)
    if det(evc) < 0.0:
        evc = matmul(evc, diag([1.0, 1.0, -1.0]))
    if debug:
        print("iEVL = ", evl)
        print("EVC = ")
        print(evc)
    return evc, diag(evl), II


# rotate about some cartesian axis
def Rabout(ang, d):
    """
    Generate a rotation matrix for a rotation about a Cartesian axis.

    Parameters:
    ang (float): The rotation angle (in radians).
    d (int): Cartesian axis of rotation (0 for x, 1 for y, 2 for z).

    Returns:
    numpy.ndarray: The rotation matrix.
    """
    if d == 1:
        ro = array([[cos(ang), sin(ang)], [-sin(ang), cos(ang)]])
    else:
        ro = array([[cos(ang), -sin(ang)], [sin(ang), cos(ang)]])
    oo = np.eye(3)
    ii = array([i for i in range(3) if i != d]).reshape(2, 1)
    oo[ii, ii.T] = ro
    return oo


# rotation matrix in euler parametrization
def iq2R(q):
    """
    Convert a quaternion to a rotation matrix in Euler parametrization.

    Parameters:
    q (numpy.ndarray): Quaternion represented as (alpha, beta, gamma) angles.

    Returns:
    numpy.ndarray: The rotation matrix.
    """
    alp, bet, gam = q
    return matmul(matmul(Rabout(alp, 2), Rabout(bet, 1)), Rabout(gam, 2))


# rotation matrix in cartesian parametrization
def ixyzR(w):
    """
    Convert a set of Euler angles to a rotation matrix in Cartesian parametrization.

    Parameters:
    w (numpy.ndarray): Euler angles represented as (alpha, beta, gamma) angles.

    Returns:
    numpy.ndarray: The rotation matrix.
    """
    return matmul(matmul(Rabout(w[2], 2), Rabout(w[1], 1)), Rabout(w[0], 0))


def iR2q(R):
    """
    Convert a rotation matrix to a quaternion in Euler parametrization.

    Parameters:
    R (numpy.ndarray): The rotation matrix.

    Returns:
    numpy.ndarray: Quaternion represented as (alpha, beta, gamma) angles.
    """
    if R[2, 2] < +1:
        if R[2, 2] > -1:
            bet = arccos(R[2, 2])
            alp = arctan2(R[1, 2], R[0, 2])
            gam = arctan2(R[2, 1], -R[2, 0])
        else:  # r22=−1
            bet = pi
            alp = -arctan2(R[1, 0], R[1, 1])
            gam = 0.0
    else:  # r22=+1
        bet = 0.0
        alp = arctan2(R[1, 0], R[1, 1])
        gam = 0.0
    return array([alp, bet, gam])


def stdq(q):
    """
    Standardize a quaternion to the range [0, 2π].

    Parameters:
    q (numpy.ndarray): Quaternion represented as (alpha, beta, gamma) angles.

    Returns:
    numpy.ndarray: Standardized quaternion.
    """
    for j in [0, 2]:
        # temp shift to 0:2pi
        q[j] += pi
        if q[j] < 0:
            q[j] = tpi + (q[j] - ceil(q[j] / tpi) * tpi)
        elif q[j] >= 2 * pi:
            q[j] = q[j] - floor(q[j] / tpi) * tpi
        q[j] -= pi
    if q[1] < 0:
        q[1] = pi + (q[1] - ceil(q[1] / pi) * pi)
    elif q[1] > pi:
        q[1] = q[1] - floor(q[1] / pi) * pi
    return q


def File2InputList(fnam):
    """
    Read input parameters from a text file.

    Parameters:
    fnam (str): Path to the input file.

    Returns:
    list: List of input parameter pairs.
    """
    lines = open(fnam, "r").readlines()
    inpd = []
    for line in lines:
        if "#" in line or len(line.strip()) == 0:
            continue
        if "=" in line:
            lns = line.strip().split()
            inpd.append([lns[0].lower(), lns[2:]])
    return inpd

