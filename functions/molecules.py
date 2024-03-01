#!/usr/bin/env python3
import numpy as np
from .constants import ang2au, au2ang, amu2au, au2cm, au2ev, au2mps, hpi, kboltz, mps2au
from .functions import ClassicalAngularMomentum, COM, el2Mass, EckartFrameTrans, File2InputList, GetRotTransVec
from .functions import iR2q, iX, iI, iq2R, mscale2, Rabout, ReadXYZ, rot_match_vec, ScaleTransform, stdq, XYZlist
from .dist import Boltzmann, EulerPolarSurface, EulerSurface, GaussianF, HarmWigner, MaxwellBoltzmann, VibPartFunc
from .dist import REBoltzmann
from .angmom import J2, Pz2, Pp2, Pm2, Pz
from .mc import SampleMC


class imolecule:
    """
    Represents a molecule and provides methods to read input data and generate molecular information.

    Attributes:
        na (int): Number of atoms.
        nd (int): Number of atomic coordinates (3 times the number of atoms).
        el (list): List of element symbols.
        mass (numpy.ndarray): Array of atomic masses.
        log (list): List of log messages.
        MaxV (int): Maximum velocity.
        Tvib (float): Vibrational temperature.
        Trot (float): Rotational temperature.
        Tvel (float): Velocity temperature.
        ordist (int): Order distribution.

    Methods:
        ReadInput(self, fnam): Read input data from a file.
        GenerateInputData(self): Process and generate input data.
        NModesOut(self, filn): Generate normal modes output.
        GenNormalModesMat(self): Generate normal modes matrix.
    """

    def __init__(self, **dic):
        """Initialize the imolecule object with default values."""
        self.na = 0
        self.nd = 0
        self.el = []
        self.mass = np.array([])
        self.log = []
        self.MaxV = 10
        self.Tvib = 0
        self.Trot = 0
        self.Tvel = 0
        self.ordist = 0
        self.nfreeze = []
    def ReadInput(self, fnam):
        """
        Read input data from a file and generate molecular information.

        Args:
            fnam (str): Name of the input file.
        """
        self.inpd = File2InputList(fnam)
        self.filename = fnam
        self.prefix = fnam.split(".")[0]
        self.name = self.prefix
        self.log.append("Reading input file : " + fnam + "\n")
        self.GenerateInputData()

    def GenerateInputData(self):
        """Process and generate molecular data from the input parameters."""
        for ky, val in self.inpd:
            if ky == "xyz":
                self.el, self.x0 = ReadXYZ(val[0])
                self.log += ["   Input Orientation (Ang): \n"]
                self.log += ["   " + ln for ln in XYZlist(self.el, self.x0)]
                self.x0 = self.x0 * ang2au
                self.na = len(self.el)
                self.nd = self.na * 3
                self.mass = el2Mass(self.el) * amu2au
                self.mass2 = np.repeat(self.mass, 3)
            if ky == "trot":
                self.log += ["   Temperature for Rotational States: " + val[0] + " \n"]
                self.Trot = float(val[0])
            if ky == "tvib":
                self.log += ["   Temperature for Vibrational States: " + val[0] + " \n"]
                self.Tvib = float(val[0])
            if ky == "tvel":
                if float(val[0]) > 0.0:                                      
                    self.log += [                                            
                        "Temperature for Intermolecular Velocity: "          
                        + val[0]+ " \n"                                      
                    ]
                    self.Tvel = float(val[0]) 
                else:
                    self.log +=[ 
                        'Intermolecular Velocity (m/s) centre: '             
                        + val[0]+ ' FWHM: ' + val[1] + '\n'                 
                    ]
                    if len(val) > 1:                                         
                        self.log +=[ 
                            'Full Width Half-Maximum (FWHM): '               
                            + val[1] + '\n'
                        ]
                        self.velfwhm = float(val[1])*mps2au                         
                    self.Tvel = float(val[0])*mps2au 
            if ky == "hess":
                self.HH = np.loadtxt(val[0])
            if ky == "name":
                self.name = val[0]
            if ky == "nfreeze":
                self.nfreeze = [int(n) for n in val]
            if ky == "ordist":
                self.log += ["Orientation Distribuition Function: " + val[0] + "\n"]
                self.ordist = int(val[0])
                self.orpars = [float(v) for v in val[1:]]
        self.GenNormalModesMat()
        self.StandardOrientat()
        if self.nm > 0:
            self.NModesOut(self.name + "_nm.xyz")

    def NModesOut(self, filn):
        """
        Generate normal modes output and save it to a file.

        Args:
            filn (str): Output file name.
        """
        out = []
        for m in range(self.ntr, self.nd):
            vv = np.zeros(self.nd)
            vv[m] = 4.0
            mo = np.reshape(np.matmul(self.n2x.T, vv), (self.na, 3))
            out += [
                str(self.na) + "\n",
                "Mode " + str(m) + " Freq :" +
                "{0:14.9f}".format(self.w[m]) + " \n",
            ]
            for i in range(self.na):
                out += [
                    self.el[i]
                    + " "
                    + "".join(
                        [
                            "{0:14.9f}".format(f) + " "
                            for f in (self.x0[i, :] * au2ang).tolist()
                            + mo[i, :].tolist()
                        ]
                    )
                    + "\n"
                ]
        open(filn, "w").writelines(out)

    def GenNormalModesMat(self):
        """
        Generate normal modes matrix and vibrational frequencies.

        If there is only one atom, it computes translational and rotational modes.
        If there are more atoms, it calculates the vibrational modes.

        Returns:
        list: Log messages.
        """
        if self.na == 1 or not hasattr(self,'HH'):
            self.c2n = GetRotTransVec(self.x0, self.mass, self.el).T
            self.ntr = self.c2n.shape[1]
            self.nm  = 0 
            self.w = np.zeros(self.ntr)
            self.log += ["   No harmonic frequencies :\n"]
        else:
            evl, self.c2n = np.linalg.eigh(self.HH)
            if evl[0] < -0.0001:
                self.log.append(
                    "   WARNING: some negative EigenValues detected in Hessian Matrix...\n"
                )
                print(
                    "WARNING: some negative EigenValues detected in Hessian Matrix..."
                )
                print("MOLECULE ", self.el)
                print(evl)
            self.w = np.array([np.sqrt(max([1.0e-15, e])) for e in evl])
            self.log += ["   Frequencies (cm-1):\n"]
            self.log += [
                "   " + str(i).rjust(2) + " " + "{0:14.9f}".format(e * au2cm) + "\n"
                for i, e in enumerate(self.w)
            ]

    def MolecularVeloc(self):
        """
        Calculate the center of mass velocity.

        Returns:
        numpy.ndarray: Center of mass velocity.
        """
        return COM(self.svv, self.mass)

    def MolecularPosition(self):
        """
        Calculate the center of mass position.

        Returns:
        numpy.ndarray: Center of mass position.
        """
        return COM(self.sxx, self.mass)

    def CalcOrient(self):
        """
        Calculate molecular orientation in terms of Euler angles.

        Returns:
        list: Log messages.
        """
        if self.na > 1:
            log = [" :" + self.name + " : \n"]
            U = EckartFrameTrans(self.xxe, self.sxx, self.mass)
            phi, theta, chi = iR2q(EckartFrameTrans(
                self.xxe, self.sxx, self.mass).T)
            self.SampInfo["angs"] = [phi, theta, chi]
            log += [
                "   Orientation, Euler Angles: "
                + "".join(["{0:10.7f}".format(a / np.pi) +
                          " " for a in [phi, theta, chi]])
                + " * pi rad \n"
            ]
        else:
            log += ["  No orientation space \n" ]
        return log

    def CalcInterEner(self):
        """
        Calculate the internal energy of the system.

        Returns:
        list: Log messages.
        """
        log = [" :" + self.name + " : \n"]
        if self.na == 1 or self.nm == 0:
         log += ["   No vibrational space  \n"]
         return log
        debug = True
        debug = False
        xx, vv = (
            self.sxx.copy() - COM(self.sxx, self.mass).T,
            self.svv.copy() - COM(self.svv, self.mass).T,
        )
        if debug:
            RR = self.m2c[: self.ntr, :]
            RR = mscale2(GetRotTransVec(
                self.xxe, self.mass, self.el), self.mass, -1)
            print("ZEROy? = ", np.linalg.norm(RR), np.diag(
                np.matmul(RR, self.c2m[:, : self.ntr])))
        # move molecule to its eckart frame
        U = EckartFrameTrans(self.xxe, xx, self.mass)
        # move to eckart frame:
        xx = np.matmul(U, xx.T).T
        vv = np.matmul(U, vv.T).T
        xe = self.xxe
        # project velocity to space of internals coordinates:
        RR = self.c2m[:, self.ntr:]
        PP = np.matmul(RR, np.matmul(np.linalg.inv(np.matmul(RR.T, RR)), RR.T))
        xx = np.matmul(PP, np.reshape(xx - xe, (self.nd)))
        # pp = np.reshape((vv.T*self.mass).T,(self.nd))
        # pp = np.matmul(PP,pp)
        pp = np.matmul(PP, np.reshape(vv, (self.nd))) * self.mass2
        # Convert to mass, frequency scaled Normal coordinates:
        Qp = np.matmul(self.n2x, pp)
        Qx = np.matmul(self.x2n.T, xx)
        EE = 0.5 * (Qx**2 + Qp**2)
        for i in range(self.ntr, self.nd):
            EE[i] = EE[i] * self.w[i]
        if debug:
            for i in range(self.nd):
                print(
                    i,
                    "{0:14.9f}".format(Qx[i])
                    + " "
                    + "{0:14.9f}".format(Qp[i])
                    + " E = "
                    + "{0:14.9f}".format(EE[i]),
                )
            quit()
        self.SampInfo["Qx"] = Qx
        self.SampInfo["Qp"] = Qp
        self.SampInfo["ViE"] = EE[self.ntr:]
        log += [
            " mode    freq     ~vstat       Q         P         QE        PE       EE (eV) \n"
        ]
        debug = False
        # debug = True
        if debug:
            print(log[-1].strip())
            for i in range(self.ntr):
                st = "NTR" + str(i).ljust(8)
                st += "{0:10.6f}".format(Qx[i])
                st += "{0:10.6f}".format(Qp[i])
                st += "{0:10.6f}".format(au2ev * 0.5 * Qx[i] ** 2)
                st += "{0:10.6f}".format(au2ev * 0.5 * Qp[i] ** 2)
                st += "{0:10.6f}".format(au2ev * 0.5 *
                                         (Qp[i] ** 2 + Qx[i] ** 2))
                print(st)
        for i in range(self.nm):
            e = (
                0.5
                * self.w[self.ntr + i]
                * (Qp[self.ntr + i] ** 2 + Qx[self.ntr + i] ** 2)
            )
            st = "   " + str(i).ljust(2) + "  "+ \
                "{0:6.1f}".format(self.w[self.ntr+i]*au2cm) + " "
            st += "  "+ "{0:8.4f}".format(e / self.w[i + self.ntr] - 0.5)
            st += "{0:10.6f}".format(Qx[self.ntr + i])
            st += "{0:10.6f}".format(Qp[self.ntr + i])
            st += "{0:10.6f}".format(
                au2ev * 0.5 * self.w[self.ntr + i] * Qx[self.ntr + i] ** 2
            )
            st += "{0:10.6f}".format(
                au2ev * 0.5 * self.w[self.ntr + i] * Qp[self.ntr + i] ** 2
            )
            st += "{0:10.6f}".format(e * au2ev)

            log += [st + "\n"]
            if debug:
                print(log[-1].strip())

        return log

    def InitializeSample(self):
        """
        Initialize the sample by setting initial coordinates.

        This method sets the initial coordinates for the simulation.
        """
        self.svv = np.zeros(self.xxe.shape)
        self.sxx = self.xxe.copy()

    # because the angular momentum and direction of the molecule should be conserved even as the molecule
    # is deformed (ie, vibrational motion), we can use the Jz and Energy from the equilibrium configuration
    # to define the angular momentum/angular velocity vector for the (potentially) deformed molecule.
    def SampleRigidRotorState(self):
        """
        Sample a state for a rigid rotor.

        Returns:
        list: Log messages.
        """
        log = ["  :" + self.name + " : \n"]
        if not hasattr(self, "rotsamp"):
            log += ["    no rotational state\n"]
            return log
        vi = self.rotsamp[0][self.rotsamp[-1]]
        self.rotsamp[-1] += 1
        # get classical angular momentum
        cJ = ClassicalAngularMomentum(self.Ib, self.RR[1][vi], self.RR[2][vi])
        # sets angular velocity from angular momentum:
        self.SetAngularVeloc(cJ)
        self.sren = self.RR[2][vi]
        log += [
            "    "
            + "Quantum   J :"
            + str(self.RR[0][vi]).rjust(5)
            + ", Jz : "
            + "{0:6.3f}".format(self.RR[1][vi])
            + ", Energy : "
            + "{0:7.5f}".format(self.RR[2][vi] * au2ev)
            + " eV\n"
        ]
        log += [
            "    "
            + "Classical J : "
            + "".join(["{0:10.7f}".format(j) + " " for j in cJ])
            + " (au) ,  |J| = "
            + "{0:8.2f}".format(np.linalg.norm(cJ))
            + " (QM: "
            + "{0:4.2f}".format(0.5 * (-1 + np.sqrt(1 + 4.0 * np.linalg.norm(cJ) ** 2)))
            + ")\n"
        ]
        return log

    def SampleZVeloc(self):
        """
        Sample z-velocity for the system.

        Returns:
        float: Z-velocity value.
        list: Log messages.
        """
        log = ["   :" + self.name + " : \n"]
        if not hasattr(self, "velsamp"):
            log += ["    No velocity sample \n"]
            return log
        v = self.velsamp[0][self.velsamp[-1]]
        self.velsamp[-1] += 1
        log += [
            "    Maxwell Velocity "
            + "{0:10.5f}".format(v)
            + ", Energy = "
            + "{0:10.5f}".format(0.5 * sum(self.mass) * v * au2ev)
            + " eV \n"
        ]
        return v, log

    def SampleOrientat(self):
        """
        Sample molecular orientation.

        Returns:
        list: Log messages.
        """
        log = ["  :" + self.name + " : \n"]
        if not hasattr(self, "osamp"):
            log += ["    no orientation state\n"]
            return log
        ang = self.osamp[0][self.osamp[-1]]
        self.osamp[-1] += 1
        # move molecule to its eckart frame
        U = EckartFrameTrans(self.xxe, self.sxx, self.mass)
        # move to eckart frame:
        self.sxx = np.matmul(U, self.sxx.T).T
        self.svv = np.matmul(U, self.svv.T).T
        # Molecule (eckart) to Space frame Rotation Matrix
        oR = iq2R(stdq(ang))
        self.svv = np.matmul(oR, self.svv.T).T
        self.sxx = np.matmul(oR, self.sxx.T).T
        log += [
            "    Orientations (phi,theta,chi): "
            + "".join(["{0:10.7f}".format(a / np.pi) + " " for a in ang])
            + " * pi rad \n"
        ]
        return log

    def InertiaFrameTransform(self):
        """
        Transform the coordinates to the inertia frame.

        Returns:
        numpy.ndarray: Transformed coordinates.
        """
        S2B, Ibm, Is = iI(iX(self.sxx), iX(self.sxx), self.mass)
        RS = np.round(np.matmul(self.Ro.T, S2B.T), 10)
        return RS

    # assumes mass frequency "dimensionless" scaled normal coordinates
    def SetHOVibrState(self, Q, P):
        """
        Set harmonic oscillator vibrational state.

        Args:
        Q (numpy.ndarray): Normal coordinates for displacement.
        P (numpy.ndarray): Normal coordinates for momentum.
        """
        x, p = np.matmul(self.n2x.T, Q), np.matmul(self.x2n, P)
        self.sxx += np.reshape(x, (self.na, 3))
        self.svv += (np.reshape(p, (self.na, 3)).T / self.mass).T

    # cJ arrives assuming the frame is in the standard orientation
    def SetAngularVeloc(self, cJ):
        """
        Set angular velocity based on classical angular momentum.

        Args:
        cJ (numpy.ndarray): Classical angular momentum.
        """
        xx = self.sxx - COM(self.sxx, self.mass).T
        S2B, Ibm, Is = iI(iX(xx), iX(xx), self.mass)
        iIs = np.linalg.pinv(Is)
        ww = np.matmul(iIs, cJ)
        vv = np.cross(ww, xx)
        self.svv += vv
        debug = False
        if debug:
            print("cJ in = ", cJ)
            # move to moment of Inertia frame:
            S2B, II, Is = iI(iX(xx), iX(xx), self.mass)
            xx = np.matmul(S2B.T, xx.T).T
            vv = np.matmul(S2B.T, vv.T).T
            LL = np.sum(np.cross(xx, vv).T * self.mass, axis=1)
            # move LL back to Eckart Molecular frame:
            Lm = np.matmul(S2B, LL)
            print("cJ ou = ", Lm)

    def CalcRotEner(self):
        """
        Calculate the rotational energy of the system.

        Returns:
        list: Log messages.
        """
        log = [" :" + self.name + " : \n"]
        if self.na == 1:
         log += ["   No rotational space  \n"]
         return log

        debug = True
        debug = False
        xx, vv = (
            self.sxx.copy() - COM(self.sxx, self.mass).T,
            self.svv.copy() - COM(self.svv, self.mass).T,
        )
        xe = self.xxe
        # move molecule to its eckart frame
        U = EckartFrameTrans(self.xxe, xx, self.mass)
        xx = np.matmul(U, xx.T).T
        vv = np.matmul(U, vv.T).T
        RR = self.c2m[:, : self.ntr - 3]
        if debug:
            R2 = mscale2(GetRotTransVec(self.xxe, self.mass, self.el), self.mass, -1)[
                :3, :
            ].T
            print("SHAP = ", RR.shape, R2.shape)
            print("DIFF? = ", np.diag(np.matmul(RR.T, R2)))
        # get projector
        PP = np.matmul(RR, np.matmul(np.linalg.inv(np.matmul(RR.T, RR)), RR.T))
        vv = np.reshape(np.matmul(PP, np.reshape(vv, (self.nd))), (self.na, 3))
        # xx = np.reshape(np.matmul(PP,np.reshape(xx-xe,(self.nd))),(self.na,3))+xe
        # move to moment of Inertia frame:
        S2B, II, Is = iI(iX(xx), iX(xx), self.mass)
        xx = np.matmul(S2B.T, xx.T).T
        vv = np.matmul(S2B.T, vv.T).T
        LL = np.sum(np.cross(xx, vv).T * self.mass, axis=1)
        ee = 0.5 * LL**2 / (1.0e-10 + np.diag(II))
        if debug:
            print("EE = ", ee)
        EE = sum(ee)
        # move LL back to Eckart Molecular frame:
        Lm = np.matmul(S2B, LL)
        # move LL back to current space frame:
        dum, O = rot_match_vec(xx, self.sxx)
        Ls = np.matmul(O, LL)
        self.SampInfo["Lm"] = Lm
        self.SampInfo["Ls"] = Ls
        self.SampInfo["RoE"] = ee
        log += [
            "   Ang Mom in Molecular Frame: "
            + "".join(["{0:10.5f}".format(p) for p in Lm])
            + ", |J| = "
            + "{0:10.5f}".format(np.linalg.norm(Lm))
            + " (QM: "
            + "{0:4.2f}".format(0.5 * (-1 + np.sqrt(1 + 4.0 * np.linalg.norm(Lm) ** 2)))
            + ")\n"
        ]
        log += [
            "   Rotational Energy (eV)    : "
            + "".join(["{0:10.5f}".format(p * au2ev) for p in ee])
            + ", Sum = "
            + "{0:10.5f}".format(EE * au2ev)
            + "\n"
        ]
        log += [
            "   Ang Mom in Current   Frame: "
            + "".join(["{0:10.5f}".format(p) for p in Ls])
            + "\n"
        ]

        return log

    def SampleHOVibrState(self):
        """
        Generate a sample of harmonic oscillator vibrational states.

        This method generates a sample of harmonic oscillator vibrational states
        for a molecule, taking into account the vibrational frequencies and
        temperature. It returns the vibrational states and associated energies.

        Returns:
            list: Log messages containing information about generated states.
        """
        debug = True
        debug = False
        if debug:
            # samp = SampleMC(5000,array([0.01,-0.02]),HarmWigner,[0],idel=1.5)[0]
            samp = self.wigsamp[0]
            N = len(samp)
            for j in range(self.nm):
                ee = 0.0
                Q, P = np.zeros(self.nm), np.zeros(self.nm)
                for i in range(N):
                    Q[j], P[j] = samp[i]
                    ee += 0.5 * self.w[j + self.ntr] * (Q[j] ** 2 + P[j] ** 2)
                print(
                    j,
                    " e = ",
                    au2cm * ee / float(N),
                    0.5 * au2cm * self.w[j + self.ntr],
                )
            quit()
        log = ["  :" + self.name + " : \n"]
        if not hasattr(self, "nvibsamp"):
            log += ["    no vibrational state\n"]
            return log
        vi = self.nvibsamp[0][self.nvibsamp[-1]]
        self.nvibsamp[-1] += 1
        Q, P = np.zeros(self.nd), np.zeros(self.nd)
        self.sven = np.zeros(self.nm)
        wn = len(self.wigsamp[0])
        log += [
            "    mode   freq  vstat    Q         P         QE        PE        EE (eV)      \n"
        ]
        for j, ii in enumerate(vi, start=self.ntr):
            i = int(ii)
            wi = self.wigsamp[i][np.random.randint(0, wn)]
            if j - self.ntr not in self.nfreeze:
             Q[j], P[j] = wi
            self.sven[j - self.ntr] = 0.5 * self.w[j] * (Q[j] ** 2 + P[j] ** 2)
            st = "    " + str(j - self.ntr).ljust(3) + "  "+ \
                "{0:6.1f}".format(self.w[j]*au2cm) + "  " + str(i).ljust(4)
            st += "{0:10.6f}".format(Q[j])
            st += "{0:10.6f}".format(P[j])
            st += "{0:10.6f}".format(au2ev * 0.5 * self.w[+j] * Q[j] ** 2)
            st += "{0:10.6f}".format(au2ev * 0.5 * self.w[+j] * P[j] ** 2)
            st += "{0:10.6f}".format(au2ev * 0.5 *
                                     self.w[+j] * (P[j] ** 2 + Q[j] ** 2))
            log += [st + "\n"]
        self.SetHOVibrState(Q, P)
        return log

    def SetOrientatConvention(self, S2B, Ib, Rep):
        """
        Set the orientation convention for the molecule's rotational state.

        This method sets the orientation convention for the molecule's rotational state.
        The orientation convention is based on the shape of the molecule and its
        principal moments of inertia. It also adjusts the orientation transformation
        matrix S2B and the moment of inertia tensor Ib accordingly.

        Args:
            S2B (numpy.ndarray): The orientation transformation matrix.
            Ib (list): The principal moments of inertia [Ia, Ib, Ic].
            Rep (int): The representation code for the orientation convention.
        """
        # a b c are ordered with incresing 1/I eigenvalue (decreasing rotational constant)
        # the space-body frame rotation matrix S2B is fixed so that the figure axis z (the  symmetry axis) is pointing along Z
        # prolate (sausage) x y z (b c a), Ib~Ic > Ia
        if Rep == 1:  # b c a
            if self.rB > self.rC:
                self.J2c = 0.5 * (self.rB + self.rC)
                self.Jzc = self.rA - 0.5 * (self.rB + self.rC)
                self.Jcc = 0.25 * (self.rB - self.rC)
                Ib[0], Ib[1], Ib[2] = Ib[1], Ib[2], Ib[0]
                R = np.matmul(Rabout(hpi, 0), Rabout(hpi, 1))
            self.S2B = np.matmul(R.T, S2B.T)
            self.Irep = 1
        elif Rep == 2:  # c a b
            if self.rC > self.rA:
                self.J2c = 0.5 * (self.rC + self.rA)
                self.Jzc = self.rB - 0.5 * (self.rC + self.rA)
                self.Jcc = 0.25 * (self.rC - self.rA)
                Ib[0], Ib[1], Ib[2] = Ib[2], Ib[0], Ib[1]
                R = np.matmul(Rabout(hpi, 2), Rabout(hpi, 1))
            self.S2B = np.matmul(R.T, S2B.T)
            self.Irep = 2
        # oblate (frizbee) x y z ( a b c ), Ic > Ia~Ib
        elif Rep == 3:  # a b c
            self.J2c = 0.5 * (self.rA + self.rB)
            self.Jzc = self.rC - 0.5 * (self.rA + self.rB)
            self.Jcc = 0.25 * (self.rA - self.rB)
            self.Irep = 3
            self.S2B = S2B.T
            R = np.identity(3)
        # linear
        elif Rep == 0:
            self.J2c, self.Jzc, self.Jcc = self.rB, 0.0, 0.0
            Ib[0], Ib[1], Ib[2] = Ib[2], Ib[1], Ib[0]
            R = Rabout(hpi, 1)
            self.S2B = np.matmul(R.T, S2B.T)
            self.Irep = 0
        self.Ib = Ib
        self.Ro = R

    def StandardOrientat(self):
        """
        Standardize the orientation of the molecule.

        This method standardizes the orientation of the molecule based on its shape
        and principal moments of inertia. It also adjusts the orientation transformation
        matrix and other relevant properties.

        Returns:
            list: Log messages describing the standard orientation.
        """
        debug = False
        debug = True
        self.log.append("   #### Standard Orientation: \n")
        self.x0 += -COM(self.x0, self.mass).T
        if self.na == 1:
            self.log.append("   No standard orinetation: \n")
            self.rsym = "atom"
            x0 = self.x0
        else:
            S2B, Ibm, Is = iI(iX(self.x0), iX(self.x0), self.mass)
            Ib = np.diag(Ibm).copy()
            self.rA, self.rB, self.rC = 0.5 / (Ib + 1.0e-20)
            asymk = (2 * self.rB - self.rA - self.rC) / (self.rA - self.rC)
            self.asymk = asymk
            self.log.append(
                "   Asymmetry constant: " + "{0:14.9f}".format(asymk) + "\n"
            )
            if abs(Ib[0]) < 0.001:
                self.rsym = "linear"
                self.SetOrientatConvention(S2B, Ib, 0)
            elif asymk > 0.333:
                self.SetOrientatConvention(S2B, Ib, 3)
                if asymk - 0.98 > 0:
                    self.rsym = "oblate"
                else:
                    self.rsym = "asym-oblate"
            elif asymk < -0.333:
                self.SetOrientatConvention(S2B, Ib, 1)
                if asymk + 0.98 < 0:
                    self.rsym = "prolate"
                else:
                    self.rsym = "asym-prolate"
            else:
                self.SetOrientatConvention(S2B, Ib, 2)
                if abs(asymk) < 0.001:
                    self.rsym = "spherical"
                else:
                    self.rsym = "asym-spherical"
            self.log.append("   Rotor type : " + self.rsym + "\n")
            x0 = np.matmul(self.S2B, self.x0.T).T
        self.xxe, self.x0 = x0.copy(), x0.copy()
        self.log += ["   Standard Orientation :\n"]
        self.log += ["   " + ln for ln in XYZlist(self.el, x0)]
        if hasattr(self, "c2n"):
            if self.na != 1 and hasattr(self,'HH'):
             for i in range(self.nd):
                 self.c2n[:, i] = np.reshape(
                     np.matmul(self.S2B, np.reshape(
                         self.c2n[:, i], (self.na, 3)).T).T,
                     (self.nd,),
                 )
             rt = GetRotTransVec(self.x0, self.mass, self.el)
             self.ntr = rt.shape[0]
             self.nm = self.nd - self.ntr
             self.c2n = self.c2n[:, self.ntr:]
             self.c2n = np.concatenate([rt, self.c2n.T]).T
            # mas scaled normal coordinates
            self.n2c = self.c2n.T
            # normal coordinates
            self.c2m, self.m2c = ScaleTransform(
                self.c2n, self.n2c, self.w, self.mass, +1, 0
            )
            # classical momenta have the mass frequency reversed
            self.p2m, self.m2p = ScaleTransform(
                self.c2n, self.n2c, self.w, self.mass, -1, 0
            )
            # mass frequency scaled normal coordinates to
            self.x2n, self.n2x = ScaleTransform(
                self.c2n, self.n2c, self.w, self.mass, +1, 1
            )

        debug = False
        if debug:
            RR = GetRotTransVec(self.x0, self.mass, self.el)
            print("ZERO?1 = ", np.linalg.norm(RR), np.diag(
                np.matmul(RR, self.c2n[:, : self.ntr])))
            print("ZERO?3 = ", np.linalg.norm(RR), np.diag(
                np.matmul(RR, self.c2n[:, : self.ntr])))
            RR = mscale2(GetRotTransVec(
                self.x0, self.mass, self.el), self.mass, -1)
            print("ZERO?5 = ", np.linalg.norm(RR), np.diag(
                np.matmul(RR, self.c2m[:, : self.ntr])))
            print("ZERO?6 = ", np.linalg.norm(RR), np.diag(
                np.matmul(RR, self.c2m[:, : self.ntr])))
        return

    def EstimateMaxR(self, T):
        """
        Estimate the maximum rotational state for a given temperature.

        This method estimates the maximum rotational state for a molecule at a
        given temperature based on the temperature and molecule's properties.

        Args:
            T (float): The temperature in Kelvin.

        Returns:
            int: The estimated maximum rotational state.
        """
        MaxR = 5
        for i in range(100):
            E = self.J2c * i * (i + 1)
            if i > 5 and Boltzmann(E, T) < 0.001:
                MaxR = i
                break
        return int(MaxR) + 1

    def RigidRotorEnergies(self, MaxR):
        """
        Calculate the energies of the rigid rotor states.

        This method calculates the energies of the rigid rotor states of the molecule
        up to a specified maximum state.

        Args:
            MaxR (int): The maximum rotational state.

        Returns:
            dict: List containing information about rotational states.
        """
        EE, ZZ, JJ = [], [], []
        for i in range(MaxR):
            HJ = self.J2c * J2(i) + self.Jzc * Pz2(i) + \
                self.Jcc * (Pp2(i) + Pm2(i))
            HJ = 0.5 * (HJ + HJ.T)
            HJ = np.round(HJ.real, 10)
            zz = np.round(Pz(i).real, 10)
            if "asym" in self.rsym:
                ee, VV = np.linalg.eigh(HJ)
                zz = np.diag(np.matmul(VV.T, np.matmul(zz, VV)))
                EE += ee.tolist()
            else:
                EE += np.diag(HJ).tolist()
                zz = np.diag(zz)
            ZZ += zz.tolist()
            JJ += [i for j in range(2 * i + 1)]
        iorder = [a[1] for a in sorted(list(zip(EE, range(len(EE)))))]
        EE = np.array([EE[i] for i in iorder])
        JJ = np.array([JJ[i] for i in iorder])
        ZZ = np.array([ZZ[i] for i in iorder])
        self.RR = [JJ, ZZ, EE]

    def EstimateMaxVI(self):
        """
        Estimate the maximum value MaxV using nested loops and a condition based on Boltzmann probability.

        This function iterates over two loops to estimate the maximum value MaxV based on certain conditions.

        Returns:
        int: The estimated maximum value MaxV.
        """
        MaxV = 0
        for i in range(self.ntr, self.ntr+self.nm):
            for j in range(self.MaxV):
                if Boltzmann(self.w[i]*j, self.Tvib) > 0.001:
                    if MaxV < j:
                        MaxV = j
                        break
                else:
                    break
        return MaxV

    def InitialDist(self, Nsamp):
        """
        Generate an initial distribution of samples for the molecule.

        This method generates samples for various properties of the molecule, including vibrational states, rotational states,
        orientation, and velocity, based on the specified parameters.

        Args:
        Nsamp (int): The number of samples to generate.

        Returns:
        list: Log messages indicating the distribution generation process.
        """
        log = []
        log += [
            "Generated "
            + str(Nsamp)
            + " Samples from distribuitions (molecule :"
            + self.filename
            + ") \n"
        ]
        # generate vibrational dist:
        T = self.Tvib
        if self.nm > 0 and T >= 0:
            log += ["Generated vibrational state distro temperature " +
                    str(T) + "\n"]
            self.MaxV = self.EstimateMaxVI()
            log += ["Maximum Excited Vibrational state " +
                    str(self.MaxV) + "\n"]
            # doesent make sense
            if self.MaxV > 0:
                # generate some samples for vib states, assumes first self.ntr vibrational frequencies are zero...
                self.nvibsamp = SampleMC(
                    Nsamp,
                    [0 for i in range(self.nm)],
                    VibPartFunc,
                    [self.w[self.ntr:], T, False],
                    domains=[[0.0, 10, False] for i in range(self.nm)],
                    idd=1
                )[0]

                self.nvibsamp = [[[int(i) for i in ii]
                                  for ii in self.nvibsamp], 0]
                log += [
                    "Total Average Vibrational excitations per Mode in Distribuition:\n"
                ]
                log += [
                    "".join(["{0:8.2f}".format(w * au2cm).rjust(9)
                            for w in self.w[self.ntr:]])
                    + "\n"
                ]
                log += [
                    "".join(
                        [
                            "{0:8.6f}".format(v).rjust(9)
                            for v in np.sum(self.nvibsamp[0], axis=0) / float(Nsamp)
                        ]
                    )
                    + "\n"
                ]
            else:
                self.nvibsamp = [
                    [[0 for j in range(self.nm)] for i in range(Nsamp)], 0]
            self.nvibsamp[0] = [
                [min([self.MaxV, int(v)]) for v in vi] for vi in self.nvibsamp[0]
            ]
            MaxV = int(max([max(vi) for vi in self.nvibsamp[0]]))
            self.MaxV = MaxV
            log += [
                "Generated "
                + str(10 * Nsamp)
                + " samples from  wigner distribuitions up to excited state "
                + str(MaxV)
                + "\n"
            ]
            self.wigsamp = []
            # generate some samples from each wigner distro for each vib state
            for m in range(MaxV + 1):
                self.wigsamp.append(
                    SampleMC(
                        10 * Nsamp, np.array([0.01, -0.02]), HarmWigner, [m], idel=1.5
                    )[0]
                )
        else:
            log += ["No vibrational state distro" ]
        # generate rotational dist:
        T = self.Trot
        if self.na > 1 and T >= 0:
            log += ["Generated rotational state distro temperature " +
                    str(T) + "\n"]
            if T > 0:
                MaxR = self.MaxR
                log += ["Maximum rigid rotor state " + str(MaxR) + "\n"]
                # generate some samples for vib state
                self.rotsamp = SampleMC(
                    Nsamp,
                    [0],
                    REBoltzmann,
                    [self.RR[2], self.RR[0], T],
                    domains=[[0, MaxR, False]],
                    idel=3.0,
                )[0]
                self.rotsamp = [[int(ii) for ii in self.rotsamp], 0]
                jd = np.zeros(MaxR + 1)
                for j in self.rotsamp[0]:
                    jd[j] += 1
                log += [" Rotational State Distribuition:\n"]
                LR = [i for i in range(MaxR) if jd[i] > 0.0][-1]+1
                log += ["".join([str(i).rjust(7) for i in range(LR)]) + "\n"]
                log += ["".join([str(j).rjust(7) for j in jd[:LR]]) + "\n"]
            else:
                self.rotsamp = [0 for i in range(Nsamp)]
        else:
            log += ["No rotational state distro" ]
        # generate orientation
        if self.na > 1:
            log += ["Generated orientational distribuition type " +
                    str(self.ordist) + "\n"]
            if self.ordist == 1:
                self.osamp = [
                    SampleMC(
                        Nsamp,
                        np.zeros(3),
                        EulerPolarSurface,
                        self.orpars,
                        domains=[[-np.pi, np.pi, True],
                                 [0.0, np.pi, True], [-np.pi, np.pi, True]],
                    )[0],
                    0,
                ]
            elif self.ordist == 0:
                self.osamp = [
                    SampleMC(
                        Nsamp,
                        np.zeros(3),
                        EulerSurface,
                        [],
                        domains=[[-np.pi, np.pi, True],
                                 [0.0, np.pi, True], [-np.pi, np.pi, True]],
                    )[0],
                    0,
                ]
            # fix specific orientation
            elif self.ordist == -1:
                self.osamp = [np.array(self.orpars) for i in range(Nsamp)]
        else:
            log += [" No orientational distribution " ]
        # generate velocity dist:
        T = self.Tvel
        if T > 0:
            log += ["Generated molecular velocity temperature " +
                    str(T) + "\n"]
            d = np.sqrt(1.0 * 2 * kboltz * T / sum(self.mass))
            self.velsamp = [
                [
                    abs(v)
                    for v in SampleMC(
                        Nsamp, [d], MaxwellBoltzmann, [T, sum(self.mass)], idel=d*0.1
                    )[0]
                ],
                0,
            ]

        elif T < 0:
            log += ["Generated molecular velocity " + str(abs(T)) + "\n"]
            d = abs(T)*mps2au
            if hasattr(self,'velfwhm'):
                log += [" FWHM: " + str(self.velfwhm) + "\n"]
                self.velsamp = [
                    SampleMC(self.Nsamp,[d],GaussianF,[d,self.velfwhm],idel=d*0.1)[0],
                    0,
                ] 
            else:  
                self.velsamp = [[abs(T) for T in range(Nsamp)], 0]
        if abs(T) > 0:
            hist, edg = np.histogram(self.velsamp[0], bins=10)
            log += ["Total Velocity Distribuition (m/s):\n"]
            hw = 0.5 * (edg[1] - edg[0])
            log += [
                "".join(
                    ["{0:8.2f}".format((w + hw) * au2mps) + " " for w in edg])
                + "\n"
            ]
            log += ["".join([str(v).rjust(8) for v in hist]) + "\n"]
        self.log += log
