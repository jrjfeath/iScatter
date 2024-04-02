#!/usr/bin/env python3
import os
import numpy as np
from .constants import ang2au, au2ang, au2ev, au2fmt, au2mps, fmt2au, kboltz, mps2au, x, y, z
from .dist import IPDist, MaxwellBoltzmann, GaussianF
from .functions import COM, File2InputList, XYZlist, ReadXYZs
from .mc import SampleMC
from .molecules import imolecule

class iscatter:
    """A class for simulating molecular scattering events."""

    def __init__(self, **dic):
        """Initialize the scattering simulation.

        Args:
            dic (dict): Additional parameters for initialization.
        """

        self.mol = [imolecule(), imolecule()]
        self.chi = 0.0
        self.log = []
        self.slog = []
        self.sampls = {'cv': [], 'info': []}
        self.sii = 0
        return

    def ReadInput(self, fnam):
        """Read input data from a file.

        Args:
            fnam (str): Input file name.
        """
        self.log.append("Reading scattering input file : " + fnam + "\n")
        self.inpd = File2InputList(fnam)
        self.filename = fnam
        self.prefix = fnam.split(".")[0]
        self.GenerateInputData()
        self.distdir = self.prefix + "_dist"
        self.fileout = 'out'
        if not os.path.isdir(self.distdir):
            os.system("mkdir " + self.distdir)

    def GenerateInputData(self):
        """Generate input data from the provided parameters.

        Reads and processes input data from the input file, setting up simulation parameters and molecular properties.
        """
        mol = self.mol
        for ky, val in self.inpd:
            if ky == "mol":
                self.log.append(
                    "########################################################\n"
                )
                self.log.append(
                    "Reading molecular input : " +
                    val[0] + " " + val[1] + " " + "\n"
                )
                mol[int(val[0])].ReadInput(val[1])
                self.log += mol[int(val[0])].log
                mol[int(val[0])].log = []
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
                        self.velfwhm = float(val[1])
                    self.Tvel = float(val[0])
            if ky == "fileout":
                self.log += ["Prefix output Prefix Name: " + val[0] + "\n"]
                self.fileout = val[0]
            if ky == "dirout":
                self.log += ["Directory Output Name: " + val[0] + "\n"]
                self.dirout = val[0]
            if ky == "seed":
                self.log += ["RNG Seed: " + val[0] + "\n"]
                self.seed = int(val[0])
                np.random.seed(self.seed)
            if ky == "trot":
                self.log += ["Temperature for Rotational States: " + val[0] + " \n"]
                self.Trot = float(val[0])
            if ky == "tvib":
                self.log += ["Temperature for Vibrational States: " + val[0] + " \n"]
                self.Tvib = float(val[0])
            if ky == "maxb":
                self.log += ["Maximum impact parameter: " + val[0] + "\n"]
                self.MaxB = float(val[0])*ang2au
            if ky == "chi":
                self.log += ["Azimuthal scattering angule : " + val[0] + "\n"]
                self.chi = float(val[0])
            if ky == "maxv":
                self.log += ["Maximum vibrational State: " + val[0] + "\n"]
                self.MaxV = int(val[0])
            if ky == "nsamp":
                self.log += ["Number of generated samples: " + val[0] + "\n"]
                self.Nsamp = int(val[0])
            if ky == "rz":
                self.log += ["Intermolecular Z-Distance: " + val[0] + "\n"]
                self.Rz = float(val[0])*ang2au
            if ky == "ordist":
                self.log += ["Orientation Distribuition Function: " + val[0] + "\n"]
                self.ordist = int(val[0])
                self.orpars = [float(v) for v in val[1:]]
        self.na = mol[0].na + mol[1].na
        self.el = mol[0].el + mol[1].el
        self.mass = np.array(mol[0].mass.tolist() + mol[1].mass.tolist())
        self.mass2 = np.repeat(self.mass, 3)
        self.rmass = (
            sum(mol[0].mass) * sum(mol[1].mass) /
            (sum(mol[0].mass) + sum(mol[1].mass))
        )
        self.log += [
            "Reduced Mass           : " + "{0:10.3e}".format(self.rmass) + "\n"
        ]
        self.nd = self.na * 3
        w0, w1 = sum(mol[0].mass), sum(mol[1].mass)
        self.w0, self.w1 = w0 / (w1 + w0), w1 / (w1 + w0)
        self.consistentT()
        self.RigidRotorEnergies()
        self.InitializeSample()
        self.SetInterZDist(self.Rz*2)
        self.Mol2Image()
        xyz = XYZlist(self.el, self.sxx * au2ang,
                      mess="Reference Geometry, double Rz (Ang)")
        open(self.fileout.split('.')[0] + '_reference.xyz','w').writelines(xyz)

    # overwrite temperatures if necessary:
    def consistentT(self):
        """Ensure temperature consistency within the system.

        This method enforces consistency in temperatures by overwriting molecular temperatures with system temperatures if provided.
        """
        mol = self.mol
        for i in range(2):
            if hasattr(self, "Trot"):
                self.log.append(
                    "Overwriting molecular Trot to system Trot.."
                    + str(self.Trot)
                    + "\n"
                )
                mol[i].Trot = self.Trot
            if hasattr(self, "Tvib"):
                self.log.append(
                    "Overwriting molecular Tvib to system Tvib.."
                    + str(self.Tvib)
                    + "\n"
                )
                mol[i].Tvib = self.Tvib

    def RigidRotorEnergies(self):
        """Calculate rigid rotor energies for molecules.

        This method calculates the rigid rotor energies for both molecules based on their estimated maximum rotational radii.
        """
        mol = self.mol
        for i in range(2):
            if mol[i].na > 1:
                mol[i].MaxR = mol[i].EstimateMaxR(self.Trot)
                mol[i].RigidRotorEnergies(mol[i].MaxR)

    def savedists(self):
        """Save distribution data to files.

        This method saves various distribution data to files, including vibrational, rotational, and orientation distributions.
        """
        out = []
        for i in range(2):
            if hasattr(self.mol[i], "nvibsamp"):
                np.savetxt(
                    self.distdir + "/nvibsamp_" + str(i) + ".dist",
                    np.array(self.mol[i].nvibsamp[0], dtype=np.int8),
                )
                for j, ws in enumerate(self.mol[i].wigsamp):
                    np.savetxt(
                        self.distdir + "/wigsamp_" +
                        str(j) + "_" + str(i) + ".dist", ws
                    )
            if hasattr(self.mol[i], "rotsamp"):
                np.savetxt(
                    self.distdir + "/rotsamp_" + str(i) + ".dist",
                    np.array(self.mol[i].rotsamp[0], dtype=np.int8),
                )
                # np.savetxt(self.distdir+'/rotsamp_'+str(i) + '.dist',self.mol[i].rotsamp[0])
            if hasattr(self.mol[i], "osamp"):
                np.savetxt(
                    self.distdir + "/osamp_" +
                    str(i) + ".dist", self.mol[i].osamp[0]
                )
            if hasattr(self.mol[i], "velsamp"):
                np.savetxt(
                    self.distdir + "/velsamp_" + str(i) + ".dist",
                    self.mol[i].velsamp[0],
                )
        if hasattr(self, "velsamp"):
            np.savetxt(self.distdir + "/velsamp.dist", self.velsamp[0])
        if hasattr(self, "bsamp"):
            np.savetxt(self.distdir + "/bsamp.dist", self.bsamp[0])
        if hasattr(self, "chisamp"):
            np.savetxt(self.distdir + "/chisamp.dist", self.chisamp)

    def loaddists(self):
        """Load distribution data from files.

        This method loads distribution data from previously saved files, allowing for resuming simulations.
        """
        out = []
        for i in range(2):
            if os.path.exists(self.distdir + "/nvibsamp_" + str(i) + ".dist"):
                self.log.append(
                    "Reading " + self.distdir +
                    "/nvibsamp_" + str(i) + ".dist\n"
                )
                self.mol[i].nvibsamp = [
                    np.loadtxt(self.distdir + "/nvibsamp_" + str(i) + ".dist").astype(
                        np.int8
                    ),
                    0,
                ]
                if self.mol[i].na == 2:
                    self.mol[i].nvibsamp[0] = [
                        np.array([v]) for v in self.mol[i].nvibsamp[0]
                    ]
                j = 0
                while True:
                    if os.path.exists(
                        self.distdir + "/wigsamp_" +
                            str(j) + "_" + str(i) + ".dist"
                    ):
                        if not hasattr(self.mol[i], "wigsamp"):
                            self.mol[i].wigsamp = []
                        self.log.append(
                            "Reading "
                            + self.distdir
                            + "/wigsamp_"
                            + str(j)
                            + "_"
                            + str(i)
                            + ".dist\n"
                        )
                        self.mol[i].wigsamp.append(
                            np.loadtxt(
                                self.distdir
                                + "/wigsamp_"
                                + str(j)
                                + "_"
                                + str(i)
                                + ".dist"
                            )
                        )
                        j += 1
                    else:
                        break
            if os.path.exists(self.distdir + "/rotsamp_" + str(i) + ".dist"):
                self.log.append(
                    "Reading " + self.distdir +
                    "/rotsamp_" + str(i) + ".dist\n"
                )
                self.mol[i].rotsamp = [
                    np.loadtxt(self.distdir + "/rotsamp_" + str(i) + ".dist").astype(
                        np.int8
                    ),
                    0,
                ]
            if os.path.exists(self.distdir + "/osamp_" + str(i) + ".dist"):
                self.log.append(
                    "Reading " + self.distdir + "/osamp_" + str(i) + ".dist\n"
                )
                self.mol[i].osamp = [
                    np.loadtxt(self.distdir + "/osamp_" + str(i) + ".dist"),
                    0,
                ]
            if os.path.exists(self.distdir + "/velsamp_" + str(i) + ".dist"):
                self.log.append(
                    "Reading " + self.distdir +
                    "/velsamp_" + str(i) + ".dist\n"
                )
                self.mol[i].velsamp = [
                    np.loadtxt(self.distdir + "/velsamp_" + str(i) + ".dist"),
                    0,
                ]
        if os.path.exists(self.distdir + "/velsamp.dist"):
            self.log.append("Reading " + self.distdir + "/velsamp.dist\n")
            self.velsamp = [np.loadtxt(self.distdir + "/velsamp.dist"), 0]
        if os.path.exists(self.distdir + "/bsamp.dist"):
            self.log.append("Reading " + self.distdir + "/bsamp.dist\n")
            self.bsamp = [np.loadtxt(self.distdir + "/bsamp.dist"), 0]
        if os.path.exists(self.distdir + "/chisamp.dist"):
            self.log.append("Reading " + self.distdir + "/chisamp.dist\n")
            self.chisamp = np.loadtxt(self.distdir + "/chisamp.dist")

    # Generates all the necessary Nsamp distribuition samples for orientational, rotational, vibrational, velocity,
    def InitialDist(self):
        """Generate initial distribution samples for different states.

        This method generates initial distribution samples for various states, including vibrational, rotational, and velocity distributions.
        """
        print("Generating Distribuitions, this may take a while...")
        # initialize sample counter and log:
        self.sii = 0
        self.slog = []
        self.log += ["Generating " +
                     str(self.Nsamp) + " Samples from distribuition \n"]

        # generates molecular vibrational and rotational distribuitions.
        for i in range(2):
            # overwrites molecular temperatures if system is provided:
            self.mol[i].Tvib, self.mol[i].Trot = self.Tvib, self.Trot
            self.mol[i].InitialDist(self.Nsamp)
            self.log += self.mol[i].log
            self.mol[i].log = []
            print("Molecule " + str(i) + " done ...")

        self.log += [
            "Generated impact parater with maximum b= " + str(self.MaxB) + "\n"
        ]
        # impact parameter sampling:
        self.bsamp = [
            [abs(b)
             for b in SampleMC(self.Nsamp, [0.5], IPDist, [self.MaxB])[0]],
            0,
        ]
        self.chisamp = [np.random.rand() * self.chi for i in range(self.Nsamp)]
        # generate Intermolecular velocity dist:
        if hasattr(self, "Tvel"):
            T = self.Tvel
        else:
            T = 0
        if T > 0:
            self.log += [
                "Generated intermolecular velocity temperature " +
                str(T) + "\n"
            ]
            d = np.sqrt(1.0 * 2 * kboltz * T / self.rmass)
            self.velsamp = [
                [
                    abs(v)
                    for v in SampleMC(
                        self.Nsamp, [d], MaxwellBoltzmann, [T, self.rmass], idel=d*0.1
                    )[0]
                ],
                0,
            ]
        elif T < 0:
            self.log += ["Generated intermolecular velocity " + str(abs(T)) + "\n"]
            d = abs(T)*mps2au
            if hasattr(self,'velfwhm'):
                self.log += [" FWHM: " + str(self.velfwhm) + "\n"]
                self.velsamp = [
                    SampleMC(self.Nsamp,[d],GaussianF,[d,self.velfwhm],idel=d*0.1)[0],
                    0,
                ]
            else:  
                self.velsamp = [abs(T) for T in range(self.Nsamp)]
        if abs(T) > 0:
            hist, edg = np.histogram(self.velsamp[0], bins=10)
            self.log += ["Total Velocity Distribuitio (m/s):\n"]
            hw = 0.5 * (edg[2] - edg[1])
            self.log += [
                "".join(
                    ["{0:8.2f}".format((w + hw) * au2mps) + " " for w in edg[:-2]])
                + "\n"
            ]
            self.log += ["".join([str(v).rjust(8)+" " for v in hist]) + "\n"]
        self.savedists()
        open(self.fileout.split(".")[0] +
             "_dist.log", "w").writelines(self.log)

    def CalcInterMolMomentum(self):
        """Calculate intermolecular momentum and energy.

        This method calculates the intermolecular momentum and energy, including rotational and radial components.
        """
        log = ["  :" + self.mol[0].name + " x " + self.mol[1].name + "\n"]
        self.Mol2Image()
        com, vcom = COM(self.sxx, self.mass), COM(self.svv, self.mass)
        xx, vv = np.zeros((2, 3)), np.zeros((2, 3))
        ms = np.array([sum(self.mol[0].mass), sum(self.mol[1].mass)])
        for i in range(2):
            xx[i, :] = (self.mol[i].MolecularPosition() - com).T
            vv[i, :] = (self.mol[i].MolecularVeloc() - vcom).T
        KE = np.sum(0.5*(vv.T**2*ms),axis=0).tolist()
        # difference jacobi:
        rr = xx[0, :] - xx[1, :]
        # projection onto radial and angular parts
        Pr = np.outer(rr, rr) / np.linalg.norm(rr) ** 2
        vr = np.matmul(Pr, vv.T).T
        vp = vv - vr
        # angular and radial momentums:
        JJ = np.sum(np.cross(xx, vp).T * ms, axis=1)
        PP = vr * self.rmass
        PP = PP[1, :] - PP[0, :]
        # angular and radial energy:
        II = self.rmass * np.linalg.norm(rr) ** 2
        RotE = np.dot(JJ, JJ) / (2 * II)
        RadE = 0.5 * np.linalg.norm(PP) ** 2 / self.rmass
        # imact parameter
        b = sum(0.5 * np.linalg.norm(JJ) / (np.linalg.norm(vv, axis=1) * ms))
        self.SampInfo["JJ"] = JJ
        self.SampInfo["II"] = II
        self.SampInfo["RoE"] = RotE
        self.SampInfo["RaE"] = RadE
        self.SampInfo["KiE"] = KE
        self.SampInfo["b"] = b
        self.slog += [
            "Total Angular Energy : " + "{0:10.5f}".format(RotE * au2ev) + "\n"
        ]
        self.slog += [
            "Total Radial  Energy : " + "{0:10.5f}".format(RadE * au2ev) + "\n"
        ]
        self.slog += [
            "Total           (eV) : " +
            "{0:10.5f}".format((RotE + RadE) * au2ev) + "\n"
        ]
        self.slog += [
            "Total Angular Momentum : "
            + "".join(["{0:10.5f}".format(p) + " " for p in JJ])
            + "\n"
        ]
        self.slog += [
            "Total Radial Momentum  : "
            + "".join(["{0:10.5f}".format(p) + " " for p in PP])
            + "\n"
        ]
        self.slog += ["Moment Of Inertia      : " +
                      "{0:10.3e}".format(II) + "\n"]
        self.slog += ["Impact Parameter (Ang) : " +
                      "{0:10.5f}".format(float(b)*au2ang) + " \n"]
        return

    def InitializeSample(self):
        """Initialize a scattering sample.

        This method initializes a new scattering sample, setting up necessary variables and data structures.
        """
        self.sii = len(self.sampls['cv'])
        self.slog = ["#####################################\n"]
        self.slog += ["## Sample Number " + str(self.sii) + "\n"]
        self.SampInfo = {}
        mol = self.mol
        self.smkin = [0.0,0.0]
        for i in range(2):
            mol[i].SampInfo = {}
            mol[i].InitializeSample()

    def GenerateSample(self):
        """Generate a scattering sample.

        This method generates a complete scattering sample, including vibrational, rotational, and orientational states, as well as intermolecular parameters.
        """
        self.log.append("Generating Sample number " + str(self.sii + 1) + "\n")
        # Initialize sample
        self.InitializeSample()
        self.SampleHOVibrState()
        self.SampleRigidRotorState()
        self.SampleOrientat()
        # sample intermolecular DOF
        self.SetInterZDist(self.Rz)
        self.SampleInterMolZVeloc()
        self.SampleImpactParam()
        # summarize energy from generated sample
        self.SummarizeLogEnergy(False)
        # summarize energy from calculated/analysed sample:
        self.AnalyseSample()
        self.ImageXYZOut(mess="Sample " + str(self.sii))
        open(
            self.dirout + "/" + self.fileout +
            "_" + str(self.sii) + ".info", "w"
        ).writelines(self.slog)

    def SummarizeLogEnergy(self,FromSample):
        """Summarize energy-related information.

        This method summarizes energy-related information, including vibrational, rotational, and velocity contributions.
        """
        ven1, ven2, ren1, ren2, ken1, ken2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if FromSample:
            if 'ViE' in self.mol[0].SampInfo.keys():
                ven1 = sum(self.mol[0].SampInfo['ViE']) * au2ev 
            if 'ViE' in self.mol[1].SampInfo.keys():
                ven2 = sum(self.mol[1].SampInfo['ViE']) * au2ev 
            if 'RoE' in self.mol[0].SampInfo.keys():
                ren1 = sum(self.mol[0].SampInfo['RoE']) * au2ev 
            if 'RoE' in self.mol[1].SampInfo.keys():
                ren2 = sum(self.mol[1].SampInfo['RoE']) * au2ev 
            if 'KiE' in self.SampInfo.keys():
                ken1 = self.SampInfo['KiE'][0] * au2ev
                ken2 = self.SampInfo['KiE'][1] * au2ev
            self.slog.append("########## Energy Decomposition (From Sample) ################ \n")
        else:
            if hasattr(self.mol[0], "sven"):
                ven1 = sum(self.mol[0].sven) * au2ev
            if hasattr(self.mol[1], "sven"):
                ven2 = sum(self.mol[1].sven) * au2ev
            if hasattr(self.mol[0], "sren"):
                ren1 = self.mol[0].sren * au2ev
            if hasattr(self.mol[1], "sren"):
                ren2 = self.mol[1].sren * au2ev
            if hasattr(self, "smkin"):
                ken1 = self.smkin[0] * au2ev
                ken2 = self.smkin[1] * au2ev
            self.slog.append("########## Energy Decomposition (From Analysis) ################ \n")
        self.slog.append(
            "              "
            + self.mol[0].name.rjust(20)
            + "  "
            + self.mol[1].name.rjust(20)
            + "               Total (eV) \n"
        )
        self.slog.append(
            "Vibrational  :"
            + "{0:8.4f}".format(ven1).rjust(20)
            + " "
            + "{0:8.4f}".format(ven2).rjust(20)
            + " "
            + "{0:8.4f}".format(ven1 + ven2).rjust(20)
            + "\n"
        )
        self.slog.append(
            "Rotational   :"
            + "{0:8.4f}".format(ren1).rjust(20)
            + " "
            + "{0:8.4f}".format(ren2).rjust(20)
            + " "
            + "{0:8.4f}".format(ren1 + ren2).rjust(20)
            + "\n"
        )
        self.slog.append(
            "Velocity     :"
            + "{0:8.4f}".format(ken1).rjust(20)
            + " "
            + "{0:8.4f}".format(ken2).rjust(20)
            + " "
            + "{0:8.4f}".format(ken1 + ken2).rjust(20)
            + "\n"
        )
        self.slog.append(
            "Total Energy :"
            + "{0:8.4f}".format(ken1 + ven1 + ren1).rjust(20)
            + " "
            + "{0:8.4f}".format(ken2 + ven2 + ren2).rjust(20)
            + " "
            + "{0:8.4f}".format(ken1 + ken2 + ven1 +
                                ven2 + ren1 + ren2).rjust(20)
            + "\n"
        )

    def Mol2Image(self):
        """Convert molecular coordinates and velocities to the image frame.

        This method transforms molecular coordinates and velocities to the image frame for calculations.
        """
        self.sxx = np.concatenate([self.mol[0].sxx, self.mol[1].sxx])
        self.svv = np.concatenate([self.mol[0].svv, self.mol[1].svv])

    def ImageXYZOut(self, **dic):
        """Generate output files for image coordinates and velocities.

        This method generates output files for image coordinates and velocities, storing them in the specified directory.
        """
        mol = self.mol
        if "mess" in dic.keys():
            message = dic["mess"]
        else:
            message = " "
        self.Mol2Image()
        self.slog += [" Sample " + str(self.sii) + " Coordinates (au) : \n"]
        xyz = XYZlist(self.el, self.sxx,
                      mess=message + " Coordinate (au)")
        vxyz = XYZlist(self.el, self.svv,
                       mess=message + " Velocities (au)")
        if not os.path.isdir(self.dirout):
            os.system("mkdir " + self.dirout)
        self.slog += xyz
        self.slog += [" Sample " +
                      str(self.sii) + " Velocities (au) : \n"]
        self.slog += vxyz[2:]
        open(
            self.dirout + "/" + self.fileout +
            "_" + str(self.sii) + ".xyz", "w"
        ).writelines(xyz)
        open(
            self.dirout + "/" + self.fileout +
            "_" + str(self.sii) + ".vel", "w"
        ).writelines(vxyz)
        # np.savetxt(self.dirout + '/' + self.fileout+'_'+str(self.sii)+'.vel2',self.svv)

    def SampleInterMolZVeloc(self):
        """Sample intermolecular z-velocity for the scattering event.

        This method samples the intermolecular z-velocity based on molecular velocities or direct input parameters.
        """
        # if the user chose intermolecular energy/velocity directly
        if hasattr(self, "velsamp"):
            V = self.velsamp[0][self.velsamp[-1]]
            self.velsamp[-1] += 1
            vv = self.GetInterMolZVeloc(V)
        else:
            # if user chose molecular temperatures/velocities
            v1, self.slog = self.mol[0].SampleZVeloc()
            v2, self.slog = self.mol[1].SampleZVeloc()
            if abs(v1) + abs(v2) == 0:
                self.slog += ["  -No intermolecular velocity \n"]
                self.smkin = [0.0, 0.0]
                return
            else:
                vv = self.GetInterMolZVelocFromMolV(v1, v2)
        mm = np.array([sum(self.mol[0].mass), sum(self.mol[1].mass)])
        self.smkin = np.sum(0.5 * vv.T**2 * mm, axis=0).tolist()
        vn = np.linalg.norm(vv)
        self.slog += [
            " -Z Inter-mol Velocity Sample (m/s): "
            + "{0:14.7f}".format(vv[0, 2]*au2mps)
            + " "
            + "{0:14.7f}".format(vv[1, 2]*au2mps)
            + " = "
            + "{0:14.7f}".format(vn * au2mps)
            + "\n"
        ]
        self.slog += [
            " -                           Energy: "
            + "{0:14.7f}".format(self.smkin[0] * au2ev)
            + " "
            + "{0:14.7f}".format(self.smkin[1] * au2ev)
            + " = "
            + "{0:14.7f}".format(sum(self.smkin) * au2ev)
            + " eV \n"
        ]
        self.mol[0].svv += vv[0]
        self.mol[1].svv += vv[1]
        return

    def SampleImpactParam(self):
        """Sample impact parameter and azimuthal scattering angle.

        This method samples impact parameter and azimuthal scattering angle for the scattering event.
        """
        if not hasattr(self, "bsamp"):
            self.slog += ["  -No impact parameter \n"]
            return
        b, chi = self.bsamp[0][self.bsamp[-1]], self.chisamp[self.bsamp[-1]]
        self.SetImpactParam(b, chi)
        self.bsamp[-1] += 1
        self.slog += [" -Impact Parameter : " +
                      str(float(b)*au2ang) + " Ang,  chi = " + str(chi) + "\n"]

    # in comes lab fixed z-velocity magnitude, out goes z-velocity
    # in centre of mass
    def GetInterMolZVelocFromMolV(self, V1, V2):
        """Get intermolecular z-velocity from molecular velocities.

        This method calculates the intermolecular z-velocity from the molecular velocities of two molecules.

        Args:
            V1 (float): Z-velocity of the first molecule.
            V2 (float): Z-velocity of the second molecule.

        Returns:
            numpy.ndarray: Inter-molecular z-velocity.
        """
        sy = np.zeros((2, 3))
        mm = np.array([sum(self.mol[0].mass), sum(self.mol[1].mass)])
        sy[0, :] = z * V1
        sy[1, :] = -z * V2
        vcom = COM(sy, mm)
        vv = sy - vcom.T
        return vv

    def GetInterMolZVeloc(self, V):
        """Calculate intermolecular z-velocity from given velocity magnitude.

        This method calculates the intermolecular z-velocity from a given velocity magnitude.

        Args:
            V (float): Inter-molecular velocity magnitude.

        Returns:
            numpy.ndarray: Inter-molecular z-velocity for both molecules.
        """
        vv = np.zeros((2, 3))
        vv[0, :] = -(z * V) * self.w1
        vv[1, :] = (z * V) * self.w0
        return vv

    def SetInterZDist(self, Rz):
        """Set the intermolecular z-coordinate distance.

        This method sets the intermolecular z-coordinate distance for both molecules.

        Args:
            Rz (float): Inter-molecular distance along the z-axis.
        """
        mol = self.mol
        mol[0].sxx += (z * Rz) * self.w1
        mol[1].sxx -= (z * Rz) * self.w0
        self.slog += [" -Z coordinate distance (Ang): " +
                      "{0:10.5f}".format(Rz*au2ang) + "\n"]

    def SetImpactParam(self, b, chi):
        """Set the impact parameter for scattering.

        This method sets the impact parameter and azimuthal angle (chi) for the scattering process.

        Args:
            b (float): Impact parameter.
            chi (float): Azimuthal angle.
        """
        mol = self.mol
        mol[0].sxx += (y * b * np.cos(chi) + x * b * np.sin(chi)) * self.w1
        mol[1].sxx -= (y * b * np.cos(chi) + x * b * np.sin(chi)) * self.w0

    def SampleRigidRotorState(self):
        """Sample the rigid rotor state for each molecule.

        This method samples the rigid rotor state for each molecule, considering their respective temperatures.
        """
        self.slog += [" -Rigid Rotor State Sample: \n"]
        mol = self.mol
        # self.slog += mol[0].SampleRigidRotorState()
        # self.slog += mol[1].SampleRigidRotorState()

    def SampleHOVibrState(self):
        """Sample harmonic oscillator vibrational states.

        This method samples the harmonic oscillator vibrational states for each molecule.
        """
        self.slog += [" -HO Vibrational State Sample: \n"]
        mol = self.mol
        self.slog += mol[0].SampleHOVibrState()
        self.slog += mol[1].SampleHOVibrState()

    def SampleOrientat(self):
        """Sample molecular orientation.

        This method samples molecular orientation in three-dimensional space for each molecule.
        """
        self.slog += [" -Orientational Sample: \n"]
        mol = self.mol
        self.slog += mol[0].SampleOrientat()
        self.slog += mol[1].SampleOrientat()

    def CalcRotEner(self):
        """Calculate rotational energies.

        This method calculates rotational energies for each molecule and the overall system.
        """
        self.slog.append(" Rotational Analysis : \n")
        mol = self.mol
        self.slog += mol[0].CalcRotEner()
        self.slog += mol[1].CalcRotEner()

    def CalcInterEner(self):
        """Calculate vibrational energies.

        This method calculates vibrational energies for each molecule and the overall system.
        """
        self.slog.append(" Vibrational Analysis : \n")
        mol = self.mol
        self.slog += mol[0].CalcInterEner()
        self.slog += mol[1].CalcInterEner()

    def CalcOrient(self):
        """Calculate molecular orientation.

        This method calculates the molecular orientation and Euler angles for each molecule.
        """
        self.slog.append(" Orientation Analysis : \n")
        mol = self.mol
        self.slog += mol[0].CalcOrient()
        self.slog += mol[1].CalcOrient()

    def GenSamples(self, **dic):
        """Generate multiple scattering samples.

        This method generates multiple scattering samples, allowing for the customization of the number of samples and other parameters.

        Args:
            dic (dict): Additional parameters for sample generation.
        """
        if "N" in dic.keys():
            N = dic["N"]
        else:
            N = self.Nsamp
        print("Generating " + str(N) + " Samples, this may take a while ...")
        for i in range(N):
            self.GenerateSample()

    def AnalyseSample(self):
        """
        Analyze a sample by performing various calculations and generating a summary.

        This method calculates various physical properties of the sample and appends the results to a log.

        It consists of the following steps:
        1. Calculate Euler orientation of the molecular frame from position and velocity (self.CalcOrient()).
        2. Calculate rotational information from position and velocity (self.CalcRotEner()).
        3. Calculate vibrational information from position and velocity (self.CalcInterEner()).
        4. Calculate intermolecular information from position and velocity (self.CalcInterMolMomentum()).
        5. Summarize and log energy information (self.SummarizeLogEnergy()).
        6. Generate output including final coordinates and velocity.

        The results are stored in the 'slog', 'sampls', and 'SampInfo' attributes of the object.

        Returns:
        None
        """
        self.slog.append(
            "####################### Sample " +
            str(self.sii) + " Final Analysis:\n"
        )
        # Calculates Euler orientation of molecular frame from xx and vv
        self.CalcOrient()
        # Calculates Rotational information from xx and vv
        self.CalcRotEner()
        # Calculates Vibrational information from xx and vv
        self.CalcInterEner()
        # Calculate InterMolecular information from xx and vv
        self.CalcInterMolMomentum()
        # write up some energy information
        self.SummarizeLogEnergy(True)
        # generate output
        self.slog.append(
            "####################### Sample "
            + str(self.sii)
            + " Final Coordinates/Velocity:\n"
        )
        self.sampls['cv'].append([self.sxx.copy(), self.svv.copy()])
        self.sampls['info'].append(self.SampInfo.copy())

    def ReadSamples(self, filx, filv):
        """
        Read sample data from files, analyze each sample, and write results to an output file.

        Parameters:
        filx (str): The filename containing position data (XYZs).
        filv (str): The filename containing velocity data.

        This method reads data from 'filx' and 'filv', assigns the data to the appropriate properties
        of the 'mol' object, and then analyzes each sample using the 'AnalyseSample' method. The results
        are stored in 'self.slog' and written to an output file named '[self.fileout]_samples.info'.

        Note:
        - The 'ReadXYZs' function is used to read position and velocity data from the specified files.
        - The 'AnalyseSample' method is called for each sample to perform the analysis.

        Returns:
        None
        """
        el, xyzs, xmess = ReadXYZs(filx)
        el, vels, vmess = ReadXYZs(filv)
        mol = self.mol
        for i, x in enumerate(xyzs):
            v = vels[i]
            m = xmess[i] + ' & ' + vmess[i]
            mol[0].sxx = x[:mol[0].na, :]*ang2au 
            mol[1].sxx = x[mol[0].na:, :]*ang2au 
            mol[0].svv = v[:mol[0].na, :]*ang2au/fmt2au
            mol[1].svv = v[mol[0].na:, :]*ang2au/fmt2au
            self.slog += [ '###### Sample Name ' + m + '\n']
            self.AnalyseSample()


# Example Usage:
if __name__ == "__main__":
    # Create an instance of the iscatter class
    sc = iscatter()

    # Read input data from a file
    sc.ReadInput("input.txt")

    # Generate scattering samples
    sc.GenSamples(N=10)
