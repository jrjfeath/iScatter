import numpy as np
from periodictable import elements

# Conversion factors
fmt2au, au2fmt = 41.3413745, 1.0 / 41.3413745
cm2au, au2cm = 1.0 / 219474.6305, 219474.6305
amu2au, au2amu = 1822.89, 1.0 / 1822.89
ang2au, au2ang = 1.88973, 1.0 / 1.88973
ev2au, au2ev = 0.0367494, 1.0 / 0.0367494
au2mps = 1.0 / (18897259886.0 / 4.1341374575751e+16)
mps2au = 1.0 / au2mps
kboltz = 3.166811563e-6

# Mathematical constants
pi = np.pi
tpi = np.pi * 2
hpi = np.pi * 0.5

# Unit vectors
x = np.array([1.0, 0.0, 0.0])
y = np.array([0.0, 1.0, 0.0])
z = np.array([0.0, 0.0, 1.0])

# Generate dictionaries containing symbol, mass, and number
an2el = {}
el2m = {}
for element in elements:
    an2el[element.number] = element.symbol
    el2m[element.symbol] = element.mass

