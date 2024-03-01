import numpy as np

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

# Atomic number to element mapping
an2el = {
    1: 'H',
    2: 'He',
    3: 'Li',
    4: 'Be',
    5: 'B',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    10: 'Ne'
}

# Element to mass mapping
el2m = {
    'H': 1.00784,
    'He': 4.0026,
    'Li': 6.9410,
    'Be': 9.0122,
    'B': 10.811,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'F': 18.998,
    'Ne': 20.180
}

