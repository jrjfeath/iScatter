from numpy import ndarray, zeros, linspace, matmul, sqrt, diag

def J2(j: int) -> ndarray:
    """
    Calculate the square of the total angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The squared total angular momentum operator.
    """
    return Jz2(j) + Jx2(j) + Jy2(j) 

def Jz2(j: int) -> ndarray:
    """
    Calculate the square of the z-component of the angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The squared z-component of the angular momentum operator.
    """
    return matmul(J_z(j), J_z(j))

def Jx2(j: int) -> ndarray:
    """
    Calculate the square of the x-component of the angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The squared x-component of the angular momentum operator.
    """
    return matmul(J_x(j), J_x(j))

def Jy2(j: int) -> ndarray:
    """
    Calculate the square of the y-component of the angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The squared y-component of the angular momentum operator.
    """
    return matmul(J_y(j), J_y(j))

def J_x(j: int) -> ndarray:
    """
    Calculate the x-component of the angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The x-component of the angular momentum operator.
    """
    return (J_plus(j) + J_minus(j)) / 2

def J_y(j: int) -> ndarray:
    """
    Calculate the y-component of the angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The y-component of the angular momentum operator.
    """
    return (J_plus(j) - J_minus(j)) / (2j)

def J_z(j: int) -> ndarray:
    """
    Calculate the z-component of the angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The z-component of the angular momentum operator.
    """
    return diag([-j + i for i in range(int(2 * j + 1))])

def Jx(j: int) -> ndarray:
    """
    Alias for J_x(j).

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The x-component of the angular momentum operator.
    """
    return J_x(j)

def Jy(j: int) -> ndarray:
    """
    Alias for J_y(j).

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The y-component of the angular momentum operator.
    """
    return J_y(j)

def Jz(j: int) -> ndarray:
    """
    Alias for J_z(j).

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The z-component of the angular momentum operator.
    """
    return J_z(j)

def P2(j: int) -> ndarray:
    """
    Calculate the square of the total angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The squared total angular momentum operator.
    """
    return J2(j) 

def Pz2(j: int) -> ndarray:
    """
    Calculate the square of the z-component of the angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The squared z-component of the angular momentum operator.
    """
    return matmul(P_z(j), P_z(j))

def Px2(j: int) -> ndarray:
    """
    Calculate the square of the x-component of the angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The squared x-component of the angular momentum operator.
    """
    return matmul(P_x(j), P_x(j))

def Py2(j: int) -> ndarray:
    """
    Calculate the square of the y-component of the angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The squared y-component of the angular momentum operator.
    """
    return matmul(P_y(j), P_y(j))

def Pp2(j: int) -> ndarray:
    """
    Calculate the square of the raising operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The squared raising operator.
    """
    return matmul(P_plus(j), P_plus(j))

def Pm2(j: int) -> ndarray:
    """
    Calculate the square of the lowering operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The squared lowering operator.
    """
    return matmul(P_minus(j), P_minus(j))

def P_x(j: int) -> ndarray:
    """
    Calculate the x-component of the angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The x-component of the angular momentum operator.
    """
    return (P_plus(j) + P_minus(j)) / 2

def P_y(j: int) -> ndarray:
    """
    Calculate the y-component of the angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The y-component of the angular momentum operator.
    """
    return (P_plus(j) - P_minus(j)) / (2j)

def P_z(j: int) -> ndarray:
    """
    Calculate the z-component of the angular momentum operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The z-component of the angular momentum operator.
    """
    return diag([-j + i for i in range(int(2 * j + 1))])

def Px(j: int) -> ndarray:
    """
    Alias for P_x(j).

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The x-component of the angular momentum operator.
    """
    return P_x(j)

def Py(j: int) -> ndarray:
    """
    Alias for P_y(j).

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The y-component of the angular momentum operator.
    """
    return P_y(j)

def Pz(j: int) -> ndarray:
    """
    Alias for P_z(j).

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The z-component of the angular momentum operator.
    """
    return P_z(j)

def P_minus(j: int) -> ndarray:
    """
    Calculate the lowering operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The lowering operator.
    """
    if int(2 * j + 1) != 2 * j + 1:
        raise ValueError(f"j must be a half-integer. Found: {j}")
    dim = int(2 * j + 1)
    mat = zeros((dim, dim))
    m_prime_list = linspace(-j, j, dim)
    m_list = linspace(-j, j, dim)
    for row, m_prime in enumerate(m_prime_list):
        for col, m in enumerate(m_list):
            mat[row, col] = J_plus_component(j, m_prime, j, m)
    return mat

def P_plus(j: int) -> ndarray:
    """
    Calculate the raising operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The raising operator.
    """
    if int(2 * j + 1) != 2 * j + 1:
        raise ValueError(f"j must be a half-integer. Found: {j}")
    dim = int(2 * j + 1)
    mat = zeros((dim, dim))
    m_prime_list = linspace(-j, j, dim)
    m_list = linspace(-j, j, dim)
    for row, m_prime in enumerate(m_prime_list):
        for col, m in enumerate(m_list):
            mat[row, col] = J_minus_component(j, m_prime, j, m)
    return mat

def J_plus(j: int) -> ndarray:
    """
    Calculate the raising operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The raising operator.
    """
    if int(2 * j + 1) != 2 * j + 1:
        raise ValueError(f"j must be a half-integer. Found: {j}")
    dim = int(2 * j + 1)
    mat = zeros((dim, dim))
    m_prime_list = linspace(-j, j, dim)
    m_list = linspace(-j, j, dim)
    for row, m_prime in enumerate(m_prime_list):
        for col, m in enumerate(m_list):
            mat[row, col] = J_plus_component(j, m_prime, j, m)
    return mat

def J_minus(j: int) -> ndarray:
    """
    Calculate the lowering operator for a given quantum number.

    Parameters:
    j (int): Quantum number.

    Returns:
    ndarray: The lowering operator.
    """
    if int(2 * j + 1) != 2 * j + 1:
        raise ValueError(f"j must be a half-integer. Found: {j}")
    dim = int(2 * j + 1)
    mat = zeros((dim, dim))
    m_prime_list = linspace(-j, j, dim)
    m_list = linspace(-j, j, dim)
    for row, m_prime in enumerate(m_prime_list):
        for col, m in enumerate(m_list):
            mat[row, col] = J_minus_component(j, m_prime, j, m)
    return mat

def J_plus_component(j_prime: int, m_prime: int, j: int, m: int) -> float:
    """
    Get the matrix element of the raising operator.

    Parameters:
    j_prime (int): Quantum number of the final state.
    m_prime (int): Magnetic quantum number of the final state.
    j (int): Quantum number of the initial state.
    m (int): Magnetic quantum number of the initial state.

    Returns:
    float: The matrix element of the raising operator.
    """
    if (j_prime != j) or (m_prime != m + 1):
        return 0
    return J_plus_coefficient(j, m)

def J_minus_component(j_prime: int, m_prime: int, j: int, m: int) -> float:
    """
    Get the matrix element of the lowering operator.

    Parameters:
    j_prime (int): Quantum number of the final state.
    m_prime (int): Magnetic quantum number of the final state.
    j (int): Quantum number of the initial state.
    m (int): Magnetic quantum number of the initial state.

    Returns:
    float: The matrix element of the lowering operator.
    """
    if (j_prime != j) or (m_prime != m - 1):
        return 0
    return J_minus_coefficient(j, m)

def J_plus_coefficient(j: int, m: int) -> float:
    """
    Calculate the coefficient for the raising operator.

    Parameters:
    j (int): Quantum number.
    m (int): Magnetic quantum number.

    Returns:
    float: The coefficient for the raising operator.
    """
    return sqrt((j - m) * (j + m + 1))

def J_minus_coefficient(j: int, m: int) -> float:
    """
    Calculate the coefficient for the lowering operator.

    Parameters:
    j (int): Quantum number.
    m (int): Magnetic quantum number.

    Returns:
    float: The coefficient for the lowering operator.
    """
    return sqrt((j + m) * (j - m + 1))

