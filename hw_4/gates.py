import numpy as np

class Gate:
    """Базовый класс для квантовых гейтов"""
    def __init__(self, matrix, name):
        self._matrix = np.array(matrix)
        self._name = name

    @property
    def matrix(self):
        """Возвращает матрицу гейта"""
        return self._matrix

    @property
    def name(self):
        """Возвращает имя гейта"""
        return self._name

    def __repr__(self):
        return f"Gate: {self._name}"
    
class I(Gate):
    def __init__(self):
        super().__init__([[1, 0], [0, 1]], "I")

class X(Gate):
    def __init__(self):
        super().__init__([[0, 1], [1, 0]], "X")

class H(Gate):
    def __init__(self):
        super().__init__(
            [[1, 1], [1, -1]] / np.sqrt(2), "H"
        )

class RX(Gate):
    def __init__(self, theta):
        matr = np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)]
        ], dtype=complex)
        super().__init__(matr, f"RX({np.round(theta, 2)})")

class RZ(Gate):
    def __init__(self, theta):
        matr = np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ], dtype=complex)
        super().__init__(matr, f"RZ({np.round(theta, 2)})")

class ZZ(Gate):
    def __init__(self, theta):
        matr = np.diag([np.exp(-1j * theta), np.exp(1j * theta), np.exp(1j * theta), np.exp(-1j * theta)]).reshape(2,2,2,2)
        super().__init__(matr, f"ZZ({np.round(theta, 2)})")


class Kraus_1_1q(Gate):
    def __init__(self, beta):
        matr = 1/np.sqrt(2 * (1 - np.cos(beta))) * np.array([[1j * np.sin(beta), -(1 - np.cos(beta))], [-(1 - np.cos(beta)), 1j * np.sin(beta)]], dtype=complex)
        # matr = RX(beta).matrix
        super().__init__(
            matr, f"K1_1q ({np.round(beta, 2)})"
        )

class Kraus_2_1q(Gate):
    def __init__(self, beta):
        matr = 1/np.sqrt(2 * (1 + np.cos(beta))) * np.array([[1j * np.sin(beta), 1 + np.cos(beta)], [1 +np.cos(beta), 1j * np.sin(beta)]], dtype=complex)
        # matr = X.matrix @ RX(beta).matrix 
        super().__init__(
            matr, f"K2_1q ({np.round(beta, 2)})"
        )


class Kraus_1_2q(Gate):
    def __init__(self, gamma):
        matr = np.diag([np.exp(-1j * gamma), np.exp(1j * gamma), np.exp(1j * gamma), np.exp(-1j * gamma)]).reshape(2,2,2,2)
        super().__init__(
            matr, f"K1_2q ({np.round(gamma, 2)})"
        )

class Kraus_2_2q(Gate):
    def __init__(self, gamma):
        matr = np.diag([np.exp(-1j * gamma), -np.exp(1j * gamma), -np.exp(1j * gamma), np.exp(-1j * gamma)]).reshape(2,2,2,2)
        super().__init__(
            matr, f"K2_2q ({np.round(gamma, 2)})"
        )

class G_1q(Gate):
    def __init__(self, beta, sigma):
        C = np.cos(beta) * np.exp(-sigma**2/2)
        S = np.sin(beta) * np.exp(-sigma**2/2)
        matr = 0.5 * np.array([[1 + C, -1j * S, 1j * S, 1-C],
                               [-1j*S, 1+C, 1-C, 1j*S],
                               [1j*S, 1-C, 1+C, -1j*S],
                               [1 - C, 1j*S, -1j*S, 1+C]], dtype=complex).reshape(2,2,2,2)
        super().__init__(
            matr, f"G_1q ({np.round(beta, 2)})"
        )

class G_2q(Gate):
    def __init__(self, gamma, sigma):
    
        matr = np.diag([1, np.exp(-2 * sigma**2 + 2j * gamma), np.exp(-2 * sigma**2 + 2j * gamma), 1,
                        np.exp(-2 * sigma**2 - 2j * gamma), 1, 1, np.exp(-2 * sigma**2 - 2j * gamma),
                        np.exp(-2 * sigma**2 - 2j * gamma), 1, 1, np.exp(-2 * sigma**2 - 2j * gamma),
                        1, np.exp(-2 * sigma**2 + 2j * gamma), np.exp(-2 * sigma**2 + 2j * gamma), 1]).reshape(2,2,2,2,2,2,2,2)
        super().__init__(
            matr, f"G_1q ({np.round(gamma, 2)})"
        )