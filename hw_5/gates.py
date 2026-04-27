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