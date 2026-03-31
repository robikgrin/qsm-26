from gates import *

from abc import ABC, abstractmethod

class QuantumSimulator(ABC):
    @abstractmethod
    def one_qubit_gate(self, qubit, gate):
        pass

    @abstractmethod
    def two_qubit_gate(self, qubit_1, qubit_2, gate):
        pass

    @abstractmethod
    def get_vector(self, qubit):
        pass

    @abstractmethod
    def get_state(self, qubit):
        pass


class EinsumQuantumSimulator(QuantumSimulator):
    def __init__(self, num_qubits = 2):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros((2,)*self.num_qubits, dtype=complex)
        self.state_vector[(0,)*self.num_qubits] = 1
        self._gates = []
    
    @property
    def gates(self):
        return self._gates
    
    def one_qubit_gate(self, qubit, gate):
        # Формируем числовые списки осей
        state_indices = list(range(self.num_qubits))
        out_indices = list(range(self.num_qubits))
        
        # Индекс для нового состояния кубита (выбираем число вне диапазона 0...N-1)
        new_idx = self.num_qubits 
        gate_indices = [new_idx, qubit]
        out_indices[qubit] = new_idx

        # Вызываем einsum с числовыми массивами и включаем оптимизацию
        self.state_vector = np.einsum(
            gate.matrix, gate_indices,
            self.state_vector, state_indices,
            out_indices,
            optimize=True
        )
        self._gates.append({"name": gate.name, "qubits": [qubit]})

    def two_qubit_gate(self, qubit_1, qubit_2, gate):
        state_indices = list(range(self.num_qubits))
        out_indices = list(range(self.num_qubits))
        
        new_idx_1 = self.num_qubits
        new_idx_2 = self.num_qubits + 1
        
        gate_indices = [new_idx_1, new_idx_2, qubit_1, qubit_2]
        out_indices[qubit_1] = new_idx_1
        out_indices[qubit_2] = new_idx_2

        self.state_vector = np.einsum(
            gate.matrix, gate_indices,
            self.state_vector, state_indices,
            out_indices,
            optimize=True
        )
        self._gates.append({"name": gate.name, "qubits": [qubit_1, qubit_2]})
    
    def set_all_zeros(self, num_qubits):
        '''
        Обнуляет все данные и ставит равными нулю
        '''
        self.num_qubits = num_qubits
        self.state_vector = np.zeros((2,)*self.num_qubits, dtype=complex)
        self.state_vector[(0,)*self.num_qubits] = 1
        self._gates = []

    def set_vector(self, vector):
        self.state_vector = vector.reshape((2,)*self.num_qubits)
        self._gates = []

    def get_vector(self):
        return self.state_vector
    
    def get_probs(self):
        probs = np.abs(self.state_vector)**2
        return probs
    
    def get_state(self):
        state_ampl = dict()
        for i in range(0, 1 << self.num_qubits):
            s = f"{i:b}"
            s = "0" * (self.num_qubits - len(s)) + s
            state_ampl[s] = self.state_vector.flatten()[i]
        return state_ampl

class DensityMatrixSimulator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        # Инициализируем матрицу плотности |0...0><0...0|
        self.rho = np.zeros((2,) * (2 * num_qubits), dtype=complex)
        self.rho[(0,) * (2 * num_qubits)] = 1.0

    def one_qubit_superoperator(self, qubit, superoperator):
        cov_contr_indices = list(range(2*self.num_qubits))
        out_indices = list(range(2*self.num_qubits))
        new_idx_1 = 2*self.num_qubits
        new_idx_2 = new_idx_1 + 1
        
        # Индексы: [out_ket, out_bra, in_ket, in_bra]
        superop_indices = [new_idx_1, new_idx_2, qubit, qubit + self.num_qubits]
        
        out_indices[qubit] = new_idx_1
        out_indices[qubit + self.num_qubits] = new_idx_2

        self.rho = np.einsum(
            superoperator.matrix, superop_indices,
            self.rho, cov_contr_indices,
            out_indices,
            optimize=True
        )

    def two_qubits_superoperator(self, qubit_1, qubit_2, superoperator):
        cov_contr_indices = list(range(2*self.num_qubits))
        out_indices = list(range(2*self.num_qubits))
        new_idx_11 = 2*self.num_qubits
        new_idx_12 = new_idx_11 + 1
        new_idx_21 = new_idx_12 + 1
        new_idx_22 = new_idx_21 + 1
        
        superop_indices = [
            new_idx_11, new_idx_12, new_idx_21, new_idx_22, 
            qubit_1, qubit_2, qubit_1 + self.num_qubits, qubit_2 + self.num_qubits
        ]
        
        out_indices[qubit_1] = new_idx_11
        out_indices[qubit_2] = new_idx_12
        out_indices[qubit_1+self.num_qubits] = new_idx_21
        out_indices[qubit_2+self.num_qubits] = new_idx_22

        self.rho = np.einsum(
            superoperator.matrix, superop_indices,
            self.rho, cov_contr_indices,
            out_indices,
            optimize=True
        )

    def set_all_zeros(self, num_qubits):
        '''
        Обнуляет все данные и ставит равными нулю
        '''
        self.num_qubits = num_qubits
        self.rho = np.zeros((2,) * (2 * num_qubits), dtype=complex)
        self.rho[(0,) * (2 * num_qubits)] = 1.0
    
    def set_density_matrix(self, rho):
        self.rho = rho.reshape((2,) * (2 * self.num_qubits))

    def get_density_matrix_tensor(self):
        return self.rho
    
    def get_probs(self):
        diag_elements = self.rho.reshape((2**self.num_qubits, 2**self.num_qubits)).diagonal()
        
        # Делаем reshape в N-мерный тензор
        probs_tensor = diag_elements.reshape((2,) * self.num_qubits)
        
        return np.real(probs_tensor) # complex -> real
        
    def get_mat(self):
        return self.rho.reshape((2**self.num_qubits, 2**self.num_qubits))

def print_state(slov):
    for state in slov.keys():
        print(f'State: {state}\t Amplitude: {slov[state]}\t Probability: {np.abs(slov[state])**2}')

def check_vector(vector):
    norm = np.linalg.norm(vector)
    if np.isclose(norm, 1):
        print("The state vector is normalized.")
    else:
        print(f"The state vector is not normalized. Norm = {norm}")