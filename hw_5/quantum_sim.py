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