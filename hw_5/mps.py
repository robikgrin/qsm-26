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

class MPS_simulator(QuantumSimulator):
    def __init__(self, num_qubits = 2, r = 4):
        self.num_qubits = num_qubits
        self.gammas, self.lambdas = self.init_mps_state(num_qubits)
        self.r = r
        self._gates = []
    
    def init_mps_state(self, n_qubits):
        gammas = []
        lambdas = []
        
        # Создаем тензоры гамма для каждого кубита
        for i in range(n_qubits):
            gamma = np.zeros((1, 1, 2), dtype=np.complex128)
            gamma[0, 0, 0] = 1
            gammas.append(gamma)
        
        for i in range(n_qubits - 1):
            lam = np.array([1], dtype=np.complex128)
            lambdas.append(lam)
            
        return gammas, lambdas
    
    def one_qubit_gate(self, qubit, gate):
        gamma_q = self.gammas[qubit]
        
        self.gammas[qubit] = np.tensordot(gamma_q, gate.matrix, axes=([2], [1]))
        self._gates.append({"name": gate.name, "qubits": [qubit]})
    
    def two_qubit_gate(self, qubit_1, qubit_2, gate):
        q_left = min(qubit_1, qubit_2)
        q_right = max(qubit_1, qubit_2)

        # Получаем тензоры гамма и лямбда
        gamma_L = self.gammas[q_left]
        gamma_R = self.gammas[q_right]
        lam_mid = self.lambdas[q_left]

        # Также берем предыдущие лямбда для левого и следующие для правого
        lam_L = self.lambdas[q_left - 1] if q_left > 0 else np.array([1.0], dtype=np.complex128)
        lam_R = self.lambdas[q_right] if q_right < self.num_qubits - 1 else np.array([1.0], dtype=np.complex128)
       
        # Левое сворачивание
        A_L = gamma_L * lam_L[:, None, None] * lam_mid[None, :, None]
        
        # Правое сворачивание
        B_R = gamma_R * lam_R[None, :, None]

        # Получение тензора тета
        theta = np.einsum('abk, bcl, ijkl -> aijc', A_L, B_R, gate.matrix)
        
        # Новые тензоры после SVD
        new_gamma_L, new_lam, new_gamma_R = self.truncate_and_split(theta, self.r, lam_L, lam_R)

        self.gammas[q_left] = new_gamma_L
        self.lambdas[q_left] = new_lam
        self.gammas[q_right] = new_gamma_R 
        
        self._gates.append({"name": getattr(gate, "name", "2Q"), "qubits": [q_left, q_right]})


    @staticmethod
    def truncate_and_split(theta, max_r, lam_L, lam_R):
        chi_L, phys_L, phys_R, chi_R = theta.shape
        
        M = theta.reshape(chi_L * phys_L, phys_R * chi_R)
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        
        r_new = min(len(S), max_r)
        U_trunc = U[:, :r_new]
        S_trunc = S[:r_new]
        Vh_trunc = Vh[:r_new, :]
        
        # Обязательная нормировка (сохранение нормализации)
        norm = np.linalg.norm(S_trunc)
        new_lam = S_trunc / norm if norm > 0 else S_trunc
        
        # Преобразуем матрицы в тензоры вида как гамма
        U_reshaped = U_trunc.reshape(chi_L, phys_L, r_new)
        Vh_reshaped = Vh_trunc.reshape(r_new, phys_R, chi_R)
        
        # Берем обратные значения лямбда для нормализации гамма-тензоров
        threshold = 1e-8
        s_l_inv = np.where(np.abs(lam_L) > threshold, 1.0 / lam_L, 0.0)
        s_r_inv = np.where(np.abs(lam_R) > threshold, 1.0 / lam_R, 0.0)
        
        # Нормализуем новые гамма-тензоры
        new_gamma_L = U_reshaped * s_l_inv[:, None, None]
        new_gamma_R = Vh_reshaped * s_r_inv[None, None, :]
        
        # Ставим индексы правильным образом
        new_gamma_L = new_gamma_L.transpose(0, 2, 1)
        new_gamma_R = new_gamma_R.transpose(0, 2, 1)
        
        return new_gamma_L, new_lam, new_gamma_R
    
    def get_vector(self):
        state = self.gammas[0][0, :, :].transpose(1, 0)
        for i in range(1, self.num_qubits):
            lam = self.lambdas[i-1]
            gamma = self.gammas[i]
            state = state * lam
            
            state = np.tensordot(state, gamma, axes=([-1], [0]))
            state = np.swapaxes(state, -1, -2)
        return state.reshape(-1)
    
    def get_probs(self):
        state_vector = self.get_vector()
        probs = np.abs(state_vector)**2
        return probs
    
    def get_state(self):
        state_vector = self.get_vector()
        state_ampl = dict()
        for i in range(0, 1 << self.num_qubits):
            s = f"{i:b}"
            s = "0" * (self.num_qubits - len(s)) + s
            state_ampl[s] = self.state_vector.flatten()[i]
        return state_ampl
    