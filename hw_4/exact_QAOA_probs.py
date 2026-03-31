from quantum_sim import EinsumQuantumSimulator, DensityMatrixSimulator
from gates import H, G_1q, G_2q
import numpy as np

def exact_probs(num_qubits: int = 6, p: int = 5, sigma_1q: float = 0.2, sigma_2q: float = 0.1, seed: int = 42):
    np.random.seed(seed)

    # Начальные значения параметров, которые служат средним для гейтов
    beta_0 = np.random.uniform(0, 2*np.pi, size = (num_qubits,))
    gamma_0 = np.random.uniform(0, 2*np.pi, size = ((num_qubits * (num_qubits - 1))//2, ))

    # Один симулятор для задачи начального состояния, второй - манипуляция с матрицей плотности
    sim_dense = DensityMatrixSimulator(num_qubits=num_qubits)
    sim_state = EinsumQuantumSimulator(num_qubits=num_qubits)

    # Приготавливаем состояние |++++++>
    for q in range(num_qubits):
        sim_state.one_qubit_gate(q, H())

    # Получаем матрицу плотности 
    rho_exact = np.outer(sim_state.state_vector.flatten(), sim_state.state_vector.flatten().conj())

    sim_dense.set_density_matrix(rho_exact)

    # Схема QAOA

    for _ in range(p):
        k = 0
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                sim_dense.two_qubits_superoperator(i, j, G_2q(gamma_0[k], sigma_2q))
                k+=1

        for i in range(num_qubits):
            sim_dense.one_qubit_superoperator(i, G_1q(beta_0[i], sigma_1q))

    # Получаем вероятности для всех базисных состояний
    probs_tensor = sim_dense.get_probs()
    return probs_tensor


if __name__ == '__main__':
    num_qubits = 6
    probs_tensor = exact_probs(num_qubits=num_qubits, p=5, sigma_1q=0.2, sigma_2q=0.1, seed=42)
    print("Вероятности для всех базисных состояний:")
    checker = 0
    for i in range(1 << num_qubits):
        s = f"{i:0{num_qubits}b}"
        idx = tuple((i >> (num_qubits - 1 - j)) & 1 for j in range(num_qubits))
        checker += probs_tensor[idx]
        print(f"|{s}>: {probs_tensor[idx]:.6f}")
    print("Probs sum", checker)