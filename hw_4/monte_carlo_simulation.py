from quantum_sim import EinsumQuantumSimulator
from gates import H, ZZ, RX
import numpy as np

def monte_carlo_probs(num_qubits: int = 6, p: int = 5, num_samples: int = 100, sigma_1q: float = 0.2, sigma_2q: float = 0.1, seed: int = 42):
    np.random.seed(seed)

    # Начальные значения параметров, которые служат средним для гейтов
    beta_0 = np.random.uniform(0.01, np.pi, size = (num_qubits,))
    gamma_0 = np.random.uniform(0.01, 2*np.pi, size = (num_qubits * (num_qubits-1)//2, ))

    # Один симулятор для задачи начального состояния, второй - манипуляция с матрицей плотности
    sim_state = EinsumQuantumSimulator(num_qubits=num_qubits)

    # Схема QAOA
    fin_probs = np.zeros((2,)*num_qubits)
    fin_probs_sq = np.zeros((2,)*num_qubits)
    for _ in range(num_samples):
        # Приготавливаем состояние |++++++>
        for q in range(num_qubits):
            sim_state.one_qubit_gate(q, H())
        for _ in range(p):
            k = 0
            for i in range(num_qubits):
                for j in range(i+1, num_qubits):
                    gamma = np.random.normal(gamma_0[k], sigma_2q)
                    sim_state.two_qubit_gate(i, j, ZZ(gamma))
                    k+=1

            for i in range(num_qubits):
                beta = np.random.normal(beta_0[i], sigma_1q)
                sim_state.one_qubit_gate(i, RX(beta))

        # Получаем вероятности для всех базисных состояний
        fin_probs += sim_state.get_probs()
        fin_probs_sq += sim_state.get_probs()**2
        sim_state.set_all_zeros(num_qubits)

    mean_probs = fin_probs / num_samples
    var_probs = (fin_probs_sq / num_samples) - (mean_probs**2)
    std_err_of_mean = np.sqrt(var_probs / num_samples)
        
    return mean_probs, std_err_of_mean

if __name__ == '__main__':
    num_qubits = 6
    probs_tensor = monte_carlo_probs(num_qubits=num_qubits, p=5, sigma_1q=0.2, sigma_2q=0.1, seed=42)
    print("Вероятности для всех базисных состояний:")
    checker = 0
    for i in range(1 << num_qubits):
        s = f"{i:0{num_qubits}b}"
        idx = tuple((i >> (num_qubits - 1 - j)) & 1 for j in range(num_qubits))
        checker += probs_tensor[idx]
        print(f"|{s}>: {probs_tensor[idx]:.6f}")
    print("Probs sum", checker)