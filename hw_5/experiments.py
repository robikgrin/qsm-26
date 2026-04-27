import numpy as np
from quantum_sim import EinsumQuantumSimulator
from mps import MPS_simulator
from gates import H, ZZ, RX
import time


def einsum_QAOA(num_qubits:int = 15, p: int = 1, seed: int =42):
    np.random.seed(seed)
    start = time.time()
    einsum_sim = EinsumQuantumSimulator(num_qubits)

    for i in range(einsum_sim.num_qubits):
        einsum_sim.one_qubit_gate(i, H())

    for _ in range(p):
        beta = np.random.uniform(0.01, np.pi, size = (num_qubits,))
        gamma = np.random.uniform(0.01, 2*np.pi, size = (num_qubits-1, ))

        for i in range(einsum_sim.num_qubits - 1):
            einsum_sim.two_qubit_gate(i, i+1, ZZ(gamma[i]))
            
        for i in range(einsum_sim.num_qubits):
            einsum_sim.one_qubit_gate(i, RX(beta[i]))

    # Итоговый вектор состояния
    true_vector = einsum_sim.get_vector().flatten()

    duration = time.time() - start

    return true_vector, duration


def mps_QAOA(num_qubits:int = 15, p: int = 1, max_bond: int = 10, seed: int =42):
    np.random.seed(seed)

    start = time.time()
    mps_sim = MPS_simulator(num_qubits=num_qubits, r=max_bond)

    for i in range(mps_sim.num_qubits):
        mps_sim.one_qubit_gate(i, H())

    for _ in range(p):
        beta = np.random.uniform(0.01, np.pi, size = (num_qubits,))
        gamma = np.random.uniform(0.01, 2*np.pi, size = (num_qubits-1, ))

        for i in range(mps_sim.num_qubits - 1):
            mps_sim.two_qubit_gate(i, i+1, ZZ(gamma[i]))
            
        for i in range(mps_sim.num_qubits):
            mps_sim.one_qubit_gate(i, RX(beta[i]))

    # Итоговый вектор состояния
    true_vector = mps_sim.get_vector()

    duration = time.time() - start

    return true_vector, duration