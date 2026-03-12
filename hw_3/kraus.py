import numpy as np
##### НАШИ НАЧАЛЬНЫЕ ДАННЫЕ #####

### ШУМ ###
mu = np.pi/10
sigma = 0.1

### УГОЛ ПОВОРОТА КУБИТА ###
theta = np.pi/4

### НАПРАВЛЕНИЕ НОРМАЛИ ###
phi = np.pi/3

### ПЕРЕОБОЗНАЧЕНИЯ ###
T = np.cos(phi)
K = np.sin(phi)
C = np.cos(theta + mu) * np.exp(-sigma**2/2)
S = np.sin(theta + mu) * np.exp(-sigma**2/2)

##### СУПЕРОПЕРАТОР #####

G = 1/2 * np.array([[1 + C, -S*(K + 1j*T), S * (-K + 1j*T), 1-C],
                    [S * (K - 1j * T), 1 + C, (T + 1j * K)**2 * (1-C), S * (-K + 1j * T)],
                    [S * (K + 1j * T), (T - 1j * K)**2 * (1 - C), 1 + C, -S * (K + 1j*T)],
                    [1 - C, S * (K + 1j * T), S * (K - 1j * T), 1 + C]], dtype=np.complex128)

##### ХИ-МАТРИЦА #####
chi = G.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
##### ОПЕРАТОРЫ КРАУСА #####
eigenvalues, eigenvectors = np.linalg.eig(chi)

idx = np.argsort(eigenvalues.real)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

kraus_operators = []
for i in range(4):
    if eigenvalues[i].real > 1e-10:
        K_i = np.sqrt(eigenvalues[i].real) * eigenvectors[:, i].reshape(2, 2)
        # Выравниваем глобальную фазу
        phase = np.exp(1j * (np.pi/2 - np.angle(K_i[0,0])))
        kraus_operators.append(K_i * phase)

print("--- ЧИСЛЕННЫЕ ОПЕРАТОРЫ ---")
for i, K_i in enumerate(kraus_operators):
    print(f"Краус (эксп) #{i+1}:\n{np.round(K_i, 5)}\n")

##### АНАЛИТИЧЕСКИЕ ОПЕРАТОРЫ КРАУСА #####
N_star = np.array([[0, np.exp(1j * phi)], [np.exp(-1j * phi), 0]], dtype=np.complex128)

E1 = 1/2 * np.sqrt((1 + np.exp(-sigma**2/2))/(1 - np.cos(theta + mu))) * (1j * np.sin(theta + mu) * np.eye(2, dtype=np.complex128) - (1 - np.cos(theta + mu)) * N_star)
E2 = 1/2 * np.sqrt((1 - np.exp(-sigma**2/2))/(1 + np.cos(theta + mu))) * (1j * np.sin(theta + mu) * np.eye(2, dtype=np.complex128) + (1 + np.cos(theta + mu)) * N_star)

print("--- АНАЛИТИЧЕСКИЕ ОПЕРАТОРЫ ---")
print(f"E1:\n{np.round(E1, 5)}\n")
print(f"E2:\n{np.round(E2, 5)}\n")

for i, K_i in enumerate(kraus_operators):
    for j, E_j in enumerate([E1, E2]):
        print(f"Проверка Крауса (эксп) #{i+1} и Крауса (теор) #{j+1}: {np.allclose(K_i, E_j, atol=1e-5)}")