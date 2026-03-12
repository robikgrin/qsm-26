import numpy as np

def run_randomized_tests(n_tests=1000):
    passed_tests = 0
    
    for _ in range(n_tests):
        mu = np.random.uniform(0, 1.0)
        sigma = np.random.uniform(0.01, 1.0)
        theta = np.random.uniform(0, np.pi/2)
        phi = np.random.uniform(0, 2*np.pi)
        
        T = np.cos(phi)
        K = np.sin(phi)
        C = np.cos(theta + mu) * np.exp(-sigma**2/2)
        S = np.sin(theta + mu) * np.exp(-sigma**2/2)
        
        G = 1/2 * np.array([
            [1 + C, -S*(K + 1j*T), S*(-K + 1j*T), 1-C],
            [S*(K - 1j*T), 1 + C, (T + 1j*K)**2 * (1-C), S*(-K + 1j*T)],
            [S*(K + 1j*T), (T - 1j*K)**2 * (1 - C), 1 + C, -S*(K + 1j*T)],
            [1 - C, S*(K + 1j*T), S*(K - 1j*T), 1 + C]
        ], dtype=np.complex128)
        
        chi = G.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
        
        eigenvalues, eigenvectors = np.linalg.eig(chi)
        idx = np.argsort(eigenvalues.real)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        kraus_operators = []
        for i in range(4):
            if eigenvalues[i].real > 1e-10:
                K_i = np.sqrt(eigenvalues[i].real) * eigenvectors[:, i].reshape(2, 2)
                phase = np.exp(1j * (np.pi/2 - np.angle(K_i[0,0])))
                kraus_operators.append(K_i * phase)
                
        N_star = np.array([[0, np.exp(1j * phi)], [np.exp(-1j * phi), 0]], dtype=np.complex128)
        
        E1 = 1/2 * np.sqrt((1 + np.exp(-sigma**2/2))/(1 - np.cos(theta + mu))) * \
             (1j * np.sin(theta + mu) * np.eye(2, dtype=np.complex128) - (1 - np.cos(theta + mu)) * N_star)
        E2 = 1/2 * np.sqrt((1 - np.exp(-sigma**2/2))/(1 + np.cos(theta + mu))) * \
             (1j * np.sin(theta + mu) * np.eye(2, dtype=np.complex128) + (1 + np.cos(theta + mu)) * N_star)
             
        analytical_kraus = [E1, E2]
        
        match_K1 = np.allclose(kraus_operators[0], analytical_kraus[0], atol=1e-5) or \
                   np.allclose(kraus_operators[0], analytical_kraus[1], atol=1e-5)
        match_K2 = np.allclose(kraus_operators[1], analytical_kraus[0], atol=1e-5) or \
                   np.allclose(kraus_operators[1], analytical_kraus[1], atol=1e-5)
                   
        if match_K1 and match_K2:
            passed_tests += 1

    print(f"Пройдено тестов: {passed_tests} из {n_tests}")

run_randomized_tests(1000)