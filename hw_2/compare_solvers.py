import numpy as np
from scipy.special import wofz
from scipy.linalg import solve_banded
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def get_amplitudes(x0, sig, k0, L, N_max):
    n = np.arange(1, N_max + 1)
    
    K_plus = k0 + np.pi * n / L
    K_minus = k0 - np.pi * n / L 
    
    def E_stable(K):
        iz1 = sig * K + 1j * (L - x0) / (2 * sig)
        iz2 = -sig * K + 1j * x0 / (2 * sig)
        
        term1 = 2 * np.exp(-sig**2 * K**2 + 1j * K * x0)
        term2 = np.exp(-(L - x0)**2 / (4 * sig**2) + 1j * K * L) * wofz(iz1)
        term3 = np.exp(-x0**2 / (4 * sig**2)) * wofz(iz2)
        
        return term1 - term2 - term3
    
    E_p = E_stable(K_plus)
    E_m = E_stable(K_minus)
    
    return -1j * np.sqrt(sig / (2 * L)) * (np.pi / 2)**0.25 * (E_p - E_m)

def phi_n(n, L, x):
    '''
    Базисная функция n бесконечного потенциала ширины L
    '''
    psi = np.sqrt(2/L) * np.sin(n * np.pi * x/L)
    return psi

def get_wavefunction(amplitudes, L, x):
    '''
    Получение ВФ из разложения по базису Гамильтониана для бесконечного потенциала
    '''
    n = np.arange(1, amplitudes.shape[0] + 1)
    psi_n_s = np.sqrt(2/L) * np.sin(np.outer(n, x) * (np.pi / L))
    psi = amplitudes @ psi_n_s
    return psi

def compare_solvers(L, H, x_b, w_b, x_0, sig, m=1.0, k0=15, T=0.5, N=1000, M=1000, 
                    name='quantum_comparison', frames=300):

    def V(x):
        return np.where((x >= x_b) & (x <= x_b + w_b), H, 0.0)

    ### ОБЩАЯ СЕТКА ###
    x = np.linspace(0, L, N+1) 
    t_frames = np.linspace(0, T, frames)
    dx = x[1] - x[0] 
    dt = T / M
    steps_per_frame = max(1, int(M / frames))

    ### ЧИСЛО БАЗИСНЫХ ФУНКЦИЙ ДЛЯ РЯДОВ ###
    N_max = np.ceil(L/np.pi * (k0 + 10/sig)).astype(int) # Число базисных функций, достаточное для точного разложения начального состояния (с запасом)
    print(f"Количество базисных функций (N_max): {N_max}")

    # Гамильтониан
    H_matrix = np.zeros((N_max, N_max), dtype=complex)
    for i in range(N_max):
        for j in range(N_max):
            if i == j:
                H_matrix[i,i] = (np.pi * (i+1)/L)**2/(2*m) + H * (w_b/L - 1/(2*np.pi * (i+1)) * (np.sin(np.pi * (x_b + w_b) * 2 * (i+1)/L) - np.sin(2 * np.pi * x_b * (i+1)/L)))
            else:
                H_matrix[i,j] = H/np.pi * ((np.sin(np.pi * (x_b + w_b) * (i-j)/L) - np.sin(np.pi * x_b * (i - j)/L))/(i - j) - (np.sin(np.pi * (x_b + w_b) * (i + j + 2)/L) - np.sin(np.pi * x_b * (i + j + 2)/L))/(i + j + 2))

    D, U = np.linalg.eigh(H_matrix)
    U_dag = U.conj().T

    ### НАЧАЛЬНЫЕ АМПЛИТУДЫ ###
    init_amplitudes = get_amplitudes(x_0, sig, k0, L, N_max)
    
    ### ГАМИЛЬТОНИАН ###
    n = np.arange(1, N_max + 1)
    
    # Создаем 2D сетки (столбец и строку)
    n_i = n[:, np.newaxis]
    n_j = n[np.newaxis, :]

    diff = n_i - n_j
    summ = n_i + n_j

    # Защита от деления на ноль на главной диагонали
    safe_diff = np.where(diff == 0, 1, diff)

    # Считаем внедиагональные элементы для всей матрицы разом
    term1 = (np.sin(np.pi * (x_b + w_b) * safe_diff / L) - np.sin(np.pi * x_b * safe_diff / L)) / safe_diff
    term2 = (np.sin(np.pi * (x_b + w_b) * summ / L) - np.sin(np.pi * x_b * summ / L)) / summ
    
    H_matrix = (H / np.pi) * (term1 - term2)

    diag_term = (np.pi * n / L)**2 / (2 * m) + H * (
        w_b / L - 1 / (2 * np.pi * n) * (
            np.sin(np.pi * (x_b + w_b) * 2 * n / L) - np.sin(2 * np.pi * x_b * n / L)
        )
    )
    np.fill_diagonal(H_matrix, diag_term)
    H_matrix = H_matrix.astype(complex)

    print(' H is Herm?', np.allclose(H_matrix, H_matrix.conj().T))

    ### SVD ###
    D, U = np.linalg.eigh(H_matrix)
    U_dag = U.conj().T

    ### ПРЕДВЫЧИСЛЕНИЕ АМПЛИТУД ###
    t_frames = np.linspace(0, T, frames)
    start = time.time()
    # Переход в базис собственных состояний H
    C_0 = U_dag @ init_amplitudes 
    
    # Broadcasting для быстрого вычисления
    phases = np.exp(-1j * D[:, np.newaxis] * t_frames[np.newaxis, :])
    
    # Умножаем начальные амплитуды на фазы для всех кадров сразу
    C_t = C_0[:, np.newaxis] * phases 
    
    # Возвращаемся в базис исходных функций
    C_fin = U @ C_t

    time_of_calc = time.time() - start
    print(f"Предвычисление амплитуд для всех кадров {frames} заняло {time_of_calc:.8f} секунд.")
    
    ### МЕТОД ЭЙЛЕРА ###

    ### КОНСТАНТА ГАММА ###
    gamma = 1j * 2 * m * dx**2/dt

    ### МАТРИЦА А ###
    main_diag = [1.0] + [1.0 - 2/gamma * (1.0 + m * dx**2 * V(x_i)) for x_i in x[1:-1]] + [1.0]
    up_diag = [0.0] + [0.0] + [1/gamma] * (N-1)
    low_diag = [1/gamma] * (N-1) + [0.0] + [0.0]

    A = np.array([up_diag, main_diag, low_diag], dtype=complex)
    print("Матрица A успешно создана. Размер:", A.shape)
    
    ### НАЧАЛЬНЫЙ ВЕКТОР ###
    psi_euler = get_wavefunction(init_amplitudes, L, x)

    ### РИСОВАНИЕ ###
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    # Настройка осей и графиков
    V_arr = V(x)
    ax1.set_xlim(0, L)
    ax1.fill_between(x, 0, 1, where=(V_arr > 0), color='gray', alpha=0.3, transform=ax1.get_xaxis_transform(), label='Barrier')
    
    # Линии (сплошная - ряд, пунктир - Эйлер)
    line_prob_s, = ax1.plot([], [], 'k-', lw=2, label='Базис (ряд)')
    line_prob_e, = ax1.plot([], [], 'r--', lw=2, label='Эйлер')
    
    line_real_s, = ax2.plot([], [], 'b-', lw=1.5, label='Re Базис')
    line_real_e, = ax2.plot([], [], 'r--', lw=1.5, label='Re Эйлер')
    
    line_imag_s, = ax3.plot([], [], 'g-', lw=1.5, label='Im Базис')
    line_imag_e, = ax3.plot([], [], 'r--', lw=1.5, label='Im Эйлер')

    # Текст для норм
    text_norm = ax1.text(0.02, 0.85, '', transform=ax1.transAxes, fontsize=11, 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Оформление
    ax1.set_title('Плотность вероятности $|\psi|^2$')
    ax1.set_ylim(0, np.max(np.abs(psi_euler)**2) * 1.5)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2.set_title('Действительная часть')
    ax2.set_ylim(-1, 1)
    ax2.legend(loc='upper right')
    ax2.grid(True)

    ax3.set_title('Мнимая часть')
    ax3.set_ylim(-1, 1)
    ax3.legend(loc='upper right')
    ax3.grid(True)

    def init():
        for line in [line_prob_s, line_prob_e, line_real_s, line_real_e, line_imag_s, line_imag_e]:
            line.set_data([], [])
        return line_prob_s, line_prob_e, line_real_s, line_real_e, line_imag_s, line_imag_e

    def update(frame_idx):
        nonlocal psi_euler

        # 1. Шаг метода рядов (просто берем предвычисленный кадр)
        current_ampl = C_fin[:, frame_idx]
        psi_series = get_wavefunction(current_ampl, L, x)
        
        # 2. Шаг метода Эйлера (прогоняем через solve_banded нужное число dt)
        if frame_idx > 0: # Пропускаем нулевой кадр (он уже задан)
            for _ in range(steps_per_frame):
                psi_euler = solve_banded((1, 1), A, psi_euler)
        
        # Нормы
        norm_s = np.sum(np.abs(psi_series)**2) * dx
        norm_e = np.sum(np.abs(psi_euler)**2) * dx
        text_norm.set_text(f'Норма (базис): {norm_s:.5f}\nНорма Эйлер: {norm_e:.5f}')
        
        # Обновление графиков
        line_prob_s.set_data(x, np.abs(psi_series)**2)
        line_prob_e.set_data(x, np.abs(psi_euler)**2)
        
        line_real_s.set_data(x, np.real(psi_series))
        line_real_e.set_data(x, np.real(psi_euler))
        
        line_imag_s.set_data(x, np.imag(psi_series))
        line_imag_e.set_data(x, np.imag(psi_euler))
        
        return line_prob_s, line_prob_e, line_real_s, line_real_e, line_imag_s, line_imag_e

    print(f"Запуск симуляции. Кадров: {frames}. Эйлер шагов на кадр: {steps_per_frame}")
    anim = FuncAnimation(fig, update, init_func=init, frames=frames, interval=30, blit=True)
    anim.save(name + '.gif', writer=PillowWriter(fps=30))
    print("Сохранено как", name + '.gif')
    plt.close(fig)

if __name__ == '__main__':
    # Физические параметры
    L = 6.0 
    x_0 = 1.0 
    sig = 0.3 
    m = 1.0 
    k0 = 20 
    H = 100
    x_b = 3
    w_b = 1 

    # Численные параметры
    N = 1000 
    T = 1
    
    # Формула для устойчивости сетки (метод Эйлера)
    M = int(100 * T * (k0**2/2 + 2*N**2/L**2)/m) 
    
    compare_solvers(L=L, H=H, x_b=x_b, w_b=w_b, x_0=x_0, sig=sig, m=m, k0=k0, 
                    T=T, N=N, M=M, name='euler_vs_series', frames=900)