import numpy as np
from scipy.special import erf
from scipy.special import wofz
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def get_exact_amplitudes(x0, sig, k0, L, N_max):
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
    psi = np.zeros_like(x, dtype=complex)
    for n in range(amplitudes.shape[0]):
        psi += amplitudes[n] * phi_n(n+1, L, x)
    return psi

def series_solver(L, H, x_b, w_b, x_0, sig, m = 1,
                            k0 = 15, T = 0.5, N = 1000, name = 'quantum_evolution', frames = 300):
    def V(x):
        '''
        Потенциал
        '''
        return 0.0 if (x < x_b or x > x_b + w_b) else H 
    
    ##№ СЕТКА ###
    x = np.linspace(0, L, N+1) 
    dx = x[1] - x[0] # Шаг по координате

    N_max = np.ceil(L/np.pi * (k0 + 10/sig)).astype(int) # Число базисных функций, достаточное для точного разложения начального состояния (с запасом)
    print(f"Количество базисных функций (N_max): {N_max}")

    ### НАЧАЛЬНЫЕ АМПЛИТУДЫ ###
    init_amplitudes = get_exact_amplitudes(x_0, sig, k0, L, N_max)
    
    ### ГАМИЛЬТОНИАН ###
    H_matrix = np.zeros((N_max, N_max), dtype=complex)
    for i,j in np.ndindex(N_max, N_max):
        if i == j:
            H_matrix[i,i] = (np.pi * (i+1)/L)**2/(2*m) + H * (w_b/L - 1/(2*np.pi * (i+1)) * (np.sin(np.pi * (x_b + w_b) * 2 * (i+1)/L) - np.sin(2 * np.pi * x_b * (i+1)/L)))
        else:
            H_matrix[i,j] = H/np.pi * ((np.sin(np.pi * (x_b + w_b) * (i-j)/L) - np.sin(np.pi * x_b * (i - j)/L))/(i - j) - (np.sin(np.pi * (x_b + w_b) * (i + j + 2)/L) - np.sin(np.pi * x_b * (i + j + 2)/L))/(i + j + 2))

    print('Is Herm?', np.allclose(H_matrix, H_matrix.conj().T))

    ### SVD ###
    D, U = np.linalg.eigh(H_matrix)
    U_dag = U.conj().T

    ### НАЧАЛЬНЫЙ ВЕКТОР ###
    psi = get_wavefunction(init_amplitudes, L, x)

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
    ### РИСОВАНИЕ ###
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    ax1.set_xlim(0, L)

    # Плотность вероятности
    line_prob, = ax1.plot([], [], 'k-', lw=2)

    text_norm = ax1.text(0.02, 0.92, '', transform=ax1.transAxes, fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    barrier_x = np.array([V(val) for val in x ])
    ax1.fill_between(x, 0, 1, where=(barrier_x > 0), 
                     color='gray', alpha=0.3, transform=ax1.get_xaxis_transform(), label='Potential')
        
    ax1.set_title('Плотность вероятности')
    ax1.set_ylabel('Плотность')
    ax1.set_ylim(0, np.max(np.abs(psi)**2) * 1.5)

    ax1.grid(True)

    # Действительная часть
    line_real, = ax2.plot([], [], 'b-', lw=1, label=r'Re($\psi$)')
    ax2.set_title('Действительная часть')
    ax2.set_ylabel('Re')
    ax2.set_ylim(-1, 1)
    ax2.grid(True)

    # Мнимая часть
    line_imag, = ax3.plot([], [], 'r-', lw=1, label=r'Im($\psi$)')
    ax3.set_title('Мнимая часть')
    ax3.set_ylabel('Im')
    ax3.set_xlabel('x')
    ax3.set_ylim(-1, 1)
    ax3.grid(True)

    def init():
        line_prob.set_data([], [])
        line_real.set_data([], [])
        line_imag.set_data([], [])
        return line_prob, line_real, line_imag

    GIF_FRAMES = frames
    
    def update(frame_idx):
        # Берем предвычисленные амплитуды для текущего кадра
        current_ampl = C_fin[:, frame_idx]
        psi = get_wavefunction(current_ampl, L, x)
        
        current_norm = np.sum(np.abs(psi)**2) * dx
        text_norm.set_text(f'Norm: {current_norm:.5f}')
        
        line_prob.set_data(x, np.abs(psi)**2)
        line_real.set_data(x, np.real(psi))
        line_imag.set_data(x, np.imag(psi))
        
        return line_prob, line_real, line_imag
    
    anim = FuncAnimation(fig, update, init_func=init, frames=GIF_FRAMES, interval=30, blit=True)

    anim.save(name + '.gif', writer=PillowWriter(fps=30))
    print("Готово!")

    plt.close(fig)



if __name__ == '__main__':
   #### ПАРАМЕТРЫ ####
    L = 6.0 # Расстояния между стенками
    x_0 = 1.0 # Среднее Гаусса
    sig = 0.3 # Стандартное отклонение Гаусса (чуть поуже для наглядности)
    m = 1.0 # Масса частицы
    k0 = 20 # Волновой вектор (чуть побыстрее)
    H = 100 # Высота барьера
    x_b = 3 # Начало барьера
    w_b = 0.5 # Ширина барьера

    ### СЕТКА ###
    N = 1000 # Число узлов сетки
    T = 1 # Время моделирования

    series_solver(L, H, x_b, w_b, x_0, sig, m, k0, T, N, frames=300)
    