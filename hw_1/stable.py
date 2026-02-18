import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def psi_0_gauss(x, sig, x_0, k0):
    if sig == 0: return np.zeros_like(x, dtype=complex)
    psi = (1/np.sqrt(2*np.pi * sig**2) * np.exp(-(x-x_0)**2/(2*sig**2)))**0.5 * np.exp(1j * k0 * x)
    psi[0] = 0.0
    psi[-1] = 0.0
    return psi

def shrodinger_euler_solver(L, H, x_b, w_b, x_0, sig, m = 1, k0 = 15, T = 0.5, N = 1000, M = 1000, init_state = 'gauss', name = 'quantum_evolution'):
    def V(x):
        '''
        Потенциал
        '''
        return 0.0 if (x < x_b or x > x_b + w_b) else H 

    ### ПРОСТРАНСТВЕННАЯ СЕТКА ###
    x = np.linspace(0, L, N+1) 
    t = np.linspace(0, T, M+1) 
    dx = x[1] - x[0] 
    dt = t[1] - t[0] # Шаг по времени

    ### КОНСТАНТА ГАММА ###
    gamma = 1j * 2 * m * dx**2 / dt
    
    # --- МАТРИЦА A (Левая часть) ---
    # Диагональ: 2*gamma - 2 - 2*m*dx^2*V
    # Побочные: 1
    main_diag_A = [1.0] + [2*gamma - 2 - 2*m*dx**2 * V(x_i) for x_i in x[1:-1]] + [1.0]
    up_diag_A = [0.0] + [1.0] * (N-1) # Единицы на побочных
    down_diag_A = [1.0] * (N-1) + [0.0] # Единицы на побочных
    
    A = diags([main_diag_A, up_diag_A, down_diag_A], [0, 1, -1], format='csc')

    # --- МАТРИЦА B (Правая часть) ---
    main_diag_B = [1.0] + [2*gamma + 2 + 2*m*dx**2 * V(x_i) for x_i in x[1:-1]] + [1.0]
    up_diag_B = [0.0] + [-1] * (N-1) # Минус единицы на побочных
    down_diag_B = [-1] * (N-1) + [0]
    
    B = diags([main_diag_B, up_diag_B, down_diag_B], [0, 1, -1], format='csc')

    ### ИНИЦИАЛИЗАЦИЯ ###
    if init_state == 'gauss':
        psi = psi_0_gauss(x, sig, x_0, k0) 
    else:
        raise ValueError('Неверное имя для начальной функции!!!')
        
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    ax1.set_xlim(0, L)

    # Плотность вероятности
    line_prob, = ax1.plot([], [], 'k-', lw=2)

    text_norm = ax1.text(0.02, 0.92, '', transform=ax1.transAxes, fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    barrier_x = [val for val in x if V(val) > 0]
    if barrier_x:
        ax1.axvspan(barrier_x[0], barrier_x[-1], color='gray', alpha=0.3, label='Potential')
        
    ax1.set_title('Плотность вероятности')
    ax1.set_ylabel('Плотность')
    
    # Динамический предел Y для красоты
    max_val = np.max(np.abs(psi)**2)
    if max_val == 0: max_val = 1.0
    ax1.set_ylim(0, max_val * 1.5)

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
        return line_prob, line_real, line_imag, text_norm

    GIF_FRAMES = 300 
    
    steps_per_frame = max(1, int(M / GIF_FRAMES))

    def update(frame):
        nonlocal psi
        
        for _ in range(steps_per_frame):
            # 1. Считаем правую часть: B * psi_old
            rhs = B.dot(psi)
            
            # 2. Решаем систему: A * psi_new = rhs
            psi = spsolve(A, rhs)
        current_norm = np.sum(np.abs(psi)**2) * dx
        text_norm.set_text(f'Norm: {current_norm:.5f}')
        
        line_prob.set_data(x, np.abs(psi)**2)
        line_real.set_data(x, np.real(psi))
        line_imag.set_data(x, np.imag(psi))
        
        return line_prob, line_real, line_imag, text_norm

    print(f"Расчет для T={T}. Всего шагов M={M}. Рисуем кадров: {GIF_FRAMES}. Ускорение: x{steps_per_frame}")

    anim = FuncAnimation(fig, update, init_func=init, frames=GIF_FRAMES, interval=30, blit=True)

    anim.save(name + '.gif', writer=PillowWriter(fps=30))
    print("Готово!")

    plt.close(fig)

if __name__ == '__main__':
    #### ПАРАМЕТРЫ ####
    L = 4.0 # Расстояния между стенками
    x_0 = 1.0 # Среднее Гаусса
    sig = 0.3 # Стандартное отклонение Гаусса (чуть поуже для наглядности)
    m = 1.0 # Масса частицы
    k0 = 10.0 # Волновой вектор (чуть побыстрее)
    H = 0 # Высота барьера
    x_b = 0 # Начало барьера
    w_b = 0 # Ширина барьера

    ### СЕТКА ###
    N = 100 # Число узлов сетки
    
    T = 0.5 # Время моделирования
    M = 10000
    shrodinger_euler_solver(L = L, H = H, x_b = x_b, w_b = w_b, x_0 = x_0, sig = sig, m = m, k0 = k0, 
                            T = T, N = N, M = M, init_state='gauss', name = 'quantum_evolution_stable')
