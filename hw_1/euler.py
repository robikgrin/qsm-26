import numpy as np
from scipy.sparse import diags
from scipy.stats import norm
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def psi_0_sin(n, x, L):
    '''
    Начальная ВФ в виде гауссова пакета
    '''
    psi = np.sqrt(2/L) * np.sin(n * np.pi * x/L)
    psi[0] = 0.0
    psi[-1] = 0.0
    return psi

def psi_0_gauss(x, sig, x_0, k0):
        '''
        Начальная ВФ в виде гауссова пакета
        '''
        psi = (1/np.sqrt(2*np.pi * sig**2) * np.exp(-(x-x_0)**2/(2*sig**2)))**0.5 * np.exp(1j * k0 * x)
        psi[0] = 0.0
        psi[-1] = 0.0
        return psi

def shrodinger_euler_solver(L, H, x_b, w_b, x_0, sig, m = 1, k0 = 15, T = 0.5, N = 1000, M = 1000, init_state = 'gauss',name = 'quantum_evolution'):
    def V(x):
        '''
        Потенциал
        '''
        return 0.0 if (x < x_b or x > x_b + w_b) else H 

    # def V(x):
    #     gap = 1.0 # Ширина промежутка между стенками
        
    #     # Проверяем, попадает ли x в первый ИЛИ во второй барьер
    #     if (x_b<=  x <= x_b + w_b) or (x_b + w_b + gap <= x <= x_b + 2*w_b + gap):
    #         return H
    #     return 0.0
    ### ПРОСТРАНСТВЕННАЯ СЕТКА ###
    x = np.linspace(0, L, N+1) 
    t = np.linspace(0, T, M+1) 
    dx = x[1] - x[0] 
    dt = t[1] - t[0] # Шаг по времени

    ### КОНСТАНТА ГАММА ###
    gamma = 1j * 2 * m * dx**2/dt

    ### МАТРИЦА А (ТВОЙ КОД) ###
    main_diag = [1.0] + [1.0 - 2/gamma * (1.0 + m * dx**2 * V(x_i)) for x_i in x[1:-1]] + [1.0]
    up_diag = [0.0] + [1/gamma] * (N-1)
    low_diag = [1/gamma] * (N-1) + [0.0]

    A = diags([main_diag, up_diag, low_diag], [0, 1, -1], format='csc', dtype=np.complex128)

    ### ИНИЦИАЛИЗАЦИЯ ###
    if init_state == 'gauss':
        psi = psi_0_gauss(x, sig, x_0, k0) # Текущий вектор состояния
    elif init_state == 'sin':
        n = 2
        psi = psi_0_sin(n, x, L)
    else:
        raise ValueError('Неверное имя для начальной функции!!!')
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

    GIF_FRAMES = 300 
    
    steps_per_frame = max(1, int(M / GIF_FRAMES))
    
    def update(frame):
        nonlocal psi

        for _ in range(steps_per_frame):
            psi = spsolve(A, psi)
        
        current_norm = np.sum(np.abs(psi)**2) * dx
        text_norm.set_text(f'Norm: {current_norm:.5f}')
        
        line_prob.set_data(x, np.abs(psi)**2)
        line_real.set_data(x, np.real(psi))
        line_imag.set_data(x, np.imag(psi))
        
        return line_prob, line_real, line_imag

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
    k0 = 5.0 # Волновой вектор (чуть побыстрее)
    H = 10 # Высота барьера
    x_b = 1.5 # Начало барьера
    w_b = 0.5 # Ширина барьера

    ### СЕТКА ###
    N = 100 # Число узлов сетки
    
    T = 1 # Время моделирования
    M = 100 * int(T * (k0**2/2 + 2*N**2/L**2)/m) # Число кадров анимации
    shrodinger_euler_solver(L = L, H = H, x_b = x_b, w_b = w_b, x_0 = x_0, sig = sig, m = m, k0 = k0, 
                            T = T, N = N, M = M, init_state='gauss', name = 'quantum_evolution_two_barriers_2')
