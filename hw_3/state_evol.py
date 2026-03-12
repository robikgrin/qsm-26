import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
from matplotlib.colors import to_rgba

# ---  ПАРАМЕТРЫ КАНАЛА ---
MU = 0.3
PHI = np.pi/2 
THETA_TOTAL = 2 * np.pi

N_FRAMES = 180
all_thetas = np.linspace(0, THETA_TOTAL, N_FRAMES)

print("Вычисляем траектории (идеальную и зашумленную)...")
psi_0 = qt.basis(2, 0)
rho_0 = psi_0 * psi_0.dag()

pts_ideal_x, pts_ideal_y, pts_ideal_z = [], [], []
pts_noisy_x, pts_noisy_y, pts_noisy_z = [], [], []
fidelities = []

# Оператор идеальной оси вращения n_phi (без затухания)
n_vector_np = np.array([np.cos(PHI), np.sin(PHI), 0])
n_vector_qt = qt.Qobj(n_vector_np)
rotation_axis_op = n_vector_np[0]*qt.sigmax() + n_vector_np[1]*qt.sigmay() + n_vector_np[2]*qt.sigmaz()

def get_kraus_operators(current_theta):
    """Считаем аналитические операторы Крауса."""
    # Эффективная ось N* (смещенная на фазу)
    N_star_np = np.array([[0, np.exp(1j * PHI)], [np.exp(-1j * PHI), 0]], dtype=np.complex128)
    
    # Добавляем микроскопический сдвиг, чтобы избежать деления на ноль в начале
    angle = current_theta + MU + 1e-12 
    current_sigma_sq = (current_theta / THETA_TOTAL)**2
    gamma = np.exp(-current_sigma_sq / 2)
    
    E1_np = 1/2 * np.sqrt((1 + gamma)/(1 - np.cos(angle))) * \
            (1j * np.sin(angle) * np.eye(2, dtype=np.complex128) - (1 - np.cos(angle)) * N_star_np)
    E2_np = 1/2 * np.sqrt((1 - gamma)/(1 + np.cos(angle))) * \
            (1j * np.sin(angle) * np.eye(2, dtype=np.complex128) + (1 + np.cos(angle)) * N_star_np)
            
    return [qt.Qobj(E1_np), qt.Qobj(E2_np)]

# Вычисляем состояние для каждого кадра
for th in all_thetas:
    K = get_kraus_operators(th)
    rho_noisy = K[0] * rho_0 * K[0].dag() + K[1] * rho_0 * K[1].dag()
    
    U_ideal_qt = np.cos(th/2) * qt.qeye(2) - 1j * np.sin(th/2) * rotation_axis_op
    rho_ideal = U_ideal_qt * rho_0 * U_ideal_qt.dag()
    
    # Извлекаем координаты зашумленного состояния
    pts_noisy_x.append(qt.expect(qt.sigmax(), rho_noisy))
    pts_noisy_y.append(qt.expect(qt.sigmay(), rho_noisy))
    pts_noisy_z.append(qt.expect(qt.sigmaz(), rho_noisy))
    
    # Извлекаем координаты идеального состояния
    pts_ideal_x.append(qt.expect(qt.sigmax(), rho_ideal))
    pts_ideal_y.append(qt.expect(qt.sigmay(), rho_ideal))
    pts_ideal_z.append(qt.expect(qt.sigmaz(), rho_ideal))
    
    # Вычисляем Fidelity
    fid = qt.fidelity(rho_ideal, rho_noisy)
    fidelities.append(fid)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
bloch = qt.Bloch(fig=fig, axes=ax)

bloch.point_color = ['b', 'r'] 
bloch.point_marker = ['o', '']
bloch.point_size = [2, 0]

bloch.vector_color = ['r', 'g', 'b']
bloch.vector_width = 3

v_init = [0, 0, 1] 
v_axis_n = [np.cos(PHI), -np.sin(PHI), 0]

def update(frame):
    """Функция отрисовки каждого кадра."""
    bloch.clear()
    
    while ax.texts:
        ax.texts[-1].remove()

    bloch.make_sphere()

    camera_azim = np.degrees(-PHI) - 90 
    ax.view_init(elev=20, azim=camera_azim)

    bloch.add_vectors(v_init)
    bloch.add_vectors(v_axis_n)
    
    v_noisy_curr = [pts_noisy_x[frame], pts_noisy_y[frame], pts_noisy_z[frame]]
    bloch.add_vectors(v_noisy_curr)
    
    bloch.add_points([pts_noisy_x[:frame+1], pts_noisy_y[:frame+1], pts_noisy_z[:frame+1]], meth='l')
    bloch.add_points([pts_ideal_x[:frame+1], pts_ideal_y[:frame+1], pts_ideal_z[:frame+1]], meth='l')

    bloch.point_color = ['b', 'r']
    bloch.point_size = [3, 0] 
    bloch.vector_color = ['r', 'g', 'b']
    
    bloch.make_sphere() 

    legend_elements = [
        mlines.Line2D([0], [0], color='r', lw=3, label=r'Начальное состояние $\rho_0$'),
        mlines.Line2D([0], [0], color='g', lw=3, label=r'Ось вращения $n_{\phi}$'),
        mlines.Line2D([0], [0], color='b', lw=3, label=r'Зашумленное $\tilde{\rho}$')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11, 
              bbox_to_anchor=(0.0, 0.0), facecolor='white', framealpha=0.8)

    params_init_text = (
        r'$\bf{Initial\ Parameters:}$' + '\n'
        rf'  $\mu = {MU:.2f}$' + '\n'
        rf'  $\phi = {PHI:.2f}$'
    )
    
    ax.text2D(0.02, 0.98, params_init_text, transform=ax.transAxes, 
              fontsize=11, verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.8})
    
    params_curr_text = (
        r'$\bf{Current:}$' + '\n'
        rf'  $\theta = {all_thetas[frame]:.2f}$' + '\n'
        rf'  $\sigma = {all_thetas[frame]/THETA_TOTAL:.2f}$' + '\n'
        rf'  $Fidelity = {fidelities[frame]:.4f}$'
    )
    ax.text2D(0.98, 0.98, params_curr_text, transform=ax.transAxes, 
              fontsize=11, verticalalignment='top', horizontalalignment='right', 
              bbox={'facecolor': 'white', 'alpha': 0.8})

    return ax,

print("Собираем анимацию из кадров...")
ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, blit=False, repeat=True)

filename = "decoherence_rotation.gif"
print(f"Сохраняем {filename}... (это займет около 30 секунд)")
ani.save(filename, writer='pillow', fps=30)

print(f"Ура! Анимация сохранена в файл {filename}")