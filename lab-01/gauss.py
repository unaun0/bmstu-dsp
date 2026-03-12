import numpy as np
import matplotlib.pyplot as plt

A = 1.0
sigma_values = [1.0, 1.5, 2.0]

n = int(input('Введите количество отсчётов (n): '))
dx = float(input('Введите шаг дискретизации dx: '))

# Частота Найквиста: максимальная частота, которую может иметь аналоговый сигнал, 
# чтобы его можно было правильно преобразовать в цифровой сигнал
F = 1 / (2 * dx)

# Спектр Фурье
def gauss_spectrum_amplitude(f, sigma):
    """
    Амплитудный спектр гауссова сигнала U(x) = exp(-x^2/sigma^2)
    Спектр: V(f) = sigma * sqrt(π) * exp(-(πσf)^2), |V(f)| = V(f)
    """
    return sigma * np.sqrt(np.pi) * np.exp(-(np.pi * sigma * f) ** 2)

# Восстановление по теореме Котельникова
def kotelnikov_reconstruct(samples, sample_points, x_fine, F):
    result = np.zeros_like(x_fine)
    for i, x in enumerate(x_fine):
        for k, sample_value in enumerate(samples):
            arg = 2 * np.pi * F * (x - sample_points[k])
            if np.abs(arg) < 1e-10:
                result[i] += sample_value
            else:
                result[i] += sample_value * np.sin(arg) / arg
    return result


# ============================================================
# ГРАФИКИ
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
fig.subplots_adjust(
    left=0.08,
    right=0.97,
    bottom=0.06,
    top=0.93,
    hspace=0.4,
    wspace=0.25
)

for idx, sigma in enumerate(sigma_values):
    x_max = dx * (n - 1) / 2
    x_samples = np.linspace(-x_max, x_max, n)
    gauss_samples = A * np.exp(-(x_samples / sigma) ** 2)
    x_fine = np.linspace(-x_max, x_max, 2000)
    gauss_original = A * np.exp(-(x_fine / sigma) ** 2)
    gauss_restored = kotelnikov_reconstruct(gauss_samples, x_samples, x_fine, F)
    
    # ========================================================
    # СИГНАЛ
    # ========================================================
    
    ax1 = axes[idx, 0]
    ax1.plot(x_fine, gauss_original, 'k-', linewidth=2, label='U(x)')
    ax1.plot(x_fine, gauss_restored, 'b--', linewidth=1, label='Û(x)')
    ax1.plot(x_samples, gauss_samples, 'ro', markersize=3, label='Отсч.')
    ax1.grid(True, alpha=0.2)
    ax1.set_xlabel('x', fontsize=8)
    ax1.set_ylabel('U(x)', fontsize=8)
    ax1.set_title(f'Гауссов сигнал (σ={sigma})', fontsize=8)
    ax1.legend(loc='upper right')
    ax1.set_xlim(-x_max, x_max)
    ax1.set_ylim(-0.2, 1.2)
    
    # ========================================================
    # СПЕКТР
    # ========================================================
    
    ax2 = axes[idx, 1]
    k = 1.5
    f_max = k * F
    f = np.linspace(-f_max, f_max, 4000)
    spectrum = gauss_spectrum_amplitude(f, sigma)
    ax2.plot(f, spectrum, 'r-', linewidth=2, label='V(f)')
    ax2.grid(True, alpha=0.2)
    ax2.set_xlabel('f', fontsize=8)
    ax2.set_ylabel('V(f)', fontsize=8)
    ax2.set_title(f'Спектр (σ={sigma})', fontsize=8)
    ax2.set_xlim(-f_max, f_max)
    spectrum_max = gauss_spectrum_amplitude(0, sigma)
    ax2.set_ylim(0, spectrum_max * 1.1)
    ax2.axvline(F, color='green', linewidth=2)
    ax2.axvline(-F, color='green', linewidth=2)
    
    ax2.text(
        0.02,
        0.95,
        f'F = {F:.2f}',
        transform=ax2.transAxes,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2),
        fontsize=6,
        verticalalignment='top'
    )

plt.show()

# ============================================================
# ВЫВОД ПАРАМЕТРОВ
# ============================================================

print("=" * 70)
print("ПАРАМЕТРЫ".center(70))
print("=" * 70)

print(f"  Количество отсчётов n = {n}")
print(f"  Шаг дискретизации dx = {dx}")

print("\n" + "=" * 70)
