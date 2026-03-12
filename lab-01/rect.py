import numpy as np
import matplotlib.pyplot as plt

n = int(input('Введите количество отсчётов (n): '))
dx = float(input('Введите шаг дискретизации dx: '))

L_values = [1, 2, 3]

# Частота Найквиста: максимальная частота, которую может иметь аналоговый сигнал, 
# чтобы его можно было правильно преобразовать в цифровой сигнал
F = 1 / (2 * dx)

# Модуль спектра Фурье
def rect_spectrum_amplitude(f, L):
    with np.errstate(divide='ignore', invalid='ignore'):
        spectrum = np.abs(2 * L * np.sinc(2 * L * f))
        spectrum[np.abs(f) < 1e-10] = 2 * L
    return spectrum


# Спектр Фурье
def rect_spectrum_complex(f, L):
    with np.errstate(divide='ignore', invalid='ignore'):
        spectrum = 2 * L * np.sinc(2 * L * f)
        spectrum[np.abs(f) < 1e-10] = 2 * L
    return spectrum


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

for idx, L in enumerate(L_values):
    x_max = dx * (n - 1) / 2
    x_samples = np.linspace(-x_max, x_max, n)
    rect_samples = np.zeros_like(x_samples)
    rect_samples[np.abs(x_samples) <= L] = 1
    x_fine = np.linspace(-x_max, x_max, 2000)
    rect_original = np.zeros_like(x_fine)
    rect_original[np.abs(x_fine) <= L] = 1
    rect_restored = kotelnikov_reconstruct(rect_samples, x_samples, x_fine, F)

    # ========================================================
    # СИГНАЛ
    # ========================================================

    ax1 = axes[idx, 0]
    ax1.plot(x_fine, rect_original, 'k-', linewidth=2, label='U(x)')
    ax1.plot(x_fine, rect_restored, 'b--', linewidth=1, label='Û(x)')
    ax1.plot(x_samples, rect_samples, 'ro', markersize=3, label='Отсч.')
    ax1.grid(True, alpha=0.2)
    ax1.set_xlabel('x', fontsize=8)
    ax1.set_ylabel('U(x)', fontsize=8)
    ax1.set_title(f'Прямоугольный сигнал (L={L})', fontsize=8)
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
    spectrum_amp = rect_spectrum_amplitude(f, L)
    spectrum_complex = rect_spectrum_complex(f, L)
    ax2.plot(f, spectrum_complex, 'b-', linewidth=2, label='V(f)')
    ax2.plot(f, spectrum_amp, 'r-', linewidth=1.5, label='|V(f)|')
    ax2.grid(True, alpha=0.2)
    ax2.set_xlabel('f', fontsize=8)
    ax2.set_ylabel('V(f), |V(f)|', fontsize=8)
    ax2.set_title(f'Спектр (L={L})', fontsize=8)
    ax2.set_xlim(-f_max, f_max)
    ax2.set_ylim(-max(2 * L + 1, 7), max(2 * L + 1, 7))
    ax2.legend()
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

print("=" * 70)
print("ПАРАМЕТРЫ".center(70))
print("=" * 70)

print(f"  Количество отсчётов n = {n}")
print(f"  Шаг дискретизации dx = {dx}")

print("\n" + "=" * 70)