import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm

print("="*80)
print("VISUALIZACIÓN COMPLETA - ANÁLISIS DE WHIRLS")
print("Con restricción correcta: s = √(c² - 1)")
print("="*80)

# Cargar datos corregidos
data = np.load('whirl_data_corrected.npz')
X = data['X']
Y = data['Y']
vtil1 = data['vtil1']
vtil2 = data['vtil2']
omegatil = data['omegatil']
omega_base = data['omega_base']
circulations = data['circulations']
radii = data['radii']
c_val = float(data['c'])
s_val = float(data['s'])

print(f"\nParámetros:")
print(f"  c = {c_val}")
print(f"  s = √(c² - 1) = {s_val:.6f}")
print()

# Calcular magnitudes
velocity_magnitude = np.sqrt(vtil1**2 + vtil2**2)
grad_omegatil_x = np.gradient(omegatil, axis=1)
grad_omegatil_y = np.gradient(omegatil, axis=0)
grad_omegatil_mag = np.sqrt(grad_omegatil_x**2 + grad_omegatil_y**2)

# Buscar puntos críticos
threshold = 0.15
critical_points = []

for i in range(1, len(Y[0])-1):
    for j in range(1, len(X)-1):
        if velocity_magnitude[j, i] < threshold:
            if velocity_magnitude[j, i] < velocity_magnitude[j-1, i] and \
               velocity_magnitude[j, i] < velocity_magnitude[j+1, i] and \
               velocity_magnitude[j, i] < velocity_magnitude[j, i-1] and \
               velocity_magnitude[j, i] < velocity_magnitude[j, i+1]:
                critical_points.append((X[j, i], Y[j, i], omegatil[j, i]))

critical_points_sorted = sorted(critical_points, key=lambda p: abs(p[2]), reverse=True)

print(f"Puntos críticos encontrados: {len(critical_points_sorted)}")
if len(critical_points_sorted) > 0:
    print(f"Top 3 (por |ω̃|):")
    for idx, (xc, yc, omega_c) in enumerate(critical_points_sorted[:3]):
        print(f"  {idx+1}. ({xc:.4f}, {yc:.4f}): ω̃ = {omega_c:+.4f}")
print()

# Crear figura con múltiples subplots
fig = plt.figure(figsize=(20, 14))

# Subplot 1: Líneas de corriente con vorticidad de fondo
ax1 = fig.add_subplot(3, 3, 1)
speed = velocity_magnitude
im1 = ax1.contourf(X, Y, omegatil, levels=30, cmap='RdBu_r', alpha=0.6)
strm1 = ax1.streamplot(X, Y, vtil1, vtil2, color='black',
                        density=2, linewidth=0.8, arrowsize=1.2)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Líneas de corriente del flujo fluctuante (ṽ₁, ṽ₂)')
ax1.plot(np.pi, 0, 'g*', markersize=20, label='(π, 0)',
         markeredgecolor='white', markeredgewidth=2)
for xc, yc, _ in critical_points_sorted[:3]:
    ax1.plot(xc, yc, 'wo', markersize=10, markeredgecolor='red', markeredgewidth=2)
ax1.legend(loc='upper right')
plt.colorbar(im1, ax=ax1, label='ω̃')

# Subplot 2: Vorticidad fluctuante ω̃
ax2 = fig.add_subplot(3, 3, 2)
levels_omega = np.linspace(omegatil.min(), omegatil.max(), 31)
im2 = ax2.contourf(X, Y, omegatil, levels=levels_omega, cmap='RdBu_r')
ax2.contour(X, Y, omegatil, levels=[0], colors='black', linewidths=3)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title(f'Vorticidad fluctuante ω̃ (s = √(c²-1) = {s_val:.3f})')
ax2.plot(np.pi, 0, 'g*', markersize=20,
         markeredgecolor='white', markeredgewidth=2)
for xc, yc, _ in critical_points_sorted[:3]:
    ax2.plot(xc, yc, 'wo', markersize=10, markeredgecolor='black', markeredgewidth=2)
plt.colorbar(im2, ax=ax2)

# Subplot 3: Vorticidad base ω
ax3 = fig.add_subplot(3, 3, 3)
im3 = ax3.contourf(X, Y, omega_base, levels=30, cmap='plasma')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title(f'Vorticidad base ω = 1/D² (max={omega_base.max():.2f})')
ax3.plot(np.pi, 0, 'c*', markersize=20,
         markeredgecolor='white', markeredgewidth=2)
# Círculos de circulación
for r in [0.3, 0.6, 0.9]:
    circle = Circle((np.pi, 0), r, fill=False, edgecolor='white',
                   linewidth=2, linestyle='--', alpha=0.7)
    ax3.add_patch(circle)
plt.colorbar(im3, ax=ax3)

# Subplot 4: Magnitud de velocidad |ṽ|
ax4 = fig.add_subplot(3, 3, 4)
im4 = ax4.contourf(X, Y, velocity_magnitude, levels=30, cmap='hot')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('Magnitud de velocidad |ṽ|')
ax4.plot(np.pi, 0, 'c*', markersize=20,
         markeredgecolor='white', markeredgewidth=2)
for xc, yc, _ in critical_points_sorted[:3]:
    ax4.plot(xc, yc, 'co', markersize=10, markeredgecolor='black', markeredgewidth=2)
plt.colorbar(im4, ax=ax4)

# Subplot 5: Campo vectorial
ax5 = fig.add_subplot(3, 3, 5)
step = 6
X_sub = X[::step, ::step]
Y_sub = Y[::step, ::step]
vtil1_sub = vtil1[::step, ::step]
vtil2_sub = vtil2[::step, ::step]
speed_sub = velocity_magnitude[::step, ::step]
q = ax5.quiver(X_sub, Y_sub, vtil1_sub, vtil2_sub, speed_sub,
               cmap='viridis', scale=30, width=0.003)
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_title('Campo vectorial (ṽ₁, ṽ₂)')
ax5.plot(np.pi, 0, 'r*', markersize=20,
         markeredgecolor='white', markeredgewidth=2)
for xc, yc, _ in critical_points_sorted[:3]:
    ax5.plot(xc, yc, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
plt.colorbar(q, ax=ax5)

# Subplot 6: Circulación vs radio
ax6 = fig.add_subplot(3, 3, 6)
ax6.plot(radii, circulations, 'bo-', linewidth=3, markersize=10,
         label=f'Γ_avg = {np.mean(circulations):.3f}')
ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax6.fill_between(radii, 0, circulations, alpha=0.3, color='blue')
ax6.set_xlabel('Radio r', fontsize=12)
ax6.set_ylabel('Circulación Γ', fontsize=12)
ax6.set_title('Circulación alrededor de (π, 0)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=11)
# Anotar valores clave
max_circ_idx = np.argmax(circulations)
ax6.annotate(f'Max: {circulations[max_circ_idx]:.2f}',
             xy=(radii[max_circ_idx], circulations[max_circ_idx]),
             xytext=(radii[max_circ_idx]+0.15, circulations[max_circ_idx]+0.2),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=10, color='red', fontweight='bold')

# Subplot 7: Gradiente de vorticidad |∇ω̃|
ax7 = fig.add_subplot(3, 3, 7)
im7 = ax7.contourf(X, Y, grad_omegatil_mag, levels=30, cmap='hot')
ax7.set_xlabel('x')
ax7.set_ylabel('y')
ax7.set_title('|∇ω̃| (Gradiente de vorticidad)')
ax7.plot(np.pi, 0, 'c*', markersize=20,
         markeredgecolor='white', markeredgewidth=2)
plt.colorbar(im7, ax=ax7)

# Subplot 8: Zoom cerca del punto más crítico
ax8 = fig.add_subplot(3, 3, 8)

if len(critical_points_sorted) > 0:
    xc, yc, omega_c = critical_points_sorted[0]
else:
    xc, yc, omega_c = np.pi, 0, 0

zoom_size = 0.4
x_idx = (X[0, :] >= xc - zoom_size) & (X[0, :] <= xc + zoom_size)
y_idx = (Y[:, 0] >= yc - zoom_size) & (Y[:, 0] <= yc + zoom_size)

X_zoom = X[np.ix_(y_idx, x_idx)]
Y_zoom = Y[np.ix_(y_idx, x_idx)]
vtil1_zoom = vtil1[np.ix_(y_idx, x_idx)]
vtil2_zoom = vtil2[np.ix_(y_idx, x_idx)]
omegatil_zoom = omegatil[np.ix_(y_idx, x_idx)]

im8 = ax8.contourf(X_zoom, Y_zoom, omegatil_zoom, levels=20, cmap='RdBu_r')
strm8 = ax8.streamplot(X_zoom, Y_zoom, vtil1_zoom, vtil2_zoom,
                       color='black', density=2.5, linewidth=1.2, arrowsize=1.5)
ax8.plot(xc, yc, 'g*', markersize=25, label=f'ω̃={omega_c:.2f}',
         markeredgecolor='white', markeredgewidth=2)
ax8.plot(np.pi, 0, 'ro', markersize=12, label='(π, 0)',
         markeredgecolor='white', markeredgewidth=2)
ax8.set_xlabel('x')
ax8.set_ylabel('y')
ax8.set_title(f'Zoom: región ({xc:.2f}, {yc:.2f})')
ax8.legend()
plt.colorbar(im8, ax=ax8)

# Subplot 9: Comparación de vorticidades en cortes
ax9 = fig.add_subplot(3, 3, 9)

# Corte en y=0
y_idx = np.argmin(np.abs(Y[:, 0]))
x_line = X[y_idx, :]
omega_base_line = omega_base[y_idx, :]
omegatil_line = omegatil[y_idx, :]

ax9.plot(x_line, omega_base_line, 'b-', linewidth=3, label='ω (base)', alpha=0.7)
ax9.plot(x_line, omegatil_line, 'r-', linewidth=3, label='ω̃ (fluctuante)', alpha=0.7)
ax9.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax9.axvline(x=np.pi, color='g', linestyle='--', linewidth=2, alpha=0.5, label='x=π')
ax9.set_xlabel('x', fontsize=12)
ax9.set_ylabel('Vorticidad', fontsize=12)
ax9.set_title('Corte en y = 0', fontsize=12, fontweight='bold')
ax9.legend(fontsize=10)
ax9.grid(True, alpha=0.3)

# Anotar valor en (π, 0)
pi_idx = np.argmin(np.abs(x_line - np.pi))
ax9.plot(np.pi, omega_base_line[pi_idx], 'bo', markersize=12,
         markeredgecolor='white', markeredgewidth=2)
ax9.annotate(f'ω = {omega_base_line[pi_idx]:.2f}',
             xy=(np.pi, omega_base_line[pi_idx]),
             xytext=(np.pi-0.3, omega_base_line[pi_idx]+1),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2),
             fontsize=10, color='blue', fontweight='bold')

plt.suptitle(f'ANÁLISIS COMPLETO DE WHIRLS - Componente 2 (ν=0)\n' +
             f'Con restricción física: s = √(c² - 1) = {s_val:.4f}',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('whirl_analysis_complete.png', dpi=200, bbox_inches='tight')
print("Figura guardada: whirl_analysis_complete.png")

# Segunda figura: Análisis detallado de circulación
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Circulación vs radio (detallado)
ax = axes[0, 0]
ax.plot(radii, circulations, 'o-', linewidth=3, markersize=12, color='blue')
ax.fill_between(radii, 0, circulations, alpha=0.3, color='blue')
ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
ax.set_xlabel('Radio r', fontsize=14)
ax.set_ylabel('Circulación Γ', fontsize=14)
ax.set_title('Circulación Γ(r) alrededor de (π, 0)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.4)
ax.text(0.5, max(circulations)*0.85,
        f'Γ_promedio = {np.mean(circulations):.3f}\n' +
        f'Γ_max = {max(circulations):.3f}\n' +
        f'σ = {np.std(circulations):.3f}',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
        fontsize=12, fontweight='bold')

# Panel 2: Γ/r vs radio (densidad de circulación)
ax = axes[0, 1]
gamma_per_r = circulations / radii
ax.plot(radii, gamma_per_r, 's-', linewidth=3, markersize=10, color='red')
ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
ax.set_xlabel('Radio r', fontsize=14)
ax.set_ylabel('Γ/r (Densidad de circulación)', fontsize=14)
ax.set_title('Densidad de circulación Γ/r', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.4)
max_density_idx = np.argmax(gamma_per_r)
ax.plot(radii[max_density_idx], gamma_per_r[max_density_idx], 'g*',
        markersize=25, markeredgecolor='white', markeredgewidth=2)
ax.text(radii[max_density_idx]+0.1, gamma_per_r[max_density_idx],
        f'Max: {gamma_per_r[max_density_idx]:.2f}',
        fontsize=11, fontweight='bold', color='green')

# Panel 3: Mapa de calor de vorticidad con círculos
ax = axes[1, 0]
im = ax.contourf(X, Y, omegatil, levels=40, cmap='RdBu_r')
for i, r in enumerate(radii[::2]):
    circle = Circle((np.pi, 0), r, fill=False, edgecolor='white',
                   linewidth=2, linestyle='-', alpha=0.8)
    ax.add_patch(circle)
    if i % 2 == 0:
        ax.text(np.pi + r/np.sqrt(2), r/np.sqrt(2), f'r={r:.1f}',
               color='white', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
ax.plot(np.pi, 0, 'g*', markersize=25, markeredgecolor='white', markeredgewidth=3)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title('Radios de integración para Γ', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='ω̃')

# Panel 4: Comparación ω vs ω̃ en (π, 0)
ax = axes[1, 1]
properties = ['ω(π,0)', 'ω(0,0)', 'Γ_avg', 'max|ṽ|']
values = [omega_base[np.argmin(np.abs(Y[:, 0])), np.argmin(np.abs(X[0, :] - np.pi))],
          omega_base[np.argmin(np.abs(Y[:, 0])), np.argmin(np.abs(X[0, :]))],
          np.mean(circulations),
          np.max(velocity_magnitude)]
colors = ['red', 'blue', 'green', 'purple']

bars = ax.bar(properties, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Magnitud', fontsize=14)
ax.set_title('Propiedades clave del sistema', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Anotar valores en las barras
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Anotar el ratio
ax.text(0.5, max(values)*0.7,
        f'ω(π,0) / ω(0,0) = {values[0]/values[1]:.1f}×',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
        fontsize=13, fontweight='bold', ha='center',
        transform=ax.transAxes)

plt.suptitle(f'ANÁLISIS DETALLADO DE CIRCULACIÓN\n' +
             f'Parámetros: c = {c_val}, s = √(c²-1) = {s_val:.4f}',
             fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('circulation_analysis_detailed.png', dpi=200, bbox_inches='tight')
print("Figura guardada: circulation_analysis_detailed.png")

print("\n" + "="*80)
print("RESUMEN DE VISUALIZACIONES")
print("="*80)
print("\n✅ Archivos generados:")
print("  1. whirl_analysis_complete.png - Análisis completo (9 paneles)")
print("  2. circulation_analysis_detailed.png - Análisis de circulación (4 paneles)")
print()
print(f"📊 Estadísticas clave:")
print(f"  • Vorticidad base en (π,0): ω = {omega_base[y_idx, pi_idx]:.2f}")
print(f"  • Circulación promedio: Γ = {np.mean(circulations):.3f}")
print(f"  • Circulación máxima: Γ_max = {max(circulations):.3f}")
print(f"  • Puntos críticos encontrados: {len(critical_points_sorted)}")
print()
print("="*80)

plt.show()
