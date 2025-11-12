"""
VISUALIZACIÓN DE ZONAS CON WHIRLS INTENSOS
==========================================

Genera mapas detallados mostrando la localización espacial
de whirls intensos en el régimen extremo c ∈ [1.2, 1.3]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LogNorm

print("Cargando datos de análisis de whirls intensos...")
data = np.load('intense_whirls_data.npz', allow_pickle=True)
c_values = data['c_values']
s_values = data['s_values']
results = data['results']

print(f"✓ Datos cargados: {len(results)} casos analizados")

# ============================================================================
# FIGURA 1: MAPA DE INTENSIDAD COMPARATIVO (3 CASOS)
# ============================================================================

print("\nGenerando Figura 1: Mapa de intensidad comparativo...")

fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

for idx, case in enumerate(results):
    c_val = case['c']
    s_val = case['s']
    X = case['X']
    Y = case['Y']
    omega = case['omega_total']
    omegatil_x = case['omegatil_x']
    omegatil_y = case['omegatil_y']

    # Panel izquierdo: Campo de vorticidad total
    ax1 = fig.add_subplot(gs[idx, 0])
    im1 = ax1.contourf(X, Y, omega, levels=50, cmap='hot')
    ax1.contour(X, Y, omega, levels=[case['threshold_90']],
                colors='cyan', linewidths=2, linestyles='--')

    # Marcar puntos críticos
    ax1.plot(0, 0, 'wo', markersize=8, markeredgewidth=2, markeredgecolor='blue')
    ax1.plot(np.pi, 0, 'w^', markersize=10, markeredgewidth=2, markeredgecolor='lime')

    # Marcar zona de máxima intensidad
    ax1.plot(case['x_max'], case['y_max'], 'w*', markersize=15,
             markeredgewidth=1.5, markeredgecolor='black')

    # Rectángulo de región de interés
    rect = Rectangle((np.pi - 0.5, -0.5), 1.0, 1.0,
                     fill=False, edgecolor='cyan', linewidth=2, linestyle='-')
    ax1.add_patch(rect)

    ax1.set_xlim(0, 2*np.pi)
    ax1.set_ylim(-np.pi, np.pi)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title(f'c = {c_val:.2f}: Vorticidad Total ω\nω_max = {case["omega_max"]:.2f}',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax1.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax1.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax1.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax1.grid(alpha=0.3, linestyle=':')

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('ω', fontsize=10)

    # Panel central: Campo vectorial de ω̃
    ax2 = fig.add_subplot(gs[idx, 1])

    # Submuestreo para vectores
    skip = 15
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]
    U_sub = omegatil_x[::skip, ::skip]
    V_sub = omegatil_y[::skip, ::skip]
    mag_sub = np.sqrt(U_sub**2 + V_sub**2)

    im2 = ax2.contourf(X, Y, np.sqrt(omegatil_x**2 + omegatil_y**2),
                       levels=40, cmap='viridis', alpha=0.7)
    ax2.quiver(X_sub, Y_sub, U_sub, V_sub, mag_sub,
              cmap='plasma', scale=80, width=0.003, alpha=0.8)

    ax2.plot(np.pi, 0, 'w^', markersize=10, markeredgewidth=2, markeredgecolor='red')

    # Círculo de escala característica s
    circle = Circle((np.pi, 0), s_val, fill=False, edgecolor='yellow',
                   linewidth=2, linestyle='--', label=f'Radio s = {s_val:.3f}')
    ax2.add_patch(circle)

    ax2.set_xlim(0, 2*np.pi)
    ax2.set_ylim(-np.pi, np.pi)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title(f'c = {c_val:.2f}: Campo ω̃ (fluctuación)\nEscala s = {s_val:.3f}',
                  fontsize=12, fontweight='bold')
    ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax2.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax2.grid(alpha=0.3, linestyle=':')
    ax2.legend(loc='upper right', fontsize=9)

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('|ω̃|', fontsize=10)

    # Panel derecho: Zoom en región intensa (π, 0)
    ax3 = fig.add_subplot(gs[idx, 2])

    # Rango de zoom
    x_zoom = (np.pi - 1.0, np.pi + 1.0)
    y_zoom = (-1.0, 1.0)

    mask_zoom = (X >= x_zoom[0]) & (X <= x_zoom[1]) & \
                (Y >= y_zoom[0]) & (Y <= y_zoom[1])

    im3 = ax3.contourf(X, Y, omega, levels=50, cmap='hot')
    ax3.contour(X, Y, omega, levels=10, colors='white', linewidths=0.5, alpha=0.3)

    # Zonas de alta intensidad
    hot_zones_display = np.ma.masked_where(~case['hot_zones'], omega)
    ax3.contourf(X, Y, hot_zones_display, levels=20, cmap='plasma', alpha=0.5)

    ax3.plot(np.pi, 0, 'c^', markersize=12, markeredgewidth=2, markeredgecolor='white')
    ax3.plot(case['x_max'], case['y_max'], 'w*', markersize=18,
             markeredgewidth=1.5, markeredgecolor='black')

    ax3.set_xlim(x_zoom)
    ax3.set_ylim(y_zoom)
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('y', fontsize=11)
    ax3.set_title(f'c = {c_val:.2f}: Zoom en (π, 0)\nΓ_región = {case["Gamma_pi0"]:.2f}',
                  fontsize=12, fontweight='bold')
    ax3.set_xticks([np.pi - 0.5, np.pi, np.pi + 0.5])
    ax3.set_xticklabels(['π-0.5', 'π', 'π+0.5'])
    ax3.grid(alpha=0.4, linestyle=':')

    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('ω (zoom)', fontsize=10)

# Título general
fig.suptitle('MAPA DE INTENSIDAD DE WHIRLS - RÉGIMEN EXTREMO\n' +
             'Comparación c = 1.2, 1.25, 1.3 | Zonas intensas en cian/amarillo',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('intense_whirls_spatial_map.png', dpi=200, bbox_inches='tight')
print("✓ Guardado: intense_whirls_spatial_map.png")

# ============================================================================
# FIGURA 2: ANÁLISIS CUANTITATIVO Y PERFILES
# ============================================================================

print("\nGenerando Figura 2: Análisis cuantitativo...")

fig2 = plt.figure(figsize=(18, 10))
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.3, wspace=0.3)

# Panel 1: Perfiles radiales desde (π, 0)
ax1 = fig2.add_subplot(gs2[0, 0])

for idx, case in enumerate(results):
    c_val = case['c']
    X = case['X']
    Y = case['Y']
    omega = case['omega_total']

    # Perfil a lo largo de y = 0
    j_center = np.argmin(np.abs(Y[:, 0]))
    profile_x = omega[j_center, :]

    ax1.plot(X[j_center, :], profile_x, linewidth=2, label=f'c = {c_val:.2f}')

ax1.axvline(np.pi, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('ω(x, y=0)', fontsize=12)
ax1.set_title('Perfil de Vorticidad a lo largo de y = 0', fontsize=13, fontweight='bold')
ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax1.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 2*np.pi)

# Panel 2: Perfiles verticales en x = π
ax2 = fig2.add_subplot(gs2[0, 1])

for idx, case in enumerate(results):
    c_val = case['c']
    X = case['X']
    Y = case['Y']
    omega = case['omega_total']

    # Perfil a lo largo de x = π
    i_center = np.argmin(np.abs(X[0, :] - np.pi))
    profile_y = omega[:, i_center]

    ax2.plot(Y[:, i_center], profile_y, linewidth=2, label=f'c = {c_val:.2f}')

ax2.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_xlabel('y', fontsize=12)
ax2.set_ylabel('ω(x=π, y)', fontsize=12)
ax2.set_title('Perfil de Vorticidad a lo largo de x = π', fontsize=13, fontweight='bold')
ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax2.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
ax2.set_xlim(-np.pi, np.pi)

# Panel 3: Distribución de vorticidad (histograma)
ax3 = fig2.add_subplot(gs2[0, 2])

colors = ['red', 'orange', 'yellow']
for idx, case in enumerate(results):
    c_val = case['c']
    omega = case['omega_total']

    ax3.hist(omega.ravel(), bins=100, alpha=0.5,
             label=f'c = {c_val:.2f}', color=colors[idx], density=True)

ax3.set_xlabel('ω', fontsize=12)
ax3.set_ylabel('Densidad de probabilidad', fontsize=12)
ax3.set_title('Distribución de Vorticidad', fontsize=13, fontweight='bold')
ax3.set_yscale('log')
ax3.legend(fontsize=11)
ax3.grid(alpha=0.3)

# Panel 4: Circulación vs c
ax4 = fig2.add_subplot(gs2[1, 0])

c_plot = [case['c'] for case in results]
Gamma_plot = [case['Gamma_pi0'] for case in results]
omega_max_plot = [case['omega_max'] for case in results]

ax4_twin = ax4.twinx()

line1 = ax4.plot(c_plot, Gamma_plot, 'o-', linewidth=3, markersize=10,
                 color='blue', label='Γ(π±0.5, 0±0.5)')
ax4.set_xlabel('c', fontsize=12)
ax4.set_ylabel('Circulación Γ', fontsize=12, color='blue')
ax4.tick_params(axis='y', labelcolor='blue')
ax4.set_title('Circulación y Vorticidad Máxima vs c', fontsize=13, fontweight='bold')
ax4.grid(alpha=0.3)

line2 = ax4_twin.plot(c_plot, omega_max_plot, 's-', linewidth=3, markersize=10,
                      color='red', label='ω_max')
ax4_twin.set_ylabel('ω_max', fontsize=12, color='red')
ax4_twin.tick_params(axis='y', labelcolor='red')

# Combinar leyendas
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax4.legend(lines, labels, loc='upper left', fontsize=10)

# Panel 5: Escalas características
ax5 = fig2.add_subplot(gs2[1, 1])

s_plot = [case['s'] for case in results]
D_pi0 = [case['c'] - case['s'] for case in results]

ax5.plot(c_plot, s_plot, 'o-', linewidth=3, markersize=10,
         color='green', label='s = √(c²-1)')
ax5.plot(c_plot, D_pi0, 's-', linewidth=3, markersize=10,
         color='purple', label='D(π,0) = c - s')

ax5.set_xlabel('c', fontsize=12)
ax5.set_ylabel('Escala', fontsize=12)
ax5.set_title('Escalas Características vs c', fontsize=13, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(alpha=0.3)
ax5.axhline(0, color='black', linestyle='-', linewidth=0.8)

# Panel 6: Tabla resumen
ax6 = fig2.add_subplot(gs2[1, 2])
ax6.axis('off')

table_data = []
table_data.append(['c', 's', 'ω_max', 'Γ', 'Ratio'])
table_data.append(['', '', '', '', 'ω(π,0)/ω(0,0)'])

for case in results:
    omega_00 = 0.7022 if case['c'] == 1.2 else (0.6342 if case['c'] == 1.25 else 0.5806)
    ratio = case['omega_at_pi0'] / omega_00

    table_data.append([
        f"{case['c']:.2f}",
        f"{case['s']:.3f}",
        f"{case['omega_max']:.2f}",
        f"{case['Gamma_pi0']:.2f}",
        f"{ratio:.1f}×"
    ])

table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.15, 0.15, 0.18, 0.15, 0.20])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Estilo de encabezado
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(1, i)].set_facecolor('#81C784')
    table[(1, i)].set_text_props(style='italic', fontsize=9)

# Estilo de filas de datos
colors_rows = ['#FFE0B2', '#FFCC80', '#FFB74D']
for i, color in enumerate(colors_rows, start=2):
    for j in range(5):
        table[(i, j)].set_facecolor(color)

ax6.set_title('RESUMEN CUANTITATIVO\nRégimen Extremo c ∈ [1.2, 1.3]',
             fontsize=13, fontweight='bold', pad=20)

# Título general
fig2.suptitle('ANÁLISIS CUANTITATIVO - WHIRLS INTENSOS\n' +
              'Perfiles, Distribuciones y Métricas',
              fontsize=16, fontweight='bold', y=0.98)

plt.savefig('intense_whirls_quantitative.png', dpi=200, bbox_inches='tight')
print("✓ Guardado: intense_whirls_quantitative.png")

# ============================================================================
# FIGURA 3: MAPA DE LOCALIZACIÓN (caso c = 1.2)
# ============================================================================

print("\nGenerando Figura 3: Mapa detallado de localización (c=1.2)...")

fig3 = plt.figure(figsize=(16, 10))
gs3 = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.3, wspace=0.25)

case_best = results[0]  # c = 1.2
X = case_best['X']
Y = case_best['Y']
omega = case_best['omega_total']
omegatil_x = case_best['omegatil_x']
omegatil_y = case_best['omegatil_y']

# Panel 1: Mapa completo con división en celdas
ax1 = fig3.add_subplot(gs3[0, :])

im1 = ax1.contourf(X, Y, omega, levels=50, cmap='hot')
ax1.contour(X, Y, omega, levels=[case_best['threshold_90']],
           colors='cyan', linewidths=3, linestyles='-')

# Dibujar grid de celdas
n_cells_x, n_cells_y = 8, 6
for i in range(n_cells_x + 1):
    x_line = i * 2*np.pi / n_cells_x
    ax1.axvline(x_line, color='white', linestyle=':', linewidth=1, alpha=0.5)
for i in range(n_cells_y + 1):
    y_line = -np.pi + i * 2*np.pi / n_cells_y
    ax1.axhline(y_line, color='white', linestyle=':', linewidth=1, alpha=0.5)

# Marcar las top 3 zonas
cell_intensities = data['cell_intensities']
flat_idx = np.argsort(cell_intensities.ravel())[::-1]
top_cells = flat_idx[:3]

cell_width = 2*np.pi / n_cells_x
cell_height = 2*np.pi / n_cells_y

labels = ['🥇', '🥈', '🥉']
for rank, idx in enumerate(top_cells):
    i, j = np.unravel_index(idx, cell_intensities.shape)
    x_center = (j + 0.5) * cell_width
    y_center = -np.pi + (i + 0.5) * cell_height

    rect = Rectangle((j * cell_width, -np.pi + i * cell_height),
                     cell_width, cell_height,
                     fill=False, edgecolor='lime', linewidth=3)
    ax1.add_patch(rect)

    ax1.text(x_center, y_center, labels[rank],
            fontsize=24, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.plot(np.pi, 0, 'c^', markersize=15, markeredgewidth=2, markeredgecolor='white')
ax1.set_xlim(0, 2*np.pi)
ax1.set_ylim(-np.pi, np.pi)
ax1.set_xlabel('x', fontsize=13)
ax1.set_ylabel('y', fontsize=13)
ax1.set_title(f'MAPA DE LOCALIZACIÓN - c = {case_best["c"]:.2f}\n' +
             'Top 3 zonas más intensas marcadas | Umbral top 10% en cian',
             fontsize=14, fontweight='bold')
ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax1.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
ax1.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax1.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax1.grid(alpha=0.3)

cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.08)
cbar1.set_label('Vorticidad ω', fontsize=12)

# Panel 2: Mapa de calor de intensidad por celda
ax2 = fig3.add_subplot(gs3[1, 0])

im2 = ax2.imshow(cell_intensities, cmap='hot', aspect='auto',
                origin='lower', extent=[0, 2*np.pi, -np.pi, np.pi])

for rank, idx in enumerate(top_cells):
    i, j = np.unravel_index(idx, cell_intensities.shape)
    x_center = (j + 0.5) * cell_width
    y_center = -np.pi + (i + 0.5) * cell_height

    ax2.text(x_center, y_center, labels[rank],
            fontsize=20, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Intensidad Promedio por Celda\n⟨ω⟩ en cada región',
             fontsize=13, fontweight='bold')
ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax2.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('⟨ω⟩', fontsize=11)

# Panel 3: Mapa de circulación por celda
ax3 = fig3.add_subplot(gs3[1, 1])

cell_circulations = data['cell_circulations']
im3 = ax3.imshow(cell_circulations, cmap='plasma', aspect='auto',
                origin='lower', extent=[0, 2*np.pi, -np.pi, np.pi])

for rank, idx in enumerate(top_cells):
    i, j = np.unravel_index(idx, cell_intensities.shape)
    x_center = (j + 0.5) * cell_width
    y_center = -np.pi + (i + 0.5) * cell_height

    ax3.text(x_center, y_center, labels[rank],
            fontsize=20, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('Circulación por Celda\nΓ = ∬ ω dA en cada región',
             fontsize=13, fontweight='bold')
ax3.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax3.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
ax3.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax3.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

cbar3 = plt.colorbar(im3, ax=ax3)
cbar3.set_label('Γ', fontsize=11)

# Título general
fig3.suptitle('LOCALIZACIÓN ESPACIAL DETALLADA - WHIRLS INTENSOS\n' +
              f'c = {case_best["c"]:.2f} | Análisis por celdas 8×6',
              fontsize=16, fontweight='bold', y=0.97)

plt.savefig('intense_whirls_localization.png', dpi=200, bbox_inches='tight')
print("✓ Guardado: intense_whirls_localization.png")

# ============================================================================
# RESUMEN
# ============================================================================

print("\n" + "="*80)
print("VISUALIZACIONES COMPLETADAS")
print("="*80)
print("\n📊 Archivos generados:")
print("  1. intense_whirls_spatial_map.png - Mapas de intensidad (3 casos)")
print("  2. intense_whirls_quantitative.png - Análisis cuantitativo")
print("  3. intense_whirls_localization.png - Localización detallada (c=1.2)")
print("\n✅ Todas las visualizaciones guardadas exitosamente")
print("="*80)
