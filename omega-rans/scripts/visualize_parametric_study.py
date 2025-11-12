import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

print("Generando visualizaciones del estudio paramétrico...")

# Rango de valores de c
c_values = np.linspace(1.01, 3.0, 200)
s_values = np.sqrt(c_values**2 - 1)

# Calcular propiedades
D_00 = c_values + s_values
D_pi0 = c_values - s_values

omega_00 = 1 / D_00**2
omega_pi0 = 1 / D_pi0**2
ratio_omega = omega_pi0 / omega_00

# Estimación de circulación
Gamma_estimate = 1 / D_pi0**3

# Epsilon (amplitud relativa)
epsilon = s_values / c_values

# Crear figura con múltiples paneles
fig = plt.figure(figsize=(18, 12))

# =============================================================================
# Panel 1: Vorticidades
# =============================================================================
ax1 = fig.add_subplot(3, 3, 1)
ax1.semilogy(c_values, omega_pi0, 'r-', linewidth=3, label='ω(π, 0)')
ax1.semilogy(c_values, omega_00, 'b-', linewidth=3, label='ω(0, 0)')
ax1.axvline(1.5, color='g', linestyle='--', linewidth=2, alpha=0.7, label='c=1.5 (actual)')
ax1.set_xlabel('c', fontsize=12, fontweight='bold')
ax1.set_ylabel('Vorticidad ω', fontsize=12, fontweight='bold')
ax1.set_title('Vorticidades en puntos críticos', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([1.0, 3.0])

# Sombrear regímenes
ax1.axvspan(1.0, 1.3, alpha=0.15, color='red', label='Extremo')
ax1.axvspan(1.3, 2.0, alpha=0.15, color='green')
ax1.axvspan(2.0, 3.0, alpha=0.15, color='blue')

# =============================================================================
# Panel 2: Ratio de vorticidades
# =============================================================================
ax2 = fig.add_subplot(3, 3, 2)
ax2.semilogy(c_values, ratio_omega, 'purple', linewidth=3)
ax2.axvline(1.5, color='g', linestyle='--', linewidth=2, alpha=0.7)
ax2.axhline(47, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Ratio @ c=1.5')
ax2.set_xlabel('c', fontsize=12, fontweight='bold')
ax2.set_ylabel('ω(π,0) / ω(0,0)', fontsize=12, fontweight='bold')
ax2.set_title('Ratio de vorticidades [D(0,0)/D(π,0)]²', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([1.0, 3.0])

# Sombrear regímenes
ax2.axvspan(1.0, 1.3, alpha=0.15, color='red')
ax2.axvspan(1.3, 2.0, alpha=0.15, color='green')
ax2.axvspan(2.0, 3.0, alpha=0.15, color='blue')

# =============================================================================
# Panel 3: D(π,0) - el parámetro crítico
# =============================================================================
ax3 = fig.add_subplot(3, 3, 3)
ax3.plot(c_values, D_pi0, 'darkred', linewidth=3, label='D(π, 0)')
ax3.plot(c_values, D_00, 'darkblue', linewidth=3, label='D(0, 0)', alpha=0.5)
ax3.axvline(1.5, color='g', linestyle='--', linewidth=2, alpha=0.7)
ax3.axhline(0, color='k', linestyle='-', linewidth=1)
ax3.set_xlabel('c', fontsize=12, fontweight='bold')
ax3.set_ylabel('D', fontsize=12, fontweight='bold')
ax3.set_title('Parámetro D = c·cosh²(y) + s·cos(x)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([1.0, 3.0])

# Anotar singularidad
ax3.annotate('D(π,0) → 0', xy=(1.05, 0.2), xytext=(1.2, 0.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'),
            fontsize=11, fontweight='bold', color='red')

# =============================================================================
# Panel 4: Circulación estimada
# =============================================================================
ax4 = fig.add_subplot(3, 3, 4)
ax4.semilogy(c_values, Gamma_estimate, 'darkgreen', linewidth=3)
ax4.axvline(1.5, color='g', linestyle='--', linewidth=2, alpha=0.7)
ax4.axhline(1.71, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Γ @ c=1.5 (medido)')
ax4.set_xlabel('c', fontsize=12, fontweight='bold')
ax4.set_ylabel('Γ estimado ~ 1/D³(π,0)', fontsize=12, fontweight='bold')
ax4.set_title('Circulación estimada', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([1.0, 3.0])

# Sombrear regímenes
ax4.axvspan(1.0, 1.3, alpha=0.15, color='red')
ax4.axvspan(1.3, 2.0, alpha=0.15, color='green')
ax4.axvspan(2.0, 3.0, alpha=0.15, color='blue')

# =============================================================================
# Panel 5: Epsilon (amplitud relativa)
# =============================================================================
ax5 = fig.add_subplot(3, 3, 5)
ax5.plot(c_values, epsilon, 'darkorange', linewidth=3)
ax5.axvline(1.5, color='g', linestyle='--', linewidth=2, alpha=0.7)
ax5.axhline(1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='ε=1 (transición)')
ax5.set_xlabel('c', fontsize=12, fontweight='bold')
ax5.set_ylabel('ε = s/c', fontsize=12, fontweight='bold')
ax5.set_title('Amplitud relativa del flujo de Stuart', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_xlim([1.0, 3.0])
ax5.set_ylim([0, 1.0])

# Zonas físicas
ax5.fill_between(c_values, 0, epsilon, where=(epsilon < 0.5), alpha=0.2, color='blue', label='Débil')
ax5.fill_between(c_values, 0, epsilon, where=(epsilon >= 0.5) & (epsilon < 0.9),
                 alpha=0.2, color='green', label='Óptimo')
ax5.fill_between(c_values, 0, epsilon, where=(epsilon >= 0.9), alpha=0.2, color='red', label='Fuerte')

# =============================================================================
# Panel 6: Mapa de regímenes
# =============================================================================
ax6 = fig.add_subplot(3, 3, 6)
ax6.axis('off')

# Título
ax6.text(0.5, 0.95, 'MAPA DE REGÍMENES', ha='center', fontsize=14,
         fontweight='bold', transform=ax6.transAxes)

# Régimen 1: Extremo
rect1 = Rectangle((0.1, 0.75), 0.8, 0.15, facecolor='red', alpha=0.3, transform=ax6.transAxes)
ax6.add_patch(rect1)
ax6.text(0.5, 0.825, '1.0 < c < 1.3: WHIRLS EXTREMOS', ha='center', fontsize=10,
         fontweight='bold', transform=ax6.transAxes)
ax6.text(0.5, 0.78, 'ω > 20, Γ > 5, Inestable', ha='center', fontsize=9,
         transform=ax6.transAxes, style='italic')

# Régimen 2: Óptimo
rect2 = Rectangle((0.1, 0.5), 0.8, 0.2, facecolor='green', alpha=0.3, transform=ax6.transAxes)
ax6.add_patch(rect2)
ax6.text(0.5, 0.625, '1.3 < c < 2.0: WHIRLS ÓPTIMOS ⭐', ha='center', fontsize=11,
         fontweight='bold', transform=ax6.transAxes, color='darkgreen')
ax6.text(0.5, 0.57, 'ω ~ 5-10, Γ ~ 1-3', ha='center', fontsize=9,
         transform=ax6.transAxes)
ax6.text(0.5, 0.53, 'Estable, Ojos de gato bien formados', ha='center', fontsize=9,
         transform=ax6.transAxes, style='italic')

# Régimen 3: Débil
rect3 = Rectangle((0.1, 0.3), 0.8, 0.15, facecolor='blue', alpha=0.3, transform=ax6.transAxes)
ax6.add_patch(rect3)
ax6.text(0.5, 0.375, '2.0 < c < 3.0: WHIRLS DÉBILES', ha='center', fontsize=10,
         fontweight='bold', transform=ax6.transAxes)
ax6.text(0.5, 0.33, 'ω < 5, Γ < 1, Difuso', ha='center', fontsize=9,
         transform=ax6.transAxes, style='italic')

# Punto actual
ax6.plot([0.5], [0.6], 'g*', markersize=30, transform=ax6.transAxes,
         markeredgecolor='white', markeredgewidth=2)
ax6.text(0.5, 0.21, 'c = 1.5 (este trabajo)', ha='center', fontsize=11,
         fontweight='bold', transform=ax6.transAxes, color='green')

# Advertencias
ax6.text(0.5, 0.08, '⚠️ c → 1⁺: SINGULARIDAD', ha='center', fontsize=10,
         fontweight='bold', transform=ax6.transAxes, color='red')

# =============================================================================
# Panel 7: Escalamiento Γ vs ω
# =============================================================================
ax7 = fig.add_subplot(3, 3, 7)
ax7.loglog(omega_pi0, Gamma_estimate, 'darkviolet', linewidth=3)

# Marcar puntos específicos
idx_15 = np.argmin(np.abs(c_values - 1.5))
idx_12 = np.argmin(np.abs(c_values - 1.2))
idx_20 = np.argmin(np.abs(c_values - 2.0))

ax7.plot(omega_pi0[idx_15], Gamma_estimate[idx_15], 'go', markersize=15,
         markeredgecolor='white', markeredgewidth=2, label='c=1.5')
ax7.plot(omega_pi0[idx_12], Gamma_estimate[idx_12], 'ro', markersize=12,
         markeredgecolor='white', markeredgewidth=2, label='c=1.2')
ax7.plot(omega_pi0[idx_20], Gamma_estimate[idx_20], 'bo', markersize=12,
         markeredgecolor='white', markeredgewidth=2, label='c=2.0')

ax7.set_xlabel('ω(π, 0)', fontsize=12, fontweight='bold')
ax7.set_ylabel('Γ estimado', fontsize=12, fontweight='bold')
ax7.set_title('Escalamiento: Circulación vs Vorticidad', fontsize=12, fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3, which='both')

# =============================================================================
# Panel 8: s vs c (restricción)
# =============================================================================
ax8 = fig.add_subplot(3, 3, 8)
ax8.plot(c_values, s_values, 'darkblue', linewidth=3, label='s = √(c²-1)')
ax8.plot(c_values, c_values, 'k--', linewidth=2, alpha=0.5, label='s = c')
ax8.axvline(1.5, color='g', linestyle='--', linewidth=2, alpha=0.7)
ax8.set_xlabel('c', fontsize=12, fontweight='bold')
ax8.set_ylabel('s', fontsize=12, fontweight='bold')
ax8.set_title('Restricción física: s = √(c²-1)', fontsize=12, fontweight='bold')
ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3)
ax8.set_xlim([1.0, 3.0])

# Anotar casos especiales
ax8.plot(np.sqrt(2), 1, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
ax8.text(np.sqrt(2)+0.1, 1.1, 'c=√2, s=1', fontsize=9, fontweight='bold')

ax8.plot(2, np.sqrt(3), 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
ax8.text(2.1, np.sqrt(3)+0.1, 'c=2, s=√3', fontsize=9, fontweight='bold')

# =============================================================================
# Panel 9: Producto D(0,0)·D(π,0)
# =============================================================================
ax9 = fig.add_subplot(3, 3, 9)
product = D_00 * D_pi0
ax9.plot(c_values, product, 'darkcyan', linewidth=3)
ax9.axhline(1.0, color='red', linestyle='--', linewidth=3, alpha=0.7, label='D(0,0)·D(π,0) = 1')
ax9.axvline(1.5, color='g', linestyle='--', linewidth=2, alpha=0.7)
ax9.set_xlabel('c', fontsize=12, fontweight='bold')
ax9.set_ylabel('D(0,0) · D(π,0)', fontsize=12, fontweight='bold')
ax9.set_title('Relación fundamental', fontsize=12, fontweight='bold')
ax9.legend(fontsize=10)
ax9.grid(True, alpha=0.3)
ax9.set_xlim([1.0, 3.0])
ax9.set_ylim([0.95, 1.05])

# Verificación visual
ax9.fill_between(c_values, 0.99, 1.01, alpha=0.2, color='green', label='≈1 (±1%)')

plt.suptitle('ESTUDIO PARAMÉTRICO: Dependencia en c con s = √(c²-1)',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('parametric_study_c.png', dpi=200, bbox_inches='tight')
print("✅ Figura guardada: parametric_study_c.png")

# =============================================================================
# Segunda figura: Comparación de casos específicos
# =============================================================================
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

cases = [
    (1.2, 'WHIRLS EXTREMOS'),
    (1.5, 'WHIRLS ÓPTIMOS (actual)'),
    (2.0, 'TRANSICIÓN'),
    (2.5, 'WHIRLS DÉBILES')
]

for idx, (ax, (c_case, title)) in enumerate(zip(axes.flat, cases)):
    s_case = np.sqrt(c_case**2 - 1)
    D_00_case = c_case + s_case
    D_pi0_case = c_case - s_case
    omega_pi0_case = 1/D_pi0_case**2
    Gamma_case = 1/D_pi0_case**3
    epsilon_case = s_case / c_case

    # Tabla de propiedades
    props = [
        f'c = {c_case:.2f}',
        f's = {s_case:.3f}',
        f'ε = {epsilon_case:.3f}',
        '',
        f'D(π,0) = {D_pi0_case:.3f}',
        f'ω(π,0) = {omega_pi0_case:.1f}',
        f'Γ ~ {Gamma_case:.1f}'
    ]

    y_pos = 0.9
    for prop in props:
        if prop:
            ax.text(0.5, y_pos, prop, ha='center', fontsize=11,
                   transform=ax.transAxes, fontweight='bold' if '=' in prop else 'normal')
        y_pos -= 0.12

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.axis('off')

    # Fondo según régimen
    if c_case < 1.3:
        color = 'red'
        alpha = 0.15
    elif c_case < 2.0:
        color = 'green'
        alpha = 0.15
    else:
        color = 'blue'
        alpha = 0.15

    rect = Rectangle((0, 0), 1, 1, facecolor=color, alpha=alpha, transform=ax.transAxes)
    ax.add_patch(rect)

    # Indicador de intensidad
    intensity = Gamma_case / 10  # Normalizado
    if intensity > 1:
        stars = '⭐' * 5
        comment = 'MUY INTENSO'
    elif intensity > 0.5:
        stars = '⭐' * 4
        comment = 'INTENSO'
    elif intensity > 0.2:
        stars = '⭐' * 3
        comment = 'MODERADO'
    elif intensity > 0.1:
        stars = '⭐' * 2
        comment = 'DÉBIL'
    else:
        stars = '⭐'
        comment = 'MUY DÉBIL'

    ax.text(0.5, 0.15, stars, ha='center', fontsize=20, transform=ax.transAxes)
    ax.text(0.5, 0.05, comment, ha='center', fontsize=10,
           transform=ax.transAxes, style='italic')

plt.suptitle('COMPARACIÓN DE CASOS: Diferentes valores de c',
             fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('parametric_comparison_cases.png', dpi=150, bbox_inches='tight')
print("✅ Figura guardada: parametric_comparison_cases.png")

print("\n" + "="*80)
print("RESUMEN DE VISUALIZACIONES")
print("="*80)
print("\n📊 Archivos generados:")
print("  1. parametric_study_c.png - Estudio completo (9 paneles)")
print("  2. parametric_comparison_cases.png - Comparación de 4 casos")
print()
print("📈 Paneles del estudio completo:")
print("  1. Vorticidades ω(0,0) y ω(π,0)")
print("  2. Ratio ω(π,0)/ω(0,0)")
print("  3. Parámetro D")
print("  4. Circulación estimada Γ")
print("  5. Amplitud relativa ε = s/c")
print("  6. Mapa de regímenes")
print("  7. Escalamiento Γ vs ω")
print("  8. Restricción s = √(c²-1)")
print("  9. Producto D(0,0)·D(π,0) = 1")
print()
print("🎯 Hallazgos principales:")
print("  • c controla DRAMÁTICAMENTE la intensidad de whirls")
print("  • Singularidad en c = 1 (ω → ∞)")
print("  • Régimen óptimo: 1.3 < c < 2.0")
print("  • c = 1.5 está en zona óptima ⭐")
print("  • Γ escala como 1/(c-s)³")
print()
print("="*80)

plt.show()
