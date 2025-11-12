"""
VISUALIZACIÓN COMPLETA DEL FLUJO DE STUART (CATS EYE)
======================================================

Genera visualización detallada del flujo base mostrando:
- Líneas de corriente (cats eye pattern)
- Función de corriente ψ
- Vorticidad ω = -∇²ψ = 1/D²
- Campo de velocidades (v₁, v₂)
- Separatrices y puntos críticos
- Todas las propiedades relevantes del flujo base
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LogNorm

print("="*80)
print("VISUALIZACIÓN DEL FLUJO DE STUART (CATS EYE)")
print("="*80)

# Parámetros del flujo de Stuart
c = 1.5
s = np.sqrt(c**2 - 1)
print(f"\n📊 Parámetros:")
print(f"   c = {c:.4f}")
print(f"   s = √(c²-1) = {s:.4f}")
print(f"   ε = s/c = {s/c:.4f}")

# Dominio espacial (alta resolución para visualización)
Nx, Ny = 500, 400
x = np.linspace(0, 2*np.pi, Nx)
y = np.linspace(-np.pi, np.pi, Ny)
X, Y = np.meshgrid(x, y)

print(f"\n🔍 Dominio:")
print(f"   x ∈ [0, 2π], {Nx} puntos")
print(f"   y ∈ [-π, π], {Ny} puntos")

# Funciones trigonométricas
u = np.cos(X)
w = np.sin(X)
C = np.cosh(Y)
S = np.sinh(Y)

# Parámetro D
D = c * C**2 + s * u
print(f"\n📐 Parámetro D:")
print(f"   D(x,y) = c·cosh²(y) + s·cos(x)")
print(f"   D(0,0) = c + s = {c + s:.4f}")
print(f"   D(π,0) = c - s = {c - s:.4f}")
print(f"   D(0,0)·D(π,0) = {(c + s)*(c - s):.4f} = 1 ✓")

# Función de corriente ψ (normalizada)
# ψ = A·ln(D) donde A es constante arbitraria
# Elegimos A = 1 para simplificar
psi = np.log(D)

print(f"\n🌊 Función de corriente ψ:")
print(f"   ψ(x,y) = ln(D)")
print(f"   ψ_min = {np.min(psi):.4f}")
print(f"   ψ_max = {np.max(psi):.4f}")

# Componentes de velocidad
# v₁ = ∂ψ/∂y,  v₂ = -∂ψ/∂x
v1 = 2 * c * S / D
v2 = -s * w / D

# Magnitud de velocidad
v_mag = np.sqrt(v1**2 + v2**2)

print(f"\n🔄 Campo de velocidad:")
print(f"   v₁ = ∂ψ/∂y = 2c·sinh(y)/D")
print(f"   v₂ = -∂ψ/∂x = s·sin(x)/D")
print(f"   |v|_max = {np.max(v_mag):.4f}")

# Vorticidad ω = -∇²ψ = 1/D²  (para este flujo específico)
omega = -2 * c / D**2

print(f"\n🌀 Vorticidad ω:")
print(f"   ω = -∇²ψ = -2c/D²")
print(f"   ω(0,0) = {-2*c/(c+s)**2:.4f}")
print(f"   ω(π,0) = {-2*c/(c-s)**2:.4f}")
print(f"   Ratio |ω(π,0)/ω(0,0)| = {((c+s)/(c-s))**2:.2f}×")

# Identificar puntos críticos (v₁ = 0, v₂ = 0)
print(f"\n📍 Puntos críticos:")
print(f"   Tipo 1: (0, 0) - Centro")
print(f"   Tipo 2: (π, 0) - Silla (separatriz)")
print(f"   Tipo 3: (2π, 0) ≡ (0, 0) - Periodicidad")

# ============================================================================
# FIGURA COMPLETA: FLUJO DE STUART
# ============================================================================

print("\n" + "="*80)
print("Generando visualización completa...")
print("="*80)

fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# ============================================================================
# Panel 1: Líneas de corriente (CATS EYE) con vorticidad de fondo
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

# Fondo: vorticidad
im1 = ax1.contourf(X, Y, omega, levels=40, cmap='RdBu_r', alpha=0.6)

# Líneas de corriente
levels_stream = np.linspace(np.min(psi), np.max(psi), 30)
cs = ax1.contour(X, Y, psi, levels=levels_stream, colors='black',
                linewidths=1.5, alpha=0.8)

# Destacar separatriz (nivel ψ(π,0))
psi_separatrix = np.log(c - s)
ax1.contour(X, Y, psi, levels=[psi_separatrix], colors='red',
           linewidths=3, linestyles='-')

# Puntos críticos
ax1.plot(0, 0, 'go', markersize=15, markeredgewidth=2,
        markeredgecolor='darkgreen', label='Centro (0,0)', zorder=10)
ax1.plot(np.pi, 0, 'r^', markersize=15, markeredgewidth=2,
        markeredgecolor='darkred', label='Silla (π,0)', zorder=10)
ax1.plot(2*np.pi, 0, 'go', markersize=15, markeredgewidth=2,
        markeredgecolor='darkgreen', zorder=10)

# Marcar "ojos" del cats eye
eye_centers_x = [0, 2*np.pi]
for ex in eye_centers_x:
    circle = Circle((ex, 0), 0.8, fill=False, edgecolor='lime',
                   linewidth=2, linestyle='--', alpha=0.7)
    ax1.add_patch(circle)

ax1.set_xlim(0, 2*np.pi)
ax1.set_ylim(-np.pi, np.pi)
ax1.set_xlabel('x', fontsize=13)
ax1.set_ylabel('y', fontsize=13)
ax1.set_title('FLUJO DE STUART (CATS EYE)\n' +
             f'Líneas de corriente ψ = ln(D) | Fondo: vorticidad ω = -2c/D² | c={c:.2f}',
             fontsize=14, fontweight='bold')
ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax1.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
ax1.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax1.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(alpha=0.3, linestyle=':')

cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.08)
cbar1.set_label('Vorticidad ω', fontsize=12)

# ============================================================================
# Panel 2: Función de corriente ψ
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])

im2 = ax2.contourf(X, Y, psi, levels=40, cmap='viridis')
ax2.contour(X, Y, psi, levels=[psi_separatrix], colors='red',
           linewidths=2, linestyles='--')

ax2.plot(0, 0, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='black')
ax2.plot(np.pi, 0, 'w^', markersize=10, markeredgewidth=2, markeredgecolor='black')
ax2.plot(2*np.pi, 0, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='black')

ax2.set_xlim(0, 2*np.pi)
ax2.set_ylim(-np.pi, np.pi)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Función de Corriente ψ = ln(D)\nSeparatriz en rojo (ψ_sep = {:.3f})'.format(psi_separatrix),
             fontsize=12, fontweight='bold')
ax2.set_xticks([0, np.pi, 2*np.pi])
ax2.set_xticklabels(['0', 'π', '2π'])
ax2.grid(alpha=0.3)

cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('ψ', fontsize=11)

# ============================================================================
# Panel 3: Vorticidad ω = -2c/D²
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

im3 = ax3.contourf(X, Y, omega, levels=40, cmap='RdBu_r')
ax3.contour(X, Y, omega, levels=15, colors='black', linewidths=0.5, alpha=0.3)

ax3.plot(0, 0, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='blue')
ax3.plot(np.pi, 0, 'w^', markersize=10, markeredgewidth=2, markeredgecolor='red')
ax3.plot(2*np.pi, 0, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='blue')

ax3.set_xlim(0, 2*np.pi)
ax3.set_ylim(-np.pi, np.pi)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('Vorticidad Base ω = -2c/D²\nω(0,0) = {:.3f}, ω(π,0) = {:.3f}'.format(
             -2*c/(c+s)**2, -2*c/(c-s)**2),
             fontsize=12, fontweight='bold')
ax3.set_xticks([0, np.pi, 2*np.pi])
ax3.set_xticklabels(['0', 'π', '2π'])
ax3.grid(alpha=0.3)

cbar3 = plt.colorbar(im3, ax=ax3)
cbar3.set_label('ω', fontsize=11)

# ============================================================================
# Panel 4: Campo de velocidad |v|
# ============================================================================
ax4 = fig.add_subplot(gs[1, 2])

im4 = ax4.contourf(X, Y, v_mag, levels=40, cmap='plasma')

# Vectores de velocidad (submuestreado)
skip = 20
X_sub = X[::skip, ::skip]
Y_sub = Y[::skip, ::skip]
v1_sub = v1[::skip, ::skip]
v2_sub = v2[::skip, ::skip]

ax4.quiver(X_sub, Y_sub, v2_sub, v1_sub, v_mag[::skip, ::skip],
          cmap='plasma', scale=20, width=0.003, alpha=0.7)

ax4.plot(0, 0, 'wo', markersize=8, markeredgewidth=2, markeredgecolor='black')
ax4.plot(np.pi, 0, 'w^', markersize=8, markeredgewidth=2, markeredgecolor='black')

ax4.set_xlim(0, 2*np.pi)
ax4.set_ylim(-np.pi, np.pi)
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('y', fontsize=12)
ax4.set_title('Magnitud de Velocidad |v|\nv₁ = ∂ψ/∂y, v₂ = -∂ψ/∂x',
             fontsize=12, fontweight='bold')
ax4.set_xticks([0, np.pi, 2*np.pi])
ax4.set_xticklabels(['0', 'π', '2π'])
ax4.grid(alpha=0.3)

cbar4 = plt.colorbar(im4, ax=ax4)
cbar4.set_label('|v|', fontsize=11)

# ============================================================================
# Panel 5: Parámetro D
# ============================================================================
ax5 = fig.add_subplot(gs[2, 0])

im5 = ax5.contourf(X, Y, D, levels=40, cmap='coolwarm')
ax5.contour(X, Y, D, levels=[c-s, c, c+s], colors='black',
           linewidths=2, linestyles=['--', '-', '--'])

ax5.plot(0, 0, 'ko', markersize=10, label=f'D(0,0)={c+s:.2f}')
ax5.plot(np.pi, 0, 'k^', markersize=10, label=f'D(π,0)={c-s:.2f}')

ax5.set_xlim(0, 2*np.pi)
ax5.set_ylim(-np.pi, np.pi)
ax5.set_xlabel('x', fontsize=12)
ax5.set_ylabel('y', fontsize=12)
ax5.set_title('Parámetro D(x,y) = c·cosh²(y) + s·cos(x)\nD(0,0)·D(π,0) = 1',
             fontsize=12, fontweight='bold')
ax5.set_xticks([0, np.pi, 2*np.pi])
ax5.set_xticklabels(['0', 'π', '2π'])
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3)

cbar5 = plt.colorbar(im5, ax=ax5)
cbar5.set_label('D', fontsize=11)

# ============================================================================
# Panel 6: Perfil horizontal (y=0)
# ============================================================================
ax6 = fig.add_subplot(gs[2, 1])

j_center = np.argmin(np.abs(y))

ax6_twin = ax6.twinx()

# Perfil de ψ
line1 = ax6.plot(x, psi[j_center, :], 'b-', linewidth=2.5, label='ψ(x,0)')
ax6.axvline(np.pi, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax6.set_xlabel('x', fontsize=12)
ax6.set_ylabel('ψ(x, y=0)', fontsize=12, color='blue')
ax6.tick_params(axis='y', labelcolor='blue')

# Perfil de ω
line2 = ax6_twin.plot(x, omega[j_center, :], 'r-', linewidth=2.5, label='ω(x,0)')
ax6_twin.set_ylabel('ω(x, y=0)', fontsize=12, color='red')
ax6_twin.tick_params(axis='y', labelcolor='red')

ax6.set_title('Perfiles a lo largo de y = 0', fontsize=12, fontweight='bold')
ax6.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax6.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
ax6.grid(alpha=0.3)

# Combinar leyendas
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax6.legend(lines, labels, loc='upper right', fontsize=10)

# ============================================================================
# Panel 7: Perfil vertical (x=π)
# ============================================================================
ax7 = fig.add_subplot(gs[2, 2])

i_pi = np.argmin(np.abs(x - np.pi))

ax7_twin = ax7.twinx()

# Perfil de ψ
line3 = ax7.plot(y, psi[:, i_pi], 'b-', linewidth=2.5, label='ψ(π,y)')
ax7.axhline(psi_separatrix, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax7.set_xlabel('y', fontsize=12)
ax7.set_ylabel('ψ(x=π, y)', fontsize=12, color='blue')
ax7.tick_params(axis='y', labelcolor='blue')

# Perfil de ω
line4 = ax7_twin.plot(y, omega[:, i_pi], 'r-', linewidth=2.5, label='ω(π,y)')
ax7_twin.set_ylabel('ω(x=π, y)', fontsize=12, color='red')
ax7_twin.tick_params(axis='y', labelcolor='red')

ax7.set_title('Perfiles a lo largo de x = π (separatriz)', fontsize=12, fontweight='bold')
ax7.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax7.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax7.grid(alpha=0.3)

# Combinar leyendas
lines = line3 + line4
labels = [l.get_label() for l in lines]
ax7.legend(lines, labels, loc='upper right', fontsize=10)

# Título general
fig.suptitle('FLUJO DE STUART (CATS EYE) - SOLUCIÓN BASE\n' +
             f'c = {c:.2f}, s = √(c²-1) = {s:.3f}, ε = s/c = {s/c:.3f}',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('stuart_flow_complete.png', dpi=200, bbox_inches='tight')
print("\n✓ Visualización guardada: stuart_flow_complete.png")

# ============================================================================
# PROPIEDADES MATEMÁTICAS Y FÍSICAS
# ============================================================================

print("\n" + "="*80)
print("PROPIEDADES DEL FLUJO DE STUART")
print("="*80)

print("\n🔢 MATEMÁTICAS:")
print("-" * 50)
print(f"  Función de corriente:  ψ(x,y) = ln[c·cosh²(y) + s·cos(x)]")
print(f"  Velocidad horizontal:  v₁ = ∂ψ/∂y = 2c·sinh(y)/D")
print(f"  Velocidad vertical:    v₂ = -∂ψ/∂x = s·sin(x)/D")
print(f"  Vorticidad:            ω = -∇²ψ = -2c/D²")
print(f"  Relación fundamental:  D(0,0)·D(π,0) = 1")

print("\n⚖️ SIMETRÍAS:")
print("-" * 50)
print(f"  Periodicidad en x:     ψ(x+2π, y) = ψ(x, y)")
print(f"  Simetría en y:         ψ(x, -y) = ψ(x, y)")
print(f"  Antisimetría de v₁:    v₁(x, -y) = -v₁(x, y)")
print(f"  Simetría de v₂:        v₂(x, -y) = v₂(x, y)")

print("\n📐 TOPOLOGÍA:")
print("-" * 50)
print(f"  Puntos críticos:")
print(f"    • (0, 0):   Centro (ojo izquierdo)")
print(f"    • (π, 0):   Silla (separatriz)")
print(f"    • (2π, 0):  Centro (ojo derecho, periódico)")
print(f"  Separatrices:  Conectan sillas, dividen ojos de flujo externo")
print(f"  Región cerrada: 'Ojos' del cats eye donde partículas quedan atrapadas")

print("\n🌊 FÍSICA:")
print("-" * 50)
print(f"  Tipo de flujo:     Estacionario, 2D, incompresible")
print(f"  Conservación:      ∇·v = 0 (incompresibilidad)")
print(f"  Vorticidad máx:    ω(π,0) = {-2*c/(c-s)**2:.4f}")
print(f"  Vorticidad mín:    ω(0,0) = {-2*c/(c+s)**2:.4f}")
print(f"  Velocidad máx:     |v|_max ≈ {np.max(v_mag):.4f}")
print(f"  Parámetro ε:       {s/c:.4f} (amplitud relativa)")

print("\n🎯 APLICACIONES:")
print("-" * 50)
print(f"  • Modelo de flujo atmosférico (jet streams)")
print(f"  • Mixing en fluidos estratificados")
print(f"  • Transporte caótico en vórtices")
print(f"  • Flujo base para estudios de turbulencia")
print(f"  • Sistemas pseudoRANS (este trabajo)")

print("\n" + "="*80)
print("✅ Análisis completo del flujo de Stuart generado")
print("="*80)

# Guardar datos
np.savez('stuart_flow_data.npz',
         X=X, Y=Y, x=x, y=y,
         psi=psi, omega=omega,
         v1=v1, v2=v2, v_mag=v_mag,
         D=D, c=c, s=s,
         psi_separatrix=psi_separatrix)

print("\n💾 Datos guardados en: stuart_flow_data.npz")
