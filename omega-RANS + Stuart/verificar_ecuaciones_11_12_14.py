"""
VERIFICACIÓN COMPLETA: Ecuaciones 11, 12, 14 del Componente 2

Este script verifica si el ω̃ propuesto es consistente con:
- Ecuación 11: ∂ṽ₁/∂x + ∂ṽ₂/∂y = 0  (incompressibilidad)
- Ecuación 12: ∂ṽ₁/∂y - ∂ṽ₂/∂x = ω̃  (vorticidad)
- Ecuación 14: ∇²ṽ₂ = ∂ω̃/∂x         (Poisson)

Procedimiento:
1. Partir de ω̃ propuesto
2. Resolver Ec. 14 para obtener ṽ₂
3. Calcular ṽ₁ desde Ec. 11
4. Verificar Ec. 12 (consistencia)
5. Si es consistente, verificar Ec. 8 con vtil correctos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print("="*80)
print("VERIFICACIÓN: ECUACIONES 11, 12, 14 DEL COMPONENTE 2")
print("="*80)

# ==============================================================================
# PARTE 1: CONFIGURACIÓN Y ω̃ PROPUESTO
# ==============================================================================

print("\n" + "="*80)
print("PARTE 1: SOLUCIÓN PROPUESTA")
print("="*80)

# Parámetros
c = 1.5
s = np.sqrt(c**2 - 1)
print(f"\nParámetros:")
print(f"  c = {c}")
print(f"  s = √(c²-1) = {s:.6f}")

# Dominio (alta resolución para derivadas precisas)
Nx, Ny = 300, 250
x = np.linspace(0, 2*np.pi, Nx)
y = np.linspace(-np.pi, np.pi, Ny)
X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]

print(f"\nDominio:")
print(f"  x ∈ [0, 2π], Nx = {Nx}, dx = {dx:.6f}")
print(f"  y ∈ [-π, π], Ny = {Ny}, dy = {dy:.6f}")

# Funciones trigonométricas
u = np.cos(X)
w = np.sin(X)
C = np.cosh(Y)
S = np.sinh(Y)

# Parámetro D
D = c * C**2 + s * u

# Solución propuesta para ω̃
omegatil_prop = 2 * (c * np.sinh(2*Y) + s * w) / (D**3)  # Componente x en realidad
omegatil_x_prop = 2 * (c * np.sinh(2*Y) + s * w) / (D**3)
omegatil_y_prop = 2 * c * np.sinh(2*Y) / (D**3)

print(f"\n✓ Solución propuesta:")
print(f"  ω̃_x = 2(c·sinh(2y) + s·sin(x))/D³")
print(f"  ω̃_y = 2c·sinh(2y)/D³")

# Para este análisis, necesitamos el campo escalar ω̃
# Asumimos que ω̃_x y ω̃_y son componentes del gradiente de ω̃
# Pero realmente, en 2D, ω̃ es escalar (componente z del rotacional)

# Reinterpretación: ω̃ es la vorticidad fluctuante (escalar)
# Vamos a usar la magnitud o recalcular desde definición

print("\n⚠️  Nota sobre ω̃:")
print("  En 2D, vorticidad es escalar: ω̃ = ∂ṽ₁/∂y - ∂ṽ₂/∂x")
print("  Vamos a trabajar hacia atrás desde ω̃_x = ∂ω̃/∂x")

# ==============================================================================
# PARTE 2: RESOLVER ECUACIÓN 14 - Obtener ṽ₂
# ==============================================================================

print("\n" + "="*80)
print("PARTE 2: ECUACIÓN 14 - RESOLVER ∇²ṽ₂ = ω̃_x")
print("="*80)

print("\nEcuación 14 (Rosenfeld-Gröbner):")
print("  ∂²ṽ₂/∂y² = ∂ω̃/∂x - ∂²ṽ₂/∂x²")
print("  → ∇²ṽ₂ = ω̃_x")

print("\nResolviendo ecuación de Poisson con método iterativo (Gauss-Seidel)...")

# RHS: ω̃_x (ya lo tenemos)
RHS = omegatil_x_prop

# Inicializar ṽ₂
vtil2 = np.zeros_like(X)

# Condiciones de contorno: ṽ₂ = 0 en bordes
# (Esto es una elección; podríamos usar otras)

max_iter = 5000
tolerance = 1e-6

print(f"  Iteraciones máximas: {max_iter}")
print(f"  Tolerancia: {tolerance}")

for iteration in range(max_iter):
    vtil2_old = vtil2.copy()

    # Gauss-Seidel interior
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            vtil2[i, j] = ((vtil2[i+1, j] + vtil2[i-1, j]) * dy**2 +
                          (vtil2[i, j+1] + vtil2[i, j-1]) * dx**2 -
                          RHS[i, j] * dx**2 * dy**2) / (2*(dx**2 + dy**2))

    # Calcular error
    error = np.max(np.abs(vtil2 - vtil2_old))

    if iteration % 500 == 0:
        print(f"    Iteración {iteration}: error = {error:.6e}")

    if error < tolerance:
        print(f"  ✓ Convergencia alcanzada en {iteration} iteraciones")
        print(f"    Error final: {error:.6e}")
        break
else:
    print(f"  ⚠️  No convergió en {max_iter} iteraciones")
    print(f"    Error final: {error:.6e}")

print(f"\n✓ ṽ₂ calculado:")
print(f"  Rango: [{np.min(vtil2):.6f}, {np.max(vtil2):.6f}]")
print(f"  Media: {np.mean(vtil2):.6e}")

# ==============================================================================
# PARTE 3: CALCULAR ṽ₁ DESDE ECUACIÓN 11
# ==============================================================================

print("\n" + "="*80)
print("PARTE 3: ECUACIÓN 11 - CALCULAR ṽ₁ DESDE INCOMPRESSIBILIDAD")
print("="*80)

print("\nEcuación 11 (incompressibilidad):")
print("  ∂ṽ₁/∂x + ∂ṽ₂/∂y = 0")
print("  → ∂ṽ₁/∂x = -∂ṽ₂/∂y")

# Calcular ∂ṽ₂/∂y
dvtil2_dy = np.zeros_like(vtil2)
for i in range(1, Ny-1):
    dvtil2_dy[i, :] = (vtil2[i+1, :] - vtil2[i-1, :]) / (2*dy)

# Bordes
dvtil2_dy[0, :] = (vtil2[1, :] - vtil2[0, :]) / dy
dvtil2_dy[-1, :] = (vtil2[-1, :] - vtil2[-2, :]) / dy

print(f"\n✓ ∂ṽ₂/∂y calculado")
print(f"  Rango: [{np.min(dvtil2_dy):.6f}, {np.max(dvtil2_dy):.6f}]")

# Integrar -∂ṽ₂/∂y en x para obtener ṽ₁
# ṽ₁(x, y) = -∫ ∂ṽ₂/∂y dx + f(y)

vtil1 = np.zeros_like(vtil2)

# Integración acumulativa en x
for i in range(Ny):
    vtil1[i, 0] = 0  # Condición inicial: ṽ₁(x=0, y) = 0
    for j in range(1, Nx):
        vtil1[i, j] = vtil1[i, j-1] - dvtil2_dy[i, j] * dx

print(f"\n✓ ṽ₁ calculado por integración:")
print(f"  Rango: [{np.min(vtil1):.6f}, {np.max(vtil1):.6f}]")
print(f"  Media: {np.mean(vtil1):.6e}")

# Verificar incompressibilidad
dvtil1_dx = np.zeros_like(vtil1)
for j in range(1, Nx-1):
    dvtil1_dx[:, j] = (vtil1[:, j+1] - vtil1[:, j-1]) / (2*dx)

dvtil1_dx[:, 0] = (vtil1[:, 1] - vtil1[:, 0]) / dx
dvtil1_dx[:, -1] = (vtil1[:, -1] - vtil1[:, -2]) / dx

divergencia = dvtil1_dx + dvtil2_dy

print(f"\n✓ Verificación incompressibilidad (∇·ṽ):")
print(f"  |∇·ṽ| max: {np.max(np.abs(divergencia)):.6e}")
print(f"  |∇·ṽ| RMS: {np.sqrt(np.mean(divergencia**2)):.6e}")

if np.max(np.abs(divergencia)) < 1e-4:
    print("  ✅ Incompressibilidad SATISFECHA")
else:
    print("  ⚠️  Incompressibilidad tiene error residual")

# ==============================================================================
# PARTE 4: VERIFICAR ECUACIÓN 12 - Consistencia de ω̃
# ==============================================================================

print("\n" + "="*80)
print("PARTE 4: ECUACIÓN 12 - VERIFICAR ω̃ = ∂ṽ₁/∂y - ∂ṽ₂/∂x")
print("="*80)

print("\nEcuación 12 (definición vorticidad fluctuante):")
print("  ∂ṽ₁/∂y - ∂ṽ₂/∂x = ω̃")

# Calcular ∂ṽ₁/∂y
dvtil1_dy = np.zeros_like(vtil1)
for i in range(1, Ny-1):
    dvtil1_dy[i, :] = (vtil1[i+1, :] - vtil1[i-1, :]) / (2*dy)

dvtil1_dy[0, :] = (vtil1[1, :] - vtil1[0, :]) / dy
dvtil1_dy[-1, :] = (vtil1[-1, :] - vtil1[-2, :]) / dy

# Calcular ∂ṽ₂/∂x
dvtil2_dx = np.zeros_like(vtil2)
for j in range(1, Nx-1):
    dvtil2_dx[:, j] = (vtil2[:, j+1] - vtil2[:, j-1]) / (2*dx)

dvtil2_dx[:, 0] = (vtil2[:, 1] - vtil2[:, 0]) / dx
dvtil2_dx[:, -1] = (vtil2[:, -1] - vtil2[:, -2]) / dx

# Vorticidad reconstruida
omegatil_recon = dvtil1_dy - dvtil2_dx

print(f"\n✓ Vorticidad reconstruida desde ṽ:")
print(f"  Rango: [{np.min(omegatil_recon):.6f}, {np.max(omegatil_recon):.6f}]")

# Comparar con ω̃_x propuesto (que usamos como RHS de Poisson)
# Nota: ω̃_x es ∂ω̃/∂x, no ω̃ mismo
# Necesitamos comparar derivadas

# Calcular ∂ω̃_recon/∂x
domegatil_recon_dx = np.zeros_like(omegatil_recon)
for j in range(1, Nx-1):
    domegatil_recon_dx[:, j] = (omegatil_recon[:, j+1] - omegatil_recon[:, j-1]) / (2*dx)

domegatil_recon_dx[:, 0] = (omegatil_recon[:, 1] - omegatil_recon[:, 0]) / dx
domegatil_recon_dx[:, -1] = (omegatil_recon[:, -1] - omegatil_recon[:, -2]) / dx

# Comparar con ω̃_x propuesto
diferencia_omegatil_x = domegatil_recon_dx - omegatil_x_prop

print(f"\n✓ Comparación ∂ω̃/∂x:")
print(f"  Propuesto: [{np.min(omegatil_x_prop):.6f}, {np.max(omegatil_x_prop):.6f}]")
print(f"  Reconstruido: [{np.min(domegatil_recon_dx):.6f}, {np.max(domegatil_recon_dx):.6f}]")
print(f"  Diferencia max: {np.max(np.abs(diferencia_omegatil_x)):.6e}")
print(f"  Diferencia RMS: {np.sqrt(np.mean(diferencia_omegatil_x**2)):.6e}")

tolerancia = 1e-2
if np.max(np.abs(diferencia_omegatil_x)) < tolerancia:
    print(f"\n  ✅ CONSISTENCIA VERIFICADA (tolerancia = {tolerancia})")
    print(f"  La solución propuesta ES CONSISTENTE con Ecuaciones 11, 12, 14")
else:
    print(f"\n  ⚠️  INCONSISTENCIA DETECTADA (tolerancia = {tolerancia})")
    print(f"  Diferencia máxima: {np.max(np.abs(diferencia_omegatil_x)):.6e}")

# ==============================================================================
# PARTE 5: VERIFICAR ECUACIÓN 8 CON vtil CORRECTOS
# ==============================================================================

print("\n" + "="*80)
print("PARTE 5: ECUACIÓN 8 - VERIFICAR CON ṽ₁, ṽ₂ CORRECTOS")
print("="*80)

print("\nAhora que tenemos ṽ₁ y ṽ₂, verificamos Ecuación 8:")
print("  ∂ω̃/∂y = f(v₁, v₂, ṽ₁, ṽ₂, ∂ω̃/∂x)")

# Flujo base (Stuart)
v1 = 2*c*C*S / D
v2 = s*w / D

# Derivadas
w_x = u
D_y = 2*c*C*S
w2_minus_1 = w**2 - 1

# Calcular ∂ω̃/∂y reconstruido
domegatil_recon_dy = np.zeros_like(omegatil_recon)
for i in range(1, Ny-1):
    domegatil_recon_dy[i, :] = (omegatil_recon[i+1, :] - omegatil_recon[i-1, :]) / (2*dy)

domegatil_recon_dy[0, :] = (omegatil_recon[1, :] - omegatil_recon[0, :]) / dy
domegatil_recon_dy[-1, :] = (omegatil_recon[-1, :] - omegatil_recon[-2, :]) / dy

# Numerador de Ecuación 8
numerador = (4*c*C*S*v1*w2_minus_1 + 4*c*C*S*vtil1*w2_minus_1
             + 2*s*w_x*u*v1*w + 2*s*w_x*u*vtil1*w
             + 2*D_y*v2*w2_minus_1 + 2*D_y*vtil2*w2_minus_1
             - domegatil_recon_dx*D**3*v1*w2_minus_1
             - domegatil_recon_dx*D**3*vtil1*w2_minus_1)

# Denominador de Ecuación 8
denominador = D**3*v2*w2_minus_1 + D**3*vtil2*w2_minus_1

# Evitar división por cero
epsilon = 1e-10
mask_singular = np.abs(denominador) < epsilon
denominador_safe = np.where(mask_singular, epsilon, denominador)

# RHS de Ecuación 8
RHS_eq8 = numerador / denominador_safe

print(f"\n✓ Ecuación 8 evaluada:")
print(f"  Puntos singulares: {np.sum(mask_singular)}")
print(f"  RHS rango: [{np.min(RHS_eq8[~mask_singular]):.6f}, {np.max(RHS_eq8[~mask_singular]):.6f}]")

# Comparar LHS vs RHS
diferencia_eq8 = domegatil_recon_dy - RHS_eq8
diferencia_eq8_nosing = diferencia_eq8[~mask_singular]

print(f"\n✓ Comparación Ecuación 8:")
print(f"  LHS (∂ω̃/∂y): [{np.min(domegatil_recon_dy):.6f}, {np.max(domegatil_recon_dy):.6f}]")
print(f"  RHS: [{np.min(RHS_eq8[~mask_singular]):.6f}, {np.max(RHS_eq8[~mask_singular]):.6f}]")
print(f"  Diferencia max: {np.max(np.abs(diferencia_eq8_nosing)):.6e}")
print(f"  Diferencia RMS: {np.sqrt(np.mean(diferencia_eq8_nosing**2)):.6e}")

tolerancia_eq8 = 1e-2
if np.max(np.abs(diferencia_eq8_nosing)) < tolerancia_eq8:
    print(f"\n  ✅ ECUACIÓN 8 SATISFECHA (tolerancia = {tolerancia_eq8})")
else:
    print(f"\n  ❌ ECUACIÓN 8 NO SATISFECHA (tolerancia = {tolerancia_eq8})")
    print(f"  Se requiere iteración autoconsistente")

# ==============================================================================
# PARTE 6: VISUALIZACIÓN
# ==============================================================================

print("\n" + "="*80)
print("PARTE 6: GENERANDO VISUALIZACIÓN")
print("="*80)

fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

# Panel 1: ṽ₁ calculado
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.contourf(X, Y, vtil1, levels=40, cmap='RdBu_r')
ax1.set_title('ṽ₁ (calculado)', fontsize=11, fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
plt.colorbar(im1, ax=ax1)

# Panel 2: ṽ₂ calculado
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.contourf(X, Y, vtil2, levels=40, cmap='RdBu_r')
ax2.set_title('ṽ₂ (de Ec. 14)', fontsize=11, fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
plt.colorbar(im2, ax=ax2)

# Panel 3: Campo vectorial ṽ
ax3 = fig.add_subplot(gs[0, 2])
skip = 10
X_sub = X[::skip, ::skip]
Y_sub = Y[::skip, ::skip]
vtil1_sub = vtil1[::skip, ::skip]
vtil2_sub = vtil2[::skip, ::skip]
mag = np.sqrt(vtil1_sub**2 + vtil2_sub**2)
ax3.quiver(X_sub, Y_sub, vtil2_sub, vtil1_sub, mag, cmap='plasma', scale=5)
ax3.set_title('Campo ṽ = (ṽ₁, ṽ₂)', fontsize=11, fontweight='bold')
ax3.set_xlabel('x')
ax3.set_ylabel('y')

# Panel 4: Divergencia ∇·ṽ
ax4 = fig.add_subplot(gs[0, 3])
im4 = ax4.contourf(X, Y, divergencia, levels=40, cmap='seismic')
ax4.set_title(f'∇·ṽ (max={np.max(np.abs(divergencia)):.2e})', fontsize=11, fontweight='bold')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
plt.colorbar(im4, ax=ax4)

# Panel 5: ω̃ reconstruido
ax5 = fig.add_subplot(gs[1, 0])
im5 = ax5.contourf(X, Y, omegatil_recon, levels=40, cmap='viridis')
ax5.set_title('ω̃ reconstruido (Ec. 12)', fontsize=11, fontweight='bold')
ax5.set_xlabel('x')
ax5.set_ylabel('y')
plt.colorbar(im5, ax=ax5)

# Panel 6: ∂ω̃/∂x propuesto vs reconstruido
ax6 = fig.add_subplot(gs[1, 1])
im6 = ax6.contourf(X, Y, diferencia_omegatil_x, levels=40, cmap='seismic')
ax6.set_title(f'Dif: ∂ω̃/∂x (max={np.max(np.abs(diferencia_omegatil_x)):.2e})',
             fontsize=11, fontweight='bold')
ax6.set_xlabel('x')
ax6.set_ylabel('y')
plt.colorbar(im6, ax=ax6)

# Panel 7: ∂ω̃/∂y (LHS Ecuación 8)
ax7 = fig.add_subplot(gs[1, 2])
im7 = ax7.contourf(X, Y, domegatil_recon_dy, levels=40, cmap='RdBu_r')
ax7.set_title('∂ω̃/∂y (LHS Ec. 8)', fontsize=11, fontweight='bold')
ax7.set_xlabel('x')
ax7.set_ylabel('y')
plt.colorbar(im7, ax=ax7)

# Panel 8: RHS Ecuación 8
ax8 = fig.add_subplot(gs[1, 3])
RHS_masked = np.ma.masked_where(mask_singular, RHS_eq8)
im8 = ax8.contourf(X, Y, RHS_masked, levels=40, cmap='RdBu_r')
ax8.set_title('RHS Ecuación 8', fontsize=11, fontweight='bold')
ax8.set_xlabel('x')
ax8.set_ylabel('y')
plt.colorbar(im8, ax=ax8)

# Panel 9: Diferencia Ecuación 8
ax9 = fig.add_subplot(gs[2, 0])
dif_masked = np.ma.masked_where(mask_singular, diferencia_eq8)
im9 = ax9.contourf(X, Y, dif_masked, levels=40, cmap='seismic')
ax9.set_title(f'Dif Ec. 8 (max={np.max(np.abs(diferencia_eq8_nosing)):.2e})',
             fontsize=11, fontweight='bold')
ax9.set_xlabel('x')
ax9.set_ylabel('y')
plt.colorbar(im9, ax=ax9)

# Panel 10: Histograma diferencia Ec. 8
ax10 = fig.add_subplot(gs[2, 1])
ax10.hist(diferencia_eq8_nosing.ravel(), bins=100, color='blue', alpha=0.7, edgecolor='black')
ax10.axvline(0, color='red', linestyle='--', linewidth=2)
ax10.set_xlabel('LHS - RHS (Ec. 8)')
ax10.set_ylabel('Frecuencia')
ax10.set_title('Distribución Diferencia Ec. 8', fontsize=11, fontweight='bold')
ax10.set_yscale('log')
ax10.grid(alpha=0.3)

# Panel 11: Texto resumen
ax11 = fig.add_subplot(gs[2, 2:])
ax11.axis('off')

resumen_text = f"""
RESUMEN DE VERIFICACIÓN

Ecuación 11 (incompressibilidad):
  ∇·ṽ max = {np.max(np.abs(divergencia)):.6e}
  {'✅ SATISFECHA' if np.max(np.abs(divergencia)) < 1e-4 else '⚠️ ERROR RESIDUAL'}

Ecuación 12 (vorticidad):
  |∂ω̃/∂x_recon - ω̃_x_prop| max = {np.max(np.abs(diferencia_omegatil_x)):.6e}
  {'✅ CONSISTENTE' if np.max(np.abs(diferencia_omegatil_x)) < tolerancia else '⚠️ INCONSISTENTE'}

Ecuación 14 (Poisson):
  ✅ USADA para calcular ṽ₂

Ecuación 8:
  |LHS - RHS| max = {np.max(np.abs(diferencia_eq8_nosing)):.6e}
  |LHS - RHS| RMS = {np.sqrt(np.mean(diferencia_eq8_nosing**2)):.6e}
  {'✅ SATISFECHA' if np.max(np.abs(diferencia_eq8_nosing)) < tolerancia_eq8 else '❌ NO SATISFECHA'}

CONCLUSIÓN:
  La solución propuesta ω̃ {'ES' if np.max(np.abs(diferencia_eq8_nosing)) < tolerancia_eq8 else 'NO ES'}
  consistente con el sistema completo
  de ecuaciones del Componente 2.
"""

ax11.text(0.1, 0.5, resumen_text, fontsize=11, verticalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('VERIFICACIÓN COMPLETA: Ecuaciones 11, 12, 14, 8 del Componente 2',
            fontsize=14, fontweight='bold')

plt.savefig('verificacion_completa_ecuaciones.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualización guardada: verificacion_completa_ecuaciones.png")

# ==============================================================================
# CONCLUSIONES FINALES
# ==============================================================================

print("\n" + "="*80)
print("CONCLUSIONES FINALES")
print("="*80)

print("\n1. ECUACIONES 11, 12, 14 (Sistema básico):")
if np.max(np.abs(diferencia_omegatil_x)) < tolerancia:
    print("   ✅ La solución propuesta ES CONSISTENTE")
    print("   ✅ ṽ₁, ṽ₂ calculados satisfacen incompressibilidad")
    print("   ✅ ω̃ reconstruido coincide con propuesto")
else:
    print("   ⚠️  Inconsistencia en Ecuaciones 11-14")
    print(f"   Diferencia: {np.max(np.abs(diferencia_omegatil_x)):.6e}")

print("\n2. ECUACIÓN 8 (Ecuación compleja):")
if np.max(np.abs(diferencia_eq8_nosing)) < tolerancia_eq8:
    print("   ✅ SATISFECHA con ṽ₁, ṽ₂ correctos")
    print("   ✅ Solución propuesta es SOLUCIÓN COMPLETA del sistema")
else:
    print("   ❌ NO SATISFECHA completamente")
    print(f"   Diferencia máxima: {np.max(np.abs(diferencia_eq8_nosing)):.6e}")
    print("   Se requiere iteración autoconsistente")

print("\n3. PRÓXIMOS PASOS:")
if np.max(np.abs(diferencia_eq8_nosing)) >= tolerancia_eq8:
    print("   → Implementar iteración autoconsistente")
    print("   → Usar Ec. 8 para actualizar ω̃")
    print("   → Recalcular ṽ desde nuevo ω̃")
    print("   → Repetir hasta convergencia")
else:
    print("   ✓ Solución propuesta VALIDADA")
    print("   ✓ Puede usarse con confianza en análisis")

print("\n" + "="*80)
