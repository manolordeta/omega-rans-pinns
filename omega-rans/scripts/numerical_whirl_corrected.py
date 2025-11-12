import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("ANÁLISIS NUMÉRICO CORREGIDO CON s = √(c² - 1)")
print("="*80)

# Parámetros CORREGIDOS
c_val = 1.5
s_val = np.sqrt(c_val**2 - 1)

print(f"\nParámetros:")
print(f"  c = {c_val}")
print(f"  s = √(c² - 1) = {s_val:.6f}")
print()

# Verificar la relación fundamental
print("Verificación de la relación D(0,0)·D(π,0) = 1:")
D_00 = c_val + s_val
D_pi0 = c_val - s_val
print(f"  D(0, 0) = {D_00:.6f}")
print(f"  D(π, 0) = {D_pi0:.6f}")
print(f"  D(0, 0)·D(π, 0) = {D_00 * D_pi0:.6f} ✓")
print()

omega_00 = 1 / D_00**2
omega_pi0 = 1 / D_pi0**2
print(f"Vorticidades base:")
print(f"  ω(0, 0) = {omega_00:.6f}")
print(f"  ω(π, 0) = {omega_pi0:.6f}")
print(f"  Ratio: ω(π,0)/ω(0,0) = {omega_pi0/omega_00:.2f}× ⭐")
print()

# Crear grilla alrededor de (π, 0)
x_center = np.pi
y_center = 0.0
window_size = 1.5

x = np.linspace(x_center - window_size, x_center + window_size, 150)
y = np.linspace(y_center - window_size, y_center + window_size, 150)
X, Y = np.meshgrid(x, y)

# Funciones trigonométricas
u = np.cos(X)
w = np.sin(X)
C = np.cosh(Y)
S = np.sinh(Y)

# D con la restricción correcta
D = c_val * C**2 + s_val * u

# Vorticidad fluctuante CORREGIDA
omegatil_x = 2 * (c_val * np.sinh(2*Y) + s_val * w) / (D**3)
omegatil_y = 2 * c_val * np.sinh(2*Y) / (D**3)

# Vorticidad base
omega_base = 1.0 / (D**2)

print("="*80)
print("ESTADÍSTICAS GLOBALES")
print("="*80)

print(f"\nVorticidad base ω:")
print(f"  Máximo: {np.max(omega_base):.6f}")
print(f"  Mínimo: {np.min(omega_base):.6f}")
print(f"  En (π, 0): {omega_pi0:.6f}")

print(f"\nVorticidad fluctuante ω̃_x:")
print(f"  Máximo: {np.max(omegatil_x):.6f}")
print(f"  Mínimo: {np.min(omegatil_x):.6f}")

print(f"\nVorticidad fluctuante ω̃_y:")
print(f"  Máximo: {np.max(omegatil_y):.6f}")
print(f"  Mínimo: {np.min(omegatil_y):.6f}")

# Resolver ecuación de Poisson para ṽ2
print("\n" + "="*80)
print("RESOLUCIÓN DE ∇²ṽ2 = ω̃_x")
print("="*80)

dx = x[1] - x[0]
dy = y[1] - y[0]

vtil2 = np.zeros_like(X)
max_iter = 2000
tolerance = 1e-6

print(f"\n  Iterando (máx {max_iter} iteraciones)...")

for iteration in range(max_iter):
    vtil2_old = vtil2.copy()

    for i in range(1, len(y)-1):
        for j in range(1, len(x)-1):
            vtil2[i, j] = (vtil2[i+1, j] + vtil2[i-1, j]) / (2 + 2*dy**2/dx**2) + \
                         (vtil2[i, j+1] + vtil2[i, j-1]) * (dy**2/dx**2) / (2 + 2*dy**2/dx**2) - \
                         omegatil_x[i, j] * dx**2 * dy**2 / (2*(dx**2 + dy**2))

    error = np.max(np.abs(vtil2 - vtil2_old))
    if error < tolerance:
        print(f"  ✓ Convergencia en {iteration+1} iteraciones (error: {error:.2e})")
        break

    if (iteration + 1) % 500 == 0:
        print(f"    Iteración {iteration+1}: error = {error:.2e}")
else:
    print(f"  ⚠ Máximo de iteraciones ({max_iter}), error final: {error:.2e}")

print(f"\n  Estadísticas de ṽ2:")
print(f"    Máximo: {np.max(vtil2):.6f}")
print(f"    Mínimo: {np.min(vtil2):.6f}")

# Calcular ṽ1
vtil2_y = np.gradient(vtil2, dy, axis=0)
vtil1 = np.zeros_like(X)
for i in range(len(y)):
    vtil1[i, :] = -np.cumsum(vtil2_y[i, :]) * dx

print(f"\n  Estadísticas de ṽ1:")
print(f"    Máximo: {np.max(vtil1):.6f}")
print(f"    Mínimo: {np.min(vtil1):.6f}")

# Verificar incompresibilidad
vtil1_x = np.gradient(vtil1, dx, axis=1)
vtil2_y_check = np.gradient(vtil2, dy, axis=0)
div_vtil = vtil1_x + vtil2_y_check

print(f"\n  Incompresibilidad |∇·ṽ|:")
print(f"    Máximo: {np.max(np.abs(div_vtil)):.2e}")
print(f"    Medio: {np.mean(np.abs(div_vtil)):.2e}")

# Calcular vorticidad del campo
vtil2_x = np.gradient(vtil2, dx, axis=1)
vtil1_y = np.gradient(vtil1, dy, axis=0)
omegatil_calculated = vtil2_x - vtil1_y

print(f"\n  Vorticidad ω̃ = ∇×ṽ:")
print(f"    Máximo: {np.max(omegatil_calculated):.6f}")
print(f"    Mínimo: {np.min(omegatil_calculated):.6f}")

# Calcular circulación
print("\n" + "="*80)
print("CIRCULACIÓN ALREDEDOR DE (π, 0)")
print("="*80)

radii = np.linspace(0.1, 1.2, 12)
circulations = []

print(f"\n  Radio    Γ         Γ/r      ∬ω̃dA")
print("  " + "-"*45)

for radius in radii:
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    x_circle = x_center + radius * np.cos(theta)
    y_circle = y_center + radius * np.sin(theta)

    circulation = 0.0
    area_integral = 0.0

    for i in range(n_points-1):
        ix = np.argmin(np.abs(x - x_circle[i]))
        iy = np.argmin(np.abs(y - y_circle[i]))

        v1 = vtil1[iy, ix]
        v2 = vtil2[iy, ix]

        dx_circle = x_circle[i+1] - x_circle[i]
        dy_circle = y_circle[i+1] - y_circle[i]

        circulation += v1 * dx_circle + v2 * dy_circle
        area_integral += omegatil_calculated[iy, ix] * np.pi * radius**2 / n_points

    circulations.append(circulation)
    print(f"  {radius:.2f}   {circulation:+.4f}   {circulation/radius:+.3f}   {area_integral:+.4f}")

circulations = np.array(circulations)

print(f"\n  Circulación promedio: Γ_avg = {np.mean(circulations):.4f}")
print(f"  Desviación estándar: σ = {np.std(circulations):.4f}")

if np.mean(circulations) > 0.01:
    print(f"\n  ✅ CIRCULACIÓN POSITIVA SIGNIFICATIVA (antihorario)")
elif np.mean(circulations) < -0.01:
    print(f"\n  ✅ CIRCULACIÓN NEGATIVA SIGNIFICATIVA (horario)")
else:
    print(f"\n  ⚠ Circulación débil o nula")

# Jacobiana en (π, 0)
print("\n" + "="*80)
print("ANÁLISIS DEL PUNTO CRÍTICO (π, 0)")
print("="*80)

center_idx_x = np.argmin(np.abs(x - x_center))
center_idx_y = np.argmin(np.abs(y - y_center))

dvtil1_dx = vtil1_x[center_idx_y, center_idx_x]
dvtil1_dy = vtil1_y[center_idx_y, center_idx_x]
dvtil2_dx = vtil2_x[center_idx_y, center_idx_x]
dvtil2_dy = vtil2_y[center_idx_y, center_idx_x]

print(f"\n  Jacobiana J en (π, 0):")
print(f"    J = ⎡ {dvtil1_dx:+.6f}  {dvtil1_dy:+.6f} ⎤")
print(f"        ⎣ {dvtil2_dx:+.6f}  {dvtil2_dy:+.6f} ⎦")

eigenvalues = np.linalg.eigvals(np.array([
    [dvtil1_dx, dvtil1_dy],
    [dvtil2_dx, dvtil2_dy]
]))

print(f"\n  Autovalores: λ₁ = {eigenvalues[0]:.6f}, λ₂ = {eigenvalues[1]:.6f}")

det_J = np.linalg.det(np.array([
    [dvtil1_dx, dvtil1_dy],
    [dvtil2_dx, dvtil2_dy]
]))
trace_J = dvtil1_dx + dvtil2_dy

print(f"\n  det(J) = {det_J:.6f}")
print(f"  tr(J) = {trace_J:.6f}")
print(f"  Δ = tr²-4det = {trace_J**2 - 4*det_J:.6f}")

print(f"\n  Clasificación:")
if np.abs(det_J) < 1e-4:
    print(f"    → DEGENERADO (det ≈ 0)")
elif det_J < 0:
    print(f"    → PUNTO DE SILLA (flujo hiperbólico)")
elif det_J > 0:
    if trace_J**2 - 4*det_J < 0:
        if np.abs(trace_J) < 0.1:
            print(f"    → CENTRO (órbitas cerradas) ⭐")
            print(f"    → ¡WHIRL IDEAL!")
        elif trace_J < 0:
            print(f"    → FOCO ESTABLE (espiral convergente)")
        else:
            print(f"    → FOCO INESTABLE (espiral divergente)")
    else:
        if trace_J < 0:
            print(f"    → NODO ESTABLE")
        else:
            print(f"    → NODO INESTABLE")

# Comparación con análisis previo
print("\n" + "="*80)
print("COMPARACIÓN CON ANÁLISIS PREVIO")
print("="*80)

print(f"\n  PARÁMETROS:")
print(f"    Previo:    c = 1.0,  s = 0.5  (INCORRECTO)")
print(f"    Corregido: c = 1.5,  s = {s_val:.3f}")
print()

print(f"  VORTICIDAD EN (π, 0):")
D_prev = 1.0 - 0.5
omega_prev = 1 / D_prev**2
print(f"    Previo:    ω = {omega_prev:.3f}")
print(f"    Corregido: ω = {omega_pi0:.3f}")
print(f"    Factor:    {omega_pi0/omega_prev:.2f}×")
print()

print(f"  CIRCULACIÓN:")
Gamma_prev = 0.728
print(f"    Previo:    Γ ≈ {Gamma_prev:.3f}")
print(f"    Corregido: Γ ≈ {np.mean(circulations):.3f}")
print(f"    Factor:    {np.mean(circulations)/Gamma_prev:.2f}×")

print("\n" + "="*80)
print("CONCLUSIÓN FINAL CON PARÁMETROS CORRECTOS")
print("="*80)

print(f"\n🎯 CON LA RESTRICCIÓN s = √(c² - 1):")
print()
print(f"1. Vorticidad base en (π, 0): ω = {omega_pi0:.2f}")
print(f"   → {omega_pi0/omega_00:.0f}× mayor que en (0, 0)")
print()
print(f"2. Circulación: Γ ≈ {np.mean(circulations):.3f}")
if np.abs(np.mean(circulations)) > 0.01:
    print(f"   → Circulación SIGNIFICATIVA detectada")
else:
    print(f"   → Circulación débil")
print()
print(f"3. Punto crítico (π, 0):")
if np.abs(det_J) < 1e-4:
    print(f"   → Degenerado (requiere análisis de orden superior)")
elif det_J < 0:
    print(f"   → Punto de silla (NO es whirl clásico)")
elif det_J > 0 and trace_J**2 - 4*det_J < 0 and np.abs(trace_J) < 0.1:
    print(f"   → ¡CENTRO! (whirl ideal)")
else:
    print(f"   → Foco o nodo")
print()

if np.abs(np.mean(circulations)) > 0.01:
    print("✅ SE CONFIRMA PRESENCIA DE ESTRUCTURAS VORTICALES")
else:
    print("⚠️  Circulación débil - requier investigación adicional")

print("\n" + "="*80)

# Guardar datos
np.savez('whirl_data_corrected.npz',
         X=X, Y=Y,
         vtil1=vtil1, vtil2=vtil2,
         omegatil=omegatil_calculated,
         omega_base=omega_base,
         circulations=circulations,
         radii=radii,
         c=c_val, s=s_val)

print("\nDatos guardados en: whirl_data_corrected.npz")
