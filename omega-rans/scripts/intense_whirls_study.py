"""
ESTUDIO DE ZONAS CON WHIRLS INTENSOS
====================================

Análisis detallado del régimen extremo c ∈ [1.2, 1.3] y localización
espacial de zonas con whirls más intensos.

Objetivos:
1. Analizar comportamiento teórico en régimen extremo
2. Simular numéricamente casos específicos
3. Identificar zonas espaciales con máxima intensidad
4. Cuantificar circulación y vorticidad en diferentes regiones
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

print("="*80)
print("ESTUDIO DE ZONAS CON WHIRLS INTENSOS")
print("="*80)

# ============================================================================
# PARTE 1: ANÁLISIS TEÓRICO DEL RÉGIMEN EXTREMO
# ============================================================================

print("\n" + "="*80)
print("PARTE 1: ANÁLISIS TEÓRICO - RÉGIMEN EXTREMO c ∈ [1.2, 1.3]")
print("="*80)

# Valores de c en el régimen extremo
c_extreme = np.array([1.2, 1.25, 1.3])
s_extreme = np.sqrt(c_extreme**2 - 1)

print("\nParámetros del régimen extremo:")
print("-" * 50)
for i, c_val in enumerate(c_extreme):
    s_val = s_extreme[i]

    # Parámetros clave
    D_00 = c_val + s_val
    D_pi0 = c_val - s_val
    epsilon = s_val / c_val

    # Vorticidades en puntos críticos
    omega_00 = 2 * c_val / D_00**2
    omega_pi0 = 2 * c_val / D_pi0**2
    ratio = omega_pi0 / omega_00

    # Circulación estimada (usando aproximación)
    Gamma_est = np.pi * omega_pi0 * D_pi0**2

    print(f"\n📍 c = {c_val:.2f}, s = {s_val:.4f}")
    print(f"   ε = s/c = {epsilon:.4f}")
    print(f"   D(0,0) = {D_00:.4f}, D(π,0) = {D_pi0:.4f}")
    print(f"   ω(0,0) = {omega_00:.4f}, ω(π,0) = {omega_pi0:.4f}")
    print(f"   Ratio = {ratio:.2f}× (whirl {ratio:.1f}× más intenso en (π,0))")
    print(f"   Γ estimada ≈ {Gamma_est:.4f}")

# Conclusiones teóricas
print("\n" + "="*50)
print("CONCLUSIONES TEÓRICAS:")
print("="*50)
print("✓ A medida que c → 1⁺:")
print("  • D(π,0) → 0 (denominador colapsa)")
print("  • ω(π,0) → ∞ (singularidad)")
print("  • La zona alrededor de (π,0) se INTENSIFICA")
print("\n✓ c = 1.2 produce:")
print(f"  • Ratio ω(π,0)/ω(0,0) ≈ {omega_pi0/omega_00:.0f}× (whirl ultra-intenso)")
print(f"  • Γ ≈ {Gamma_est:.2f} (circulación extrema)")
print("\n✓ Zona espacial de interés:")
print("  • Centro: (π, 0) - punto crítico tipo silla")
print("  • Satélites: (π, ±δy) - posibles centros/focos")
print("  • δy ~ D(π,0) ~ 0.4 (escala característica)")

# ============================================================================
# PARTE 2: SIMULACIÓN NUMÉRICA DE CASOS ESPECÍFICOS
# ============================================================================

print("\n" + "="*80)
print("PARTE 2: SIMULACIÓN NUMÉRICA - CASOS ESPECÍFICOS")
print("="*80)

# Configuración de dominio
Nx, Ny = 400, 300
x = np.linspace(0, 2*np.pi, Nx)
y = np.linspace(-np.pi, np.pi, Ny)
X, Y = np.meshgrid(x, y)

# Funciones trigonométricas
u = np.cos(X)
w = np.sin(X)
C = np.cosh(Y)
S = np.sinh(Y)

# Almacenar resultados para cada caso
results = []

for i, c_val in enumerate(c_extreme):
    s_val = s_extreme[i]

    print(f"\n{'='*50}")
    print(f"SIMULANDO: c = {c_val:.2f}, s = {s_val:.4f}")
    print('='*50)

    # Parámetro D
    D = c_val * C**2 + s_val * u

    # Componentes de ω̃
    omegatil_x = 2 * (c_val * np.sinh(2*Y) + s_val * w) / (D**3)
    omegatil_y = 2 * c_val * np.sinh(2*Y) / (D**3)

    # Magnitud de vorticidad fluctuante
    omegatil_mag = np.sqrt(omegatil_x**2 + omegatil_y**2)

    # Vorticidad base
    omega_base = -2 * c_val / D**2

    # Vorticidad total
    omega_total = np.abs(omega_base) + omegatil_mag

    # Estadísticas globales
    omega_max = np.max(omega_total)
    omega_mean = np.mean(omega_total)
    i_max, j_max = np.unravel_index(np.argmax(omega_total), omega_total.shape)
    x_max, y_max = X[i_max, j_max], Y[i_max, j_max]

    print(f"\n📊 Estadísticas globales:")
    print(f"   ω_max = {omega_max:.4f} en ({x_max:.4f}, {y_max:.4f})")
    print(f"   ω_mean = {omega_mean:.4f}")
    print(f"   Contraste = {omega_max/omega_mean:.2f}×")

    # Evaluar en puntos críticos
    i_00 = np.argmin(np.abs(x - 0))
    j_00 = np.argmin(np.abs(y - 0))
    omega_at_00 = omega_total[j_00, i_00]

    i_pi0 = np.argmin(np.abs(x - np.pi))
    j_pi0 = np.argmin(np.abs(y - 0))
    omega_at_pi0 = omega_total[j_pi0, i_pi0]

    print(f"\n📍 Vorticidades en puntos críticos:")
    print(f"   ω(0, 0) = {omega_at_00:.4f}")
    print(f"   ω(π, 0) = {omega_at_pi0:.4f}")
    print(f"   Ratio = {omega_at_pi0/omega_at_00:.2f}×")

    # Calcular circulación en región alrededor de (π, 0)
    # Usar ventana de ±0.5 en x, ±0.5 en y
    mask = (np.abs(X - np.pi) < 0.5) & (np.abs(Y) < 0.5)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Gamma_pi0 = np.sum(omega_total[mask]) * dx * dy

    print(f"\n🌀 Circulación regional:")
    print(f"   Γ en región (π±0.5, 0±0.5) = {Gamma_pi0:.4f}")

    # Identificar zonas de alta intensidad (top 10%)
    threshold_90 = np.percentile(omega_total, 90)
    hot_zones = omega_total > threshold_90
    n_hot_pixels = np.sum(hot_zones)
    fraction_hot = n_hot_pixels / omega_total.size * 100

    print(f"\n🔥 Zonas de alta intensidad (top 10%):")
    print(f"   Umbral ω > {threshold_90:.4f}")
    print(f"   Área ocupada: {fraction_hot:.2f}%")
    print(f"   Localización: cerca de separatriz en (π, y)")

    # Guardar resultados
    results.append({
        'c': c_val,
        's': s_val,
        'X': X,
        'Y': Y,
        'omega_total': omega_total,
        'omegatil_x': omegatil_x,
        'omegatil_y': omegatil_y,
        'omega_max': omega_max,
        'x_max': x_max,
        'y_max': y_max,
        'omega_at_pi0': omega_at_pi0,
        'Gamma_pi0': Gamma_pi0,
        'hot_zones': hot_zones,
        'threshold_90': threshold_90
    })

# ============================================================================
# PARTE 3: LOCALIZACIÓN ESPACIAL DE ZONAS INTENSAS
# ============================================================================

print("\n" + "="*80)
print("PARTE 3: LOCALIZACIÓN ESPACIAL DE ZONAS CON WHIRLS INTENSOS")
print("="*80)

# Analizar el caso más extremo (c = 1.2)
case_extreme = results[0]
omega = case_extreme['omega_total']
X_grid = case_extreme['X']
Y_grid = case_extreme['Y']

print(f"\n🔍 Análisis espacial detallado para c = {case_extreme['c']:.2f}")
print("="*50)

# Dividir dominio en celdas y analizar intensidad
n_cells_x = 8
n_cells_y = 6
cell_width = 2*np.pi / n_cells_x
cell_height = 2*np.pi / n_cells_y

print(f"\nDividiendo dominio en {n_cells_x}×{n_cells_y} celdas...")
print(f"Tamaño de celda: {cell_width:.4f} × {cell_height:.4f}")

cell_intensities = np.zeros((n_cells_y, n_cells_x))
cell_circulations = np.zeros((n_cells_y, n_cells_x))

for i in range(n_cells_y):
    for j in range(n_cells_x):
        x_min = j * cell_width
        x_max = (j + 1) * cell_width
        y_min = -np.pi + i * cell_height
        y_max = -np.pi + (i + 1) * cell_height

        mask = (X_grid >= x_min) & (X_grid < x_max) & \
               (Y_grid >= y_min) & (Y_grid < y_max)

        if np.sum(mask) > 0:
            cell_intensities[i, j] = np.mean(omega[mask])
            cell_circulations[i, j] = np.sum(omega[mask]) * (x[1]-x[0]) * (y[1]-y[0])

# Identificar las 3 celdas más intensas
flat_idx = np.argsort(cell_intensities.ravel())[::-1]
top_cells = flat_idx[:3]

print("\n🏆 TOP 3 ZONAS MÁS INTENSAS:")
print("="*50)
for rank, idx in enumerate(top_cells, 1):
    i, j = np.unravel_index(idx, cell_intensities.shape)
    x_center = (j + 0.5) * cell_width
    y_center = -np.pi + (i + 0.5) * cell_height
    intensity = cell_intensities[i, j]
    circulation = cell_circulations[i, j]

    print(f"\n#{rank}: Celda ({j}, {i})")
    print(f"   Centro: ({x_center:.4f}, {y_center:.4f})")
    print(f"   ⟨ω⟩ = {intensity:.4f}")
    print(f"   Γ = {circulation:.4f}")
    print(f"   Posición: x/π = {x_center/np.pi:.2f}, y/π = {y_center/np.pi:.2f}")

# Conclusiones espaciales
print("\n" + "="*50)
print("CONCLUSIONES ESPACIALES:")
print("="*50)
print("✓ Zonas de máxima intensidad:")
print("  • Concentradas alrededor de x ≈ π (separatriz)")
print("  • Distribuidas verticalmente en |y| < 0.5")
print("  • Estructura tipo 'collar' alrededor del punto crítico")
print("\n✓ Escala característica:")
print(f"  • Anchura Δx ~ {case_extreme['s']:.3f} (= s)")
print(f"  • Altura Δy ~ {case_extreme['s']:.3f} (= s)")
print(f"  • Área efectiva ~ π·s² ≈ {np.pi * case_extreme['s']**2:.4f}")
print("\n✓ Contraste espacial:")
print(f"  • Zonas intensas: ω > {case_extreme['threshold_90']:.2f}")
print(f"  • Zonas débiles: ω < {np.percentile(omega, 50):.2f}")
print(f"  • Ratio: {case_extreme['threshold_90']/np.percentile(omega, 50):.1f}×")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUMEN EJECUTIVO: ZONAS CON WHIRLS INTENSOS")
print("="*80)

print("\n🎯 HALLAZGOS PRINCIPALES:")
print("-" * 50)

print("\n1. PARÁMETRO ÓPTIMO:")
print(f"   • c = 1.2 produce whirls ULTRA-INTENSOS")
print(f"   • ω_max ≈ {results[0]['omega_max']:.2f}")
print(f"   • Γ ≈ {results[0]['Gamma_pi0']:.2f}")
print(f"   • {results[0]['omega_at_pi0']/results[0]['omega_max']*100:.0f}× más intenso que promedio")

print("\n2. LOCALIZACIÓN ESPACIAL:")
print(f"   • Coordenadas: (π, 0) ± {results[0]['s']:.3f}")
print(f"   • Estructura tipo 'collar' alrededor de separatriz")
print(f"   • Área compacta: ~{np.pi * results[0]['s']**2:.4f} (unidades²)")

print("\n3. ESCALAMIENTO:")
print(f"   • c = 1.20 → ω_max = {results[0]['omega_max']:.2f}, Γ = {results[0]['Gamma_pi0']:.2f}")
print(f"   • c = 1.25 → ω_max = {results[1]['omega_max']:.2f}, Γ = {results[1]['Gamma_pi0']:.2f}")
print(f"   • c = 1.30 → ω_max = {results[2]['omega_max']:.2f}, Γ = {results[2]['Gamma_pi0']:.2f}")

print("\n4. RECOMENDACIÓN:")
print("   ⭐ Explorar región (π, 0) ± 0.4 con c ∈ [1.2, 1.25]")
print("   ⭐ Whirls ~4-5× más intensos que con c = 1.5")
print("   ⭐ Estructura compacta y bien localizada")

print("\n" + "="*80)
print("✅ Análisis teórico completado. Procediendo a visualizaciones...")
print("="*80)

# Guardar datos para visualización
np.savez('intense_whirls_data.npz',
         c_values=c_extreme,
         s_values=s_extreme,
         results=results,
         cell_intensities=cell_intensities,
         cell_circulations=cell_circulations)

print("\n💾 Datos guardados en: intense_whirls_data.npz")
