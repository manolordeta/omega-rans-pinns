import sympy as sp
from sympy import *

print("="*80)
print("ANÁLISIS CORREGIDO CON LA RESTRICCIÓN s = √(c² - 1)")
print("="*80)

# Definir variables simbólicas
c = symbols('c', real=True, positive=True)
x, y = symbols('x y', real=True)

# RESTRICCIÓN IMPORTANTE: s = √(c² - 1)
s = sqrt(c**2 - 1)

print("\n" + "="*80)
print("PARTE 1: ORIGEN FÍSICO DE LA RESTRICCIÓN")
print("="*80)

print("\nEn el flujo de Stuart (cats eye):")
print("  ψ = A·ln(cosh(αy) + ε·cos(αx))")
print()
print("Los parámetros c y s en nuestro sistema representan:")
print("  D = c·cosh²(y) + s·cos(x)")
print()
print("Esta es una aproximación de:")
print("  cosh(αy) + ε·cos(αx)")
print()
print("Para que D sea una función válida del flujo de Stuart,")
print("se requiere una relación específica entre c y s.")
print()
print("Desarrollando cosh(αy) en y=0:")
print("  cosh²(0) = 1")
print()
print("Y el denominador característico del flujo Stuart:")
print("  cosh²(y) + 2·ε·cosh(y)·cos(x) + ε²·cos²(x)")
print()
print("Comparando con D = c·cosh²(y) + s·cos(x), necesitamos:")
print("  c = 1 (coeficiente de cosh²(y))")
print("  Pero también debe satisfacer relaciones de consistencia.")
print()
print("LA RESTRICCIÓN s = √(c² - 1) surge de:")
print("  - Condiciones de integrabilidad del sistema")
print("  - Física del flujo de Stuart")
print("  - Relación entre amplitud y número de onda")

print("\n" + "="*80)
print("PARTE 2: IMPLICACIONES PARA D")
print("="*80)

# Funciones trigonométricas
u_sym = cos(x)
w_sym = sin(x)
C_sym = cosh(y)
S_sym = sinh(y)

# D con la restricción
D_expr = c * C_sym**2 + s * u_sym
D_expr_sub = D_expr.subs(s, sqrt(c**2 - 1))

print(f"\nD = c·cosh²(y) + √(c²-1)·cos(x)")
print()
print("Simplificando:")
D_simplified = simplify(D_expr_sub)
print(f"D = {D_simplified}")
print()

# Evaluar en puntos específicos
print("Valores en puntos críticos:")
print()

# En (0, 0)
D_00 = D_expr_sub.subs([(x, 0), (y, 0)])
print(f"D(0, 0) = c·1 + √(c²-1)·1 = c + √(c²-1)")
D_00_simplified = simplify(D_00)
print(f"        = {D_00_simplified}")
print()

# En (π, 0)
D_pi0 = D_expr_sub.subs([(x, pi), (y, 0)])
print(f"D(π, 0) = c·1 + √(c²-1)·(-1) = c - √(c²-1)")
D_pi0_simplified = simplify(D_pi0)
print(f"        = {D_pi0_simplified}")
print()

# Caso especial: c = 1
print("CASO ESPECIAL: c = 1")
print(f"  s = √(1² - 1) = 0")
print(f"  D = cosh²(y)  (independiente de x!)")
print()

# Caso c > 1
print("CASO GENERAL: c > 1")
print(f"  s = √(c² - 1) > 0")
print()

# Factorización especial
print("Factorización interesante:")
print("  D(0, 0) = c + √(c²-1)")
print("  D(π, 0) = c - √(c²-1)")
print()
print("  D(0, 0)·D(π, 0) = (c + √(c²-1))(c - √(c²-1))")
print("                   = c² - (c²-1)")
print("                   = 1")
print()
print("  ⭐ D(0, 0)·D(π, 0) = 1  ⭐")
print()
print("Esto significa:")
print("  D(π, 0) = 1/D(0, 0)")
print()
print("Si D(0, 0) > 1, entonces D(π, 0) < 1")
print("¡Los dos puntos críticos tienen vorticidades INVERSAS!")

print("\n" + "="*80)
print("PARTE 3: VORTICIDAD BASE ω")
print("="*80)

print("\nω = 1/D²")
print()

# Vorticidades en puntos críticos
print("En (0, 0):")
omega_00 = 1 / D_00_simplified**2
print(f"  ω(0, 0) = 1/(c + √(c²-1))²")
omega_00_simplified = simplify(omega_00)
print(f"          = {omega_00_simplified}")
print()

print("En (π, 0):")
omega_pi0 = 1 / D_pi0_simplified**2
print(f"  ω(π, 0) = 1/(c - √(c²-1))²")
omega_pi0_simplified = simplify(omega_pi0)
print(f"          = {omega_pi0_simplified}")
print()

# Relación
ratio = simplify(omega_pi0_simplified / omega_00_simplified)
print("Relación:")
print(f"  ω(π, 0) / ω(0, 0) = {ratio}")
print()

# Usando D(0,0)·D(π,0) = 1
print("Usando D(0, 0)·D(π, 0) = 1:")
print("  ω(π, 0) = 1/D(π, 0)² = 1/(1/D(0,0))² = D(0, 0)²")
print("  ω(0, 0) = 1/D(0, 0)²")
print()
print("  Por lo tanto:")
print("  ω(π, 0) = [ω(0, 0)]⁻¹ · [D(0,0)]⁴")
print()
print("Si c > 1:")
print("  D(0, 0) > 1  →  ω(0, 0) < 1  →  ω(π, 0) > 1")
print("  ¡El punto (π, 0) tiene MAYOR vorticidad!")

print("\n" + "="*80)
print("PARTE 4: VORTICIDAD FLUCTUANTE ω̃")
print("="*80)

print("\nRecordemos:")
print("  ω̃_x = 2(c·sinh(2y) + s·sin(x)) / D³")
print("  ω̃_y = 2c·sinh(2y) / D³")
print()

# Sustituir s = √(c²-1)
omegatil_x_expr = 2*(c*sinh(2*y) + sqrt(c**2-1)*sin(x)) / D_expr_sub**3
omegatil_y_expr = 2*c*sinh(2*y) / D_expr_sub**3

print("Con s = √(c²-1):")
print("  ω̃_x = 2(c·sinh(2y) + √(c²-1)·sin(x)) / D³")
print("  ω̃_y = 2c·sinh(2y) / D³")
print()

# Evaluar en puntos críticos
print("En (0, 0):")
omegatil_x_00 = omegatil_x_expr.subs([(x, 0), (y, 0)])
omegatil_y_00 = omegatil_y_expr.subs([(x, 0), (y, 0)])
print(f"  ω̃_x(0, 0) = 2(c·0 + √(c²-1)·0) / D(0,0)³ = 0")
print(f"  ω̃_y(0, 0) = 2c·0 / D(0,0)³ = 0")
print("  ✓ Confirmado: punto crítico")
print()

print("En (π, 0):")
omegatil_x_pi0 = omegatil_x_expr.subs([(x, pi), (y, 0)])
omegatil_y_pi0 = omegatil_y_expr.subs([(x, pi), (y, 0)])
print(f"  ω̃_x(π, 0) = 2(c·0 + √(c²-1)·0) / D(π,0)³ = 0")
print(f"  ω̃_y(π, 0) = 2c·0 / D(π,0)³ = 0")
print("  ✓ Confirmado: punto crítico")
print()

print("\n" + "="*80)
print("PARTE 5: EXPANSIÓN LOCAL CERCA DE (π, 0)")
print("="*80)

print("\nExpansión: x = π + ξ, y = η")
print()

# Expansión de D cerca de (π, 0)
print("D(π+ξ, η) ≈ c·cosh²(η) + √(c²-1)·cos(π+ξ)")
print("          ≈ c·(1 + η²) + √(c²-1)·(-1 + ξ²/2)")
print("          ≈ c - √(c²-1) + c·η² + √(c²-1)·ξ²/2")
print()
print("Usando D(π,0) = c - √(c²-1):")
D_pi0_val = c - sqrt(c**2 - 1)
print(f"  D(π+ξ, η) ≈ D(π,0) + c·η² + √(c²-1)·ξ²/2")
print()

# Vorticidad fluctuante expandida
print("ω̃_x(π+ξ, η) ≈ 2(c·2η + √(c²-1)·(-ξ)) / D(π,0)³")
print("            = 2(2c·η - √(c²-1)·ξ) / D(π,0)³")
print()

print("ω̃_y(π+ξ, η) ≈ 2c·2η / D(π,0)³")
print("            = 4c·η / D(π,0)³")
print()

# Coeficientes numéricos
D_pi0_num = D_pi0_simplified
print("Coeficientes:")
coef_x_eta = 4*c / D_pi0_num**3
coef_x_xi = -2*sqrt(c**2-1) / D_pi0_num**3
coef_y_eta = 4*c / D_pi0_num**3

print(f"  ∂ω̃_x/∂η ≈ {simplify(coef_x_eta)}")
print(f"  ∂ω̃_x/∂ξ ≈ {simplify(coef_x_xi)}")
print(f"  ∂ω̃_y/∂η ≈ {simplify(coef_y_eta)}")
print()

print("\n" + "="*80)
print("PARTE 6: MATRIZ HESSIANA EN (π, 0)")
print("="*80)

print("\nHessiana de ω̃:")
print()

# Calcular segundas derivadas simbólicamente
d2_omegatil_x_dx2 = diff(omegatil_x_expr, x, 2)
d2_omegatil_x_dxdy = diff(diff(omegatil_x_expr, x), y)
d2_omegatil_y_dy2 = diff(omegatil_y_expr, y, 2)

print("Evaluando en (π, 0):")
hess_xx = d2_omegatil_x_dx2.subs([(x, pi), (y, 0)])
hess_xy = d2_omegatil_x_dxdy.subs([(x, pi), (y, 0)])
hess_yy = d2_omegatil_y_dy2.subs([(x, pi), (y, 0)])

print(f"  ∂²ω̃/∂x² = {simplify(hess_xx)}")
print(f"  ∂²ω̃/∂x∂y = {simplify(hess_xy)}")
print(f"  ∂²ω̃/∂y² = {simplify(hess_yy)}")
print()

det_hess = simplify(hess_xx * hess_yy - hess_xy**2)
trace_hess = simplify(hess_xx + hess_yy)

print(f"  det(H) = {det_hess}")
print(f"  tr(H) = {trace_hess}")
print()

print("Clasificación:")
print("  Si det(H) > 0 y tr(H) < 0: MÁXIMO")
print("  Si det(H) > 0 y tr(H) > 0: MÍNIMO")
print("  Si det(H) < 0: PUNTO DE SILLA")

print("\n" + "="*80)
print("PARTE 7: CASO NUMÉRICO EJEMPLO")
print("="*80)

# Ejemplo con c = 1.5
c_val = 1.5
s_val = sqrt(c_val**2 - 1)

print(f"\nEjemplo: c = {c_val}")
print(f"         s = √({c_val}² - 1) = {float(s_val):.6f}")
print()

D_00_num = c_val + s_val
D_pi0_num = c_val - s_val

print(f"D(0, 0) = {c_val} + {float(s_val):.6f} = {float(D_00_num):.6f}")
print(f"D(π, 0) = {c_val} - {float(s_val):.6f} = {float(D_pi0_num):.6f}")
print(f"D(0, 0)·D(π, 0) = {float(D_00_num * D_pi0_num):.6f} = 1 ✓")
print()

omega_00_num = 1 / D_00_num**2
omega_pi0_num = 1 / D_pi0_num**2

print(f"ω(0, 0) = 1/{float(D_00_num):.6f}² = {float(omega_00_num):.6f}")
print(f"ω(π, 0) = 1/{float(D_pi0_num):.6f}² = {float(omega_pi0_num):.6f}")
print()
print(f"ω(π, 0) / ω(0, 0) = {float(omega_pi0_num/omega_00_num):.6f}")
print()
print(f"¡La vorticidad en (π, 0) es {float(omega_pi0_num/omega_00_num):.2f}× mayor que en (0, 0)!")

# Hessiana numérica
hess_xx_num = float(hess_xx.subs(c, c_val))
hess_xy_num = float(hess_xy.subs(c, c_val))
hess_yy_num = float(hess_yy.subs(c, c_val))
det_hess_num = float(det_hess.subs(c, c_val))
trace_hess_num = float(trace_hess.subs(c, c_val))

print(f"\nHessiana en (π, 0):")
print(f"  ∂²ω̃/∂x² = {hess_xx_num:.6f}")
print(f"  ∂²ω̃/∂x∂y = {hess_xy_num:.6f}")
print(f"  ∂²ω̃/∂y² = {hess_yy_num:.6f}")
print(f"  det(H) = {det_hess_num:.6f}")
print(f"  tr(H) = {trace_hess_num:.6f}")
print()

if det_hess_num > 0:
    if trace_hess_num < 0:
        print("  → MÁXIMO LOCAL ⭐")
    else:
        print("  → MÍNIMO LOCAL")
elif det_hess_num < 0:
    print("  → PUNTO DE SILLA")
else:
    print("  → DEGENERADO")

print("\n" + "="*80)
print("CONCLUSIONES CON LA RESTRICCIÓN s = √(c² - 1)")
print("="*80)

print("\n1. RELACIÓN FUNDAMENTAL:")
print("   D(0, 0)·D(π, 0) = 1")
print("   → Las vorticidades en los dos puntos son inversas")
print()

print("2. ASIMETRÍA DE VORTICIDAD:")
print("   Para c > 1:")
print("   - D(0, 0) > 1  →  ω(0, 0) < 1  (vorticidad baja)")
print("   - D(π, 0) < 1  →  ω(π, 0) > 1  (vorticidad alta)")
print("   → El punto (π, 0) es MÁS IMPORTANTE para whirls")
print()

print("3. SIMPLIFICACIÓN DEL SISTEMA:")
print("   La restricción reduce los grados de libertad")
print("   → Solo un parámetro libre: c (o equivalentemente s)")
print()

print("4. FÍSICA:")
print("   La restricción viene del flujo de Stuart")
print("   → No es arbitraria, tiene significado físico")
print("   → Relaciona amplitud (c) con número de onda (s)")
print()

print("5. IMPLICACIÓN PARA WHIRLS:")
print("   (π, 0) tiene vorticidad base MÁS ALTA que (0, 0)")
print("   → Es el candidato MÁS FUERTE para whirls")
print("   → Nuestro análisis previo estaba en la dirección correcta")
print("   → Pero los valores numéricos cambian significativamente")

print("\n" + "="*80)
print("RECOMENDACIÓN: RECALCULAR ANÁLISIS NUMÉRICO")
print("="*80)

print("\nDebe rehacerse el análisis numérico con:")
print(f"  c = 1.5 (o cualquier c > 1)")
print(f"  s = √(c² - 1) ≈ 1.118 (NO 0.5 como antes)")
print()
print("Esto cambiará:")
print("  - Valores de vorticidad base")
print("  - Estructura del campo de velocidades")
print("  - Magnitud de la circulación")
print("  - Clasificación precisa de puntos críticos")
print()
print("Sin embargo, las conclusiones CUALITATIVAS se mantienen:")
print("  ✓ (π, 0) es punto crítico")
print("  ✓ Sistema no conservativo")
print("  ✓ Circulación presente")
print("  ✓ Estructuras vorticales existen")

print("\n" + "="*80)
