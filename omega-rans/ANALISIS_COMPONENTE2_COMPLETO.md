# ANÁLISIS COMPLETO DEL COMPONENTE 2

## Sistema de Ecuaciones del Componente 2

Del output de Rosenfeld-Gröbner, tenemos **14 ecuaciones**:

### Ecuaciones 1-7: Relaciones algebraicas básicas

```
1. nu = 0                                    [Euler, sin viscosidad]
2. u² + w² = 1                               [identidad trigonométrica]
3. C² = (-s·u + D)/c                         [de definición de D]
4. S² = (-c - s·u + D)/c                     [otra relación]
5. ω = 1/D²                                  [vorticidad base]
6. ∂w/∂y = u                                 [∂sin(x)/∂y = 0, pero w es función]
7. ∂D/∂x = (2cCS(w²-1) + s·u·w·∂w/∂x)/(w²-1) [derivada de D]
```

### Ecuaciones 8: La famosa (relación ∂ω̃/∂y)

```
8. ∂ω̃/∂y = [numerador complejo] / [denominador]
```

### Ecuaciones 9-10: Flujo base (Stuart)

```
9.  ∂v₁/∂x = -∂v₂/∂y                        [incompressibilidad]
10. ∂v₁/∂y = (∂v₂/∂x · D² - 1)/D²          [relación vorticidad base]
```

### ⭐ Ecuaciones 11-12: VELOCIDAD FLUCTUANTE ⭐

```
11. ∂ṽ₁/∂x = -∂ṽ₂/∂y                       [incompressibilidad fluctuante]
12. ∂ṽ₁/∂y = ∂ṽ₂/∂x - ω̃                   [definición vorticidad fluctuante]
```

### Ecuaciones 13-14: EDPs de segundo orden

```
13. ∂²v₂/∂y² = (-4cCS(w²-1) - 2s·u·w·∂w/∂x - ∂²v₂/∂x²·D³(w²-1)) / (D³(w²-1))

14. ∂²ṽ₂/∂y² = ∂ω̃/∂x - ∂²ṽ₂/∂x²           [⭐ CLAVE: Ecuación de Poisson para ṽ₂]
```

---

## ¡DESCUBRIMIENTO CLAVE! Ecuaciones 11-12

Las Ecuaciones 11 y 12 nos dan **RELACIONES ENTRE ṽ₁, ṽ₂ Y ω̃**:

### Ecuación 11: Incompressibilidad
```
∂ṽ₁/∂x + ∂ṽ₂/∂y = 0
```

### Ecuación 12: Vorticidad
```
∂ṽ₁/∂y - ∂ṽ₂/∂x = ω̃
```

### ¿Qué podemos hacer con esto?

Si conocemos **ω̃(x,y)**, podemos **DERIVAR ṽ₁ y ṽ₂** usando estas ecuaciones!

---

## MÉTODO: Función de Corriente

De las Ecuaciones 11-12, podemos usar **función de corriente ψ̃**:

```
ṽ₁ = ∂ψ̃/∂y
ṽ₂ = -∂ψ̃/∂x
```

Esto automáticamente satisface Ecuación 11 (incompressibilidad).

Sustituyendo en Ecuación 12:
```
∂²ψ̃/∂y∂y - ∂(-∂ψ̃/∂x)/∂x = ω̃
∂²ψ̃/∂y² + ∂²ψ̃/∂x² = ω̃
∇²ψ̃ = ω̃
```

Pero espera... esto daría ∇²ψ̃ = ω̃, cuando debería ser ω̃ = -∇²ψ̃.

Déjame recalcular:

```
ṽ₁ = ∂ψ̃/∂y  →  ∂ṽ₁/∂y = ∂²ψ̃/∂y²
ṽ₂ = -∂ψ̃/∂x  →  ∂ṽ₂/∂x = -∂²ψ̃/∂x²

Ecuación 12:
∂ṽ₁/∂y - ∂ṽ₂/∂x = ω̃
∂²ψ̃/∂y² - (-∂²ψ̃/∂x²) = ω̃
∂²ψ̃/∂y² + ∂²ψ̃/∂x² = ω̃
∇²ψ̃ = ω̃
```

Hmm, esto no es la convención usual. Déjame verificar con la definición estándar:

Definición estándar de vorticidad 2D:
```
ω = ∇×v = ∂v₁/∂x - ∂v₂/∂y  (componente z)
```

Pero la Ecuación 12 dice:
```
∂ṽ₁/∂y - ∂ṽ₂/∂x = ω̃
```

Esto es el **negativo** de la definición usual. Entonces:

```
ω̃ = ∂ṽ₁/∂y - ∂ṽ₂/∂x = -(∂ṽ₂/∂x - ∂ṽ₁/∂y) = -∇×ṽ
```

Con función de corriente:
```
ṽ₁ = ∂ψ̃/∂y
ṽ₂ = -∂ψ̃/∂x

ω̃ = ∂²ψ̃/∂y² + ∂²ψ̃/∂x² = ∇²ψ̃
```

OK, entonces con esta convención:
```
∇²ψ̃ = ω̃
```

---

## ECUACIÓN 14: La Clave

La Ecuación 14 es:
```
∂²ṽ₂/∂y² = ∂ω̃/∂x - ∂²ṽ₂/∂x²
```

Reordenando:
```
∂²ṽ₂/∂x² + ∂²ṽ₂/∂y² = ∂ω̃/∂x
∇²ṽ₂ = ∂ω̃/∂x = ω̃ₓ
```

**¡Esta es una ecuación de Poisson para ṽ₂!**

Si conocemos ω̃(x,y), podemos calcular ω̃ₓ = ∂ω̃/∂x, y luego resolver:
```
∇²ṽ₂ = ω̃ₓ
```

para obtener ṽ₂!

---

## ESTRATEGIA: Iterar entre ω̃ y ṽ

### Algoritmo iterativo:

**Paso 1**: Proponer ω̃⁽⁰⁾ inicial (por ejemplo, la fórmula que tenemos)

**Paso 2**: Calcular ṽ₂⁽¹⁾ resolviendo:
```
∇²ṽ₂ = ∂ω̃⁽⁰⁾/∂x
```

**Paso 3**: Calcular ṽ₁⁽¹⁾ desde:
```
∂ṽ₁/∂x = -∂ṽ₂/∂y
```
Integrando en x.

**Paso 4**: Verificar Ecuación 12:
```
∂ṽ₁⁽¹⁾/∂y - ∂ṽ₂⁽¹⁾/∂x =? ω̃⁽⁰⁾
```

**Paso 5**: Si no se satisface, calcular ω̃⁽¹⁾ = ∂ṽ₁⁽¹⁾/∂y - ∂ṽ₂⁽¹⁾/∂x

**Paso 6**: Usar Ecuación 8 para actualizar ω̃:
```
∂ω̃⁽²⁾/∂y = RHS_Eq8(ṽ₁⁽¹⁾, ṽ₂⁽¹⁾, ω̃⁽¹⁾)
```
Integrar para obtener ω̃⁽²⁾.

**Paso 7**: Repetir pasos 2-6 hasta convergencia.

---

## VERIFICACIÓN: ¿Qué tenemos del análisis anterior?

En `numerical_whirl_corrected.py` (líneas 76-98), ya estábamos resolviendo:
```python
# Resolver ecuación de Poisson para ṽ2
# ∇²ṽ2 = ω̃_x

for iteration in range(max_iter):
    for i in range(1, len(y)-1):
        for j in range(1, len(x)-1):
            vtil2[i, j] = ... - omegatil_x[i, j] * dx**2 * dy**2 / (...)
```

¡Ya lo estábamos haciendo! Pero con ω̃ₓ fijo (no iterando).

---

## PLAN DE ACCIÓN

### Opción A: Usar Ecuación 14 directamente

**Script**: `resolver_con_ecuacion14.py`

```python
# 1. Usar ω̃ propuesto
omegatil_x = 2*(c*sinh(2*y) + s*sin(x))/D³
omegatil_y = 2*c*sinh(2*y)/D³

# 2. Resolver ∇²ṽ₂ = ω̃ₓ (Ecuación 14)
vtil2 = solve_poisson(omegatil_x)

# 3. Calcular ṽ₁ desde Ecuación 11
# ∂ṽ₁/∂x = -∂ṽ₂/∂y
dvtil2_dy = gradient(vtil2, axis=y)
vtil1 = -integrate(dvtil2_dy, axis=x)

# 4. Verificar Ecuación 12
omegatil_calculado = gradient(vtil1, axis=y) - gradient(vtil2, axis=x)

# 5. Comparar con ω̃ propuesto
diferencia = omegatil_calculado - omegatil_original

# 6. Si diferencia grande → iterar
```

### Opción B: Iterar autoconsistentemente

```python
# Inicializar
omegatil = omegatil_propuesto

for iter in range(max_iterations):
    # Resolver Ecuación 14
    vtil2 = solve_poisson(nabla^2 vtil2 = d_omegatil_dx)

    # Calcular vtil1 desde Ecuación 11
    vtil1 = calcular_vtil1(vtil2)

    # Calcular nuevo omegatil desde Ecuación 12
    omegatil_new = d_vtil1_dy - d_vtil2_dx

    # Verificar Ecuación 8
    check_eq8 = verificar_ecuacion8(omegatil_new, vtil1, vtil2)

    # Actualizar
    omegatil = alpha * omegatil_new + (1-alpha) * omegatil

    # Chequear convergencia
    if converged:
        break
```

---

## VENTAJAS DE ESTE ENFOQUE

✅ **Usa ecuaciones del Rosenfeld-Gröbner directamente**
- Ecuación 11: incompressibilidad
- Ecuación 12: definición vorticidad
- Ecuación 14: Poisson para ṽ₂

✅ **No necesita asumir vtil=0**
- Calculamos vtil₁, vtil₂ correctamente

✅ **Verificable en cada paso**
- Podemos chequear cada ecuación

✅ **Ya teníamos parte del código**
- El solver de Poisson ya existe

---

## DESVENTAJAS

⚠️ **Puede no converger**
- Iteración puede diverger o oscilar

⚠️ **Condiciones de contorno**
- Necesitamos especificar ṽ en bordes

⚠️ **Ecuación 8 sigue siendo compleja**
- Para iterar, necesitamos integrarla

---

## RECOMENDACIÓN INMEDIATA

### 🎯 Hacer primero: `verificar_ecuacion14.py`

```python
"""
Verificar si ω̃ propuesto es consistente con Ecuación 14
"""

# 1. Calcular ṽ₂ resolviendo ∇²ṽ₂ = ω̃ₓ
# 2. Calcular ṽ₁ desde incompressibilidad
# 3. Verificar si ω̃ = ∂ṽ₁/∂y - ∂ṽ₂/∂x
# 4. Reportar diferencia
```

Si esto funciona (diferencia pequeña), entonces:
- ✅ ω̃ propuesto ES consistente con Ecuaciones 11, 12, 14
- ✅ Solo falta verificar Ecuación 8

Si no funciona:
- ❌ Necesitamos iterar autoconsistentemente

---

## ECUACIONES CLAVE A USAR

```
Ecuación 11:  ∂ṽ₁/∂x + ∂ṽ₂/∂y = 0
Ecuación 12:  ∂ṽ₁/∂y - ∂ṽ₂/∂x = ω̃
Ecuación 14:  ∇²ṽ₂ = ω̃ₓ
```

De estas 3, podemos:
1. Calcular ṽ₂ de Ec. 14 (Poisson)
2. Calcular ṽ₁ de Ec. 11 (incompressibilidad)
3. Verificar Ec. 12 (consistencia)

¡Esto es más tractable que Ecuación 8 directamente!

---

**¿Quieres que implemente `verificar_ecuacion14.py`?**
