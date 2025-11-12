# Reconstrucción Algebraico-Diferencial de Flujos Incompresibles

Este repositorio contiene el código y documentación para el artículo de investigación sobre reconstrucción de campos de velocidad a partir de distribuciones de presión.

## Archivos Principales

### 📄 Documento LaTeX
- `reconstruccion-agebraica.tex` - Artículo completo en formato LaTeX

### 🐍 Código Python

1. **flow_reconstruction.py** - Implementación numérica principal
   - Método algebraico con ansatz polinomiales
   - Método de Fourier para presiones periódicas
   - Verificación de incompresibilidad y ecuación de Poisson
   - Generación de visualizaciones científicas

2. **symbolic_analysis.py** - Análisis simbólico con SymPy
   - Derivación automática de ecuaciones
   - Verificación analítica de soluciones
   - Generación de código LaTeX

## Instalación

### Requisitos
```bash
python >= 3.8
numpy
matplotlib
scipy
sympy
```

### Instalación de dependencias
```bash
pip install numpy matplotlib scipy sympy
```

## Uso

### 1. Generar todas las visualizaciones

```bash
python flow_reconstruction.py
```

Este script generará:
- `caso_a_parabolic.png` - Visualización completa del Caso A
- `caso_a_pressure_3d.png` - Vista 3D de presión parabólica
- `caso_b_oscillating.png` - Visualización completa del Caso B
- `caso_b_pressure_3d.png` - Vista 3D de presión oscilante

### 2. Ejecutar análisis simbólico

```bash
python symbolic_analysis.py
```

Este script mostrará:
- Derivación paso a paso de las ecuaciones
- Restricciones de incompresibilidad
- Soluciones analíticas
- Código LaTeX para el artículo

## Estructura del Proyecto

```
.
├── reconstruccion-agebraica.tex    # Artículo LaTeX
├── flow_reconstruction.py          # Implementación numérica
├── symbolic_analysis.py            # Análisis simbólico
├── README_reconstruccion.md        # Este archivo
├── caso_a_parabolic.png           # (Generado)
├── caso_a_pressure_3d.png         # (Generado)
├── caso_b_oscillating.png         # (Generado)
└── caso_b_pressure_3d.png         # (Generado)
```

## Casos de Estudio

### Caso A: Presión Parabólica Radial
- **Presión**: p(x,y) = P₀ - α(x² + y²)
- **Método**: Reducción algebraica
- **Flujo resultante**: Expansión radial con rotación

### Caso B: Presión Oscilante
- **Presión**: p(x,y) = P₀ + A·cos(kx)·sin(ky)
- **Método**: Series de Fourier
- **Flujo resultante**: Celdas de convección periódicas

## Personalización

### Modificar parámetros del Caso A
```python
data_a = alg_recon.case_a_parabolic_pressure(
    alpha=2.0,  # Intensidad de presión
    a1=0.3      # Rotación del flujo
)
```

### Modificar parámetros del Caso B
```python
data_b = fourier_recon.case_b_oscillating_pressure(
    A=3.0,           # Amplitud de oscilación
    L=2*np.pi        # Longitud del dominio
)
```

## Visualizaciones Incluidas

Cada caso genera 4 gráficos:
1. **Campo de presión con líneas de corriente**
2. **Campo vectorial de velocidad** (código de colores por magnitud)
3. **Magnitud de velocidad** (mapa de calor)
4. **Verificación de divergencia** (debe ser ≈0)

## Verificaciones Numéricas

El código verifica automáticamente:
- ✓ Incompresibilidad: ||∇·u||₂ < 10⁻¹²
- ✓ Ecuación de Poisson: ||∇²p + ρ∇·(u·∇u)||₂ < 10⁻¹⁰

## Extensiones Futuras

El código está diseñado para facilitar extensiones:
- Flujos viscosos (Navier-Stokes completa)
- Geometrías 3D
- Physics-Informed Neural Networks (PINNs)
- Casos con condiciones de frontera complejas

## Autor

**M. Romero de Terreros**
Departamento de Física y Matemáticas
Universidad Iberoamericana Ciudad de México
Email: manuel@verastrata.com

## Licencia

Este código es material complementario del artículo de investigación y está disponible para uso académico.

## Citas

Si utilizas este código en tu investigación, por favor cita:

```bibtex
@article{RomeroDeTerreros2025,
  title={Reconstrucción algebraico-diferencial de un flujo incompresible a partir de un campos de presión definido},
  author={Romero de Terreros, M.},
  journal={Journal of Fluid Dynamics Research},
  year={2025},
  publisher={Universidad Iberoamericana}
}
```

## Contacto

Para preguntas o sugerencias:
- Email: manuel@verastrata.com
- GitHub: [Agregar URL del repositorio]

---

**Última actualización**: Noviembre 2025
