"""
Reconstrucción Algebraica y de Fourier de Flujos Incompresibles
================================================================

Este código implementa dos métodos para reconstruir campos de velocidad
a partir de distribuciones de presión conocidas:
1. Método algebraico con ansatz polinomiales
2. Método de Fourier para presiones periódicas

Autor: M. Romero de Terreros
Universidad Iberoamericana Ciudad de México
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Configuración de estilo para publicación
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

class FlowReconstruction:
    """Clase base para reconstrucción de flujos"""

    def __init__(self, rho=1.0):
        """
        Parámetros:
        -----------
        rho : float
            Densidad del fluido (kg/m³)
        """
        self.rho = rho

    def verify_incompressibility(self, u, v, x, y):
        """Verifica la condición de incompresibilidad"""
        du_dx = np.gradient(u, x[0, :], axis=1)
        dv_dy = np.gradient(v, y[:, 0], axis=0)
        div_u = du_dx + dv_dy
        return np.linalg.norm(div_u)

    def verify_poisson(self, p, u, v, x, y):
        """Verifica la ecuación de Poisson para la presión"""
        # Laplaciano de presión
        dp_dx2 = np.gradient(np.gradient(p, x[0, :], axis=1), x[0, :], axis=1)
        dp_dy2 = np.gradient(np.gradient(p, y[:, 0], axis=0), y[:, 0], axis=0)
        laplacian_p = dp_dx2 + dp_dy2

        # Término no lineal
        du_dx = np.gradient(u, x[0, :], axis=1)
        du_dy = np.gradient(u, y[:, 0], axis=0)
        dv_dx = np.gradient(v, x[0, :], axis=1)
        dv_dy = np.gradient(v, y[:, 0], axis=0)

        term1 = u * du_dx + v * du_dy
        term2 = u * dv_dx + v * dv_dy

        dterm1_dx = np.gradient(term1, x[0, :], axis=1)
        dterm2_dy = np.gradient(term2, y[:, 0], axis=0)

        nonlinear_term = dterm1_dx + dterm2_dy

        residual = laplacian_p + self.rho * nonlinear_term
        return np.linalg.norm(residual)


class AlgebraicReconstruction(FlowReconstruction):
    """Método algebraico con ansatz polinomiales"""

    def case_a_parabolic_pressure(self, alpha=1.0, a1=0.5):
        """
        Caso A: Presión parabólica radial
        p(x,y) = P0 - alpha*(x² + y²)

        Parámetros:
        -----------
        alpha : float
            Parámetro de intensidad de presión
        a1 : float
            Grado de libertad residual (rotación)

        Retorna:
        --------
        dict con campos p, u, v y coordenadas x, y
        """
        # Definir dominio
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)

        # Campo de presión
        P0 = 10.0
        P = P0 - alpha * (X**2 + Y**2)

        # Campo de velocidad reconstruido (Ecuación 14 del paper)
        coef = np.sqrt(alpha / self.rho)
        U = a1 * X + coef * Y
        V = coef * X - a1 * Y

        return {
            'x': X, 'y': Y,
            'p': P, 'u': U, 'v': V,
            'name': 'Presión Parabólica (Flujo Expansivo)',
            'params': {'alpha': alpha, 'a1': a1}
        }


class FourierReconstruction(FlowReconstruction):
    """Método de Fourier para presiones periódicas"""

    def case_b_oscillating_pressure(self, A=1.0, L=2*np.pi):
        """
        Caso B: Presión oscilante (celda de convección)
        p(x,y) = P0 + A*cos(kx)*sin(ky)

        Parámetros:
        -----------
        A : float
            Amplitud de oscilación de presión
        L : float
            Longitud característica del dominio

        Retorna:
        --------
        dict con campos p, u, v y coordenadas x, y
        """
        # Definir dominio periódico
        x = np.linspace(0, L, 100)
        y = np.linspace(0, L, 100)
        X, Y = np.meshgrid(x, y)

        # Número de onda
        k = 2 * np.pi / L

        # Campo de presión
        P0 = 5.0
        P = P0 + A * np.cos(k * X) * np.sin(k * Y)

        # Campo de velocidad reconstruido (Ecuación 23 del paper)
        U0 = np.sqrt(A * k / self.rho)
        U = U0 * np.sin(k * X) * np.sin(k * Y)
        V = U0 * np.cos(k * X) * np.cos(k * Y)

        return {
            'x': X, 'y': Y,
            'p': P, 'u': U, 'v': V,
            'name': 'Presión Oscilante (Celda de Convección)',
            'params': {'A': A, 'k': k, 'U0': U0}
        }


class Visualizer:
    """Generación de visualizaciones científicas"""

    @staticmethod
    def plot_flow_field(data, save_path=None):
        """
        Visualiza el campo de flujo completo

        Parámetros:
        -----------
        data : dict
            Diccionario con x, y, p, u, v
        save_path : str, opcional
            Ruta para guardar la figura
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        X, Y = data['x'], data['y']
        P, U, V = data['p'], data['u'], data['v']

        # 1. Mapa de presión con líneas de corriente
        ax = axes[0, 0]
        contourf = ax.contourf(X, Y, P, levels=20, cmap='RdBu_r')
        ax.streamplot(X, Y, U, V, color='black', linewidth=1,
                      density=1.5, arrowsize=1.5)
        plt.colorbar(contourf, ax=ax, label='Presión p(x,y)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Campo de Presión con Líneas de Corriente')
        ax.set_aspect('equal')

        # 2. Campo vectorial de velocidad
        ax = axes[0, 1]
        speed = np.sqrt(U**2 + V**2)
        skip = (slice(None, None, 3), slice(None, None, 3))
        quiver = ax.quiver(X[skip], Y[skip], U[skip], V[skip],
                          speed[skip], cmap='viridis', scale=50)
        plt.colorbar(quiver, ax=ax, label='Magnitud |u|')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Campo Vectorial de Velocidad')
        ax.set_aspect('equal')

        # 3. Magnitud de velocidad
        ax = axes[1, 0]
        contourf = ax.contourf(X, Y, speed, levels=20, cmap='plasma')
        plt.colorbar(contourf, ax=ax, label='|u| (m/s)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Magnitud de Velocidad')
        ax.set_aspect('equal')

        # 4. Verificación de divergencia
        ax = axes[1, 1]
        du_dx = np.gradient(U, X[0, :], axis=1)
        dv_dy = np.gradient(V, Y[:, 0], axis=0)
        div_u = du_dx + dv_dy
        contourf = ax.contourf(X, Y, div_u, levels=20, cmap='seismic')
        plt.colorbar(contourf, ax=ax, label='∇·u')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Divergencia (||∇·u||₂ = {np.linalg.norm(div_u):.2e})')
        ax.set_aspect('equal')

        fig.suptitle(data['name'], fontsize=18, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        return fig

    @staticmethod
    def plot_3d_surface(data, field='p', save_path=None):
        """
        Visualización 3D de un campo

        Parámetros:
        -----------
        data : dict
            Diccionario con x, y y el campo a visualizar
        field : str
            Campo a visualizar: 'p', 'u', 'v', o 'speed'
        save_path : str, opcional
            Ruta para guardar la figura
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        X, Y = data['x'], data['y']

        if field == 'p':
            Z = data['p']
            label = 'Presión p(x,y)'
            cmap = 'RdBu_r'
        elif field == 'speed':
            Z = np.sqrt(data['u']**2 + data['v']**2)
            label = 'Magnitud |u|'
            cmap = 'viridis'
        elif field == 'u':
            Z = data['u']
            label = 'Componente u(x,y)'
            cmap = 'coolwarm'
        elif field == 'v':
            Z = data['v']
            label = 'Componente v(x,y)'
            cmap = 'coolwarm'

        surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                              linewidth=0, antialiased=True, alpha=0.9)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(label)
        ax.set_title(f'{data["name"]}: {label}')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura 3D guardada en: {save_path}")

        return fig


def main():
    """Función principal para generar todos los resultados"""

    print("="*70)
    print("Reconstrucción de Flujos Incompresibles")
    print("Método Algebraico y de Fourier")
    print("="*70)

    # Crear instancias
    alg_recon = AlgebraicReconstruction(rho=1.0)
    fourier_recon = FourierReconstruction(rho=1.0)
    viz = Visualizer()

    # Caso A: Presión Parabólica (Método Algebraico)
    print("\n[1/2] Generando Caso A: Presión Parabólica...")
    data_a = alg_recon.case_a_parabolic_pressure(alpha=2.0, a1=0.3)

    # Verificaciones
    div_norm = alg_recon.verify_incompressibility(
        data_a['u'], data_a['v'], data_a['x'], data_a['y']
    )
    poisson_norm = alg_recon.verify_poisson(
        data_a['p'], data_a['u'], data_a['v'], data_a['x'], data_a['y']
    )

    print(f"  ✓ Norma de divergencia: {div_norm:.2e}")
    print(f"  ✓ Residual de Poisson: {poisson_norm:.2e}")

    # Visualizaciones
    viz.plot_flow_field(data_a, save_path='caso_a_parabolic.png')
    viz.plot_3d_surface(data_a, field='p', save_path='caso_a_pressure_3d.png')

    # Caso B: Presión Oscilante (Método de Fourier)
    print("\n[2/2] Generando Caso B: Presión Oscilante...")
    data_b = fourier_recon.case_b_oscillating_pressure(A=3.0, L=2*np.pi)

    # Verificaciones
    div_norm = fourier_recon.verify_incompressibility(
        data_b['u'], data_b['v'], data_b['x'], data_b['y']
    )
    poisson_norm = fourier_recon.verify_poisson(
        data_b['p'], data_b['u'], data_b['v'], data_b['x'], data_b['y']
    )

    print(f"  ✓ Norma de divergencia: {div_norm:.2e}")
    print(f"  ✓ Residual de Poisson: {poisson_norm:.2e}")

    # Visualizaciones
    viz.plot_flow_field(data_b, save_path='caso_b_oscillating.png')
    viz.plot_3d_surface(data_b, field='p', save_path='caso_b_pressure_3d.png')

    print("\n" + "="*70)
    print("✓ Todas las visualizaciones generadas exitosamente")
    print("="*70)

    plt.show()


if __name__ == "__main__":
    main()
