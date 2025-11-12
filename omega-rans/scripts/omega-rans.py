import sympy as sp
from sympy import *
from DifferentialAlgebra import *
init_printing ()

nu, c, s, alpha = var("nu, c, s, alpha")
x, y, t = var("x, y, t")
v1, v2, vtil1, vtil2, omega, omegatil, u, w, S, C, D, psi, psitil = indexedbase("v1, v2, vtil1, vtil2, omega, omegatil, u, w, S, C, D, psi, psitil")
cnsts = [nu, c, s]
cnsts2 = [nu]
cnsts3 = [nu, alpha]

R1 = DifferentialRing(derivations = [y, x],
                     blocks=[[vtil1, vtil2, v1, v2, omegatil, omega, S, C, D, u, w], cnsts],
                     parameters = cnsts,
                     notation = 'jet')

syst1 = [
  + (v1*omega[x] + v2*omega[y])
  + (v1*omegatil[x] + v2*omegatil[y])
  + (vtil1*omega[x] + vtil2*omega[y])
  + (vtil1*omegatil[x] + vtil2*omegatil[y])
  - nu*(omega[x,x] + omega[y,y] + omegatil[x,x] + omegatil[y,y]),

  v1[x] + v2[y],
  vtil1[x] + vtil2[y],

  omega - (v2[x] - v1[y]),
  omegatil - (vtil2[x] - vtil1[y]),

  u[y] + w,
  w[y] - u,
  u**2 + w**2 - 1,

  C[x] - S,
  S[x] - C,
  C**2 - S**2 - 1,

  D - (c*C**2 + s*u),
  D**2*omega - 1,


];

C1 = R1.RosenfeldGroebner(syst1)
result = [C.equations(solved=True) for C in C1]

print("="*80)
print("RESULTADOS DEL ANÁLISIS ROSENFELD-GROEBNER")
print("="*80)
print(f"\nNúmero de componentes características: {len(C1)}\n")

for i, C in enumerate(C1):
    print(f"\n{'='*80}")
    print(f"COMPONENTE {i+1}")
    print(f"{'='*80}")
    eqs = C.equations(solved=True)
    for j, eq in enumerate(eqs, 1):
        print(f"\nEcuación {j}:")
        print(eq)
    print()