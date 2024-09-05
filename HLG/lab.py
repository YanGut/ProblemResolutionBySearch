import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

def perturb(x,e):
    x1, x2 = x
    if (x1 < -100 or x1 > 100) or (x2 < -100 or x2 > 100):
        raise ValueError('x1 e x2 devem estar no intervalo [-100, 100]')
    x1_new = np.random.uniform(low=x1 - e, high=x1 + e)
    x2_new = np.random.uniform(low=x2 - e, high=x2 + e)
    return (x1_new, x2_new)

def f(x1, x2):
    return x1**2 + x2**2

x1_axis = np.linspace(-100, 100, 100)
x2_axis = np.linspace(-100, 100, 100)
X1, X2 = np.meshgrid(x1_axis, x2_axis)
Z = f(X1, X2)

x_opt = (
    np.random.uniform(low=-100, high=100),
    np.random.uniform(low=-100, high=100)
)
f_opt = f(*x_opt)

e = 1.0
max_iteracoes = 10000
max_vizinhos = 20
melhoria = True
i = 0
valores = [f_opt]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm, alpha=0.6, linewidth=1, antialiased=False)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')

while i < max_iteracoes and melhoria:
    melhoria = False
    for j in range(max_vizinhos):
        x_cand = perturb(x_opt, e)    # Gera uma nova perturbação
        f_cand = f(*x_cand)           # Calcula o valor da função no novo ponto
        if f_cand < f_opt:            # Queremos minimizar, então procuramos menor valor
            x_opt = x_cand
            f_opt = f_cand
            valores.append(f_opt)
            # ax.scatter(x_opt[0], x_opt[1], f_cand, c='red', marker='*', s=100, linewidth=3)
            melhoria = True
            break
    i += 1

ax.scatter(x_opt[0], x_opt[1], f_cand, c='green', marker='*', s=100, linewidth=3, label='Ponto Ótimo')
plt.legend()
plt.show()