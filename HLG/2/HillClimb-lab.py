import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

def perturb(x,e):
    x1, x2 = x
    x1_new = np.clip(np.random.uniform(low=x1 - e, high=x1 + e), -2, 4)
    x2_new = np.clip(np.random.uniform(low=x2 - e, high=x2 + e), -2, 5)
    return (x1_new, x2_new)

def f(x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 1.7)**2))

x1_axis = np.linspace(-2, 4, 200)
x2_axis = np.linspace(-2, 5, 200)
X1, X2 = np.meshgrid(x1_axis, x2_axis)
Z = f(X1, X2)

x_opt = (
    np.random.uniform(low=-2, high=4),
    np.random.uniform(low=-2, high=5)
)
f_opt = f(*x_opt)

e = 1.0
max_iteracoes = 1000
max_vizinhos = 20
melhoria = True
i = 0
valores = [f_opt]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm, alpha=0.2, linewidth=1, antialiased=False)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')

while i < max_iteracoes and melhoria:
    melhoria = False
    for j in range(max_vizinhos):
        x_cand = perturb(x_opt, e)
        f_cand = f(*x_cand)
        if f_cand > f_opt:
            x_opt = x_cand
            f_opt = f_cand
            valores.append(f_opt)
            ax.scatter(x_opt[0], x_opt[1], f_cand, color='r', marker='o')
            melhoria = True
            break
    i += 1

print(f"Ponto encontrado: [{x_opt[0], x_opt[1]}]")

ax.scatter(x_opt[0], x_opt[1], f_cand, color='g', marker='x', s=100, label='Ponto Ótimo')
plt.show()