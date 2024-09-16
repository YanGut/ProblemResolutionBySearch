import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def perturb(x, e=0.5):
    x = x + np.random.uniform(-e, e, size=x.shape)
    
    x[0] = np.clip(x[0], -1, 3)
    x[1] = np.clip(x[1], -1, 3)
    
    return x

def f(x):
    x1, x2 = x
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1

def plot_inicial():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1 = np.linspace(-1, 3, 100)
    x2 = np.linspace(-1, 3, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f([X1, X2])

    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('f(X1, X2)')
    return ax

def atualiza_plot(x_opt, f_opt, ax):
    ax.scatter(x_opt[0], x_opt[1], f_opt, color='r', marker='o')
    plt.pause(0.1)

x_opt = np.random.uniform(low=[-1, -1], high=[3, 3], size=2)
f_opt = f(x_opt)

ax = plot_inicial()

# Local Random Search: Melhorando a solução atual
max_it = 1000
for i in range(max_it):
    x_cand = perturb(x_opt)
    f_cand = f(x_cand)
    if f_cand > f_opt:
        x_opt = x_cand
        f_opt = f_cand
        atualiza_plot(x_opt, f_opt, ax)

ax.scatter(x_opt[0], x_opt[1], f_opt, color='g', marker='x', s=100, label='Ponto Ótimo')
plt.show()

print(f"Valor ótimo encontrado: x1 = {x_opt[0]}, x2 = {x_opt[1]}, f(x1, x2) = {f_opt}")
