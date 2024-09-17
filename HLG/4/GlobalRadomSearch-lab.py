import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def f(x):
    x1, x2 = x
    return (x1**2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2**2 - 10 * np.cos(2 * np.pi * x2) + 10)

def plot_inicial():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1 = np.linspace(-5.12, 5.12, 200)
    x2 = np.linspace(-5.12, 5.12, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f([X1, X2])

    ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm, alpha=0.6, linewidth=1, antialiased=False)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('f(X1, X2)')
    return ax

def atualiza_plot(x_opt, f_opt, ax):
    ax.scatter(x_opt[0], x_opt[1], f_opt, color='r', marker='o')
    plt.pause(0.1)

p = 2
max_sol = 500
eixo_lim_x1 = [-5.12, 5.12]
eixo_lim_x2 = [-5.12, 5.12]

# Solução inicial aleatória
solucoes = np.array([[
    np.random.uniform(eixo_lim_x1[0], eixo_lim_x1[1]), 
    np.random.uniform(eixo_lim_x2[0], eixo_lim_x2[1])
]])
avaliacoes = [f(solucoes[0, :])]

# Plot inicial
x_opt = solucoes[0, :]
f_opt = avaliacoes[0]
ax = plot_inicial()

i = 1
while i < max_sol:
    x = np.array([[
        np.random.uniform(eixo_lim_x1[0], eixo_lim_x1[1]), 
        np.random.uniform(eixo_lim_x2[0], eixo_lim_x2[1])
    ]])
    
    # Verifica se essa solução já foi gerada antes
    if not np.any(a=np.all(a=x == solucoes, axis=1), axis=0):
        solucoes = np.concatenate((solucoes, x))  # Armazena a nova solução
        f_cand = f(x[0, :])  # Avalia a nova solução
        avaliacoes.append(f_cand)  # Armazena a avaliação
        i += 1
        
        # Se a nova solução for melhor, atualiza a rota no gráfico
        if f_cand < f_opt:
            x_opt = x[0, :]
            f_opt = f_cand
            atualiza_plot(x_opt, f_opt, ax)

ax.scatter(x_opt[0], x_opt[1], f_opt, color='g', marker='x', s=100)
plt.show()

print(f"Valor ótimo encontrado: x1 = {x_opt[0]}, x2 = {x_opt[1]}, f(x1, x2) = {f_opt}")
