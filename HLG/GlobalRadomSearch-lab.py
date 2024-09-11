import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    x1, x2 = x
    return x1**2 + x2**2

def plot_inicial():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Gera os dados para o gráfico de superfície
    x1 = np.linspace(-100, 100, 200)
    x2 = np.linspace(-100, 100, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f([X1, X2])

    # Plotar a superfície da função
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('f(X1, X2)')
    return ax

# Função para atualizar o gráfico durante a execução
def atualiza_plot(x_opt, f_opt, ax):
    ax.scatter(x_opt[0], x_opt[1], f_opt, color='r', marker='o')
    plt.pause(0.1)

p = 2
max_sol = 500
eixo_lim = [-100, 100]

# Solução inicial aleatória
solucoes = np.random.uniform(low=eixo_lim[0], high=eixo_lim[1], size=(1, p))
avaliacoes = [f(solucoes[0, :])]

# Plot inicial
x_opt = solucoes[0, :]
f_opt = avaliacoes[0]
ax = plot_inicial()

i = 1
while i < max_sol:
    x = np.random.uniform(eixo_lim[0], eixo_lim[1], (1, p))
    
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
