import numpy as np
import matplotlib.pyplot as plt

def calcular_ataques(rainhas):
    ataques = 0
    n = len(rainhas)
    
    for i in range(n):
        for j in range(i + 1, n):
            if rainhas[i] == rainhas[j]:  # Mesma linha
                ataques += 1
            if abs(rainhas[i] - rainhas[j]) == abs(i - j):  # Mesma diagonal
                ataques += 1
    return ataques

def f(rainhas):
    return 28 - calcular_ataques(rainhas)

def perturb(rainhas):
    nova_solucao = np.copy(rainhas)
    coluna = np.random.randint(low=0, high=len(rainhas))
    nova_solucao[coluna] = np.random.randint(0, 8)
    return nova_solucao

def definir_temperatura_inicial(rainhas_iniciais, num_simulacoes=100):
    diffs = []
    
    for _ in range(num_simulacoes):
        rainhas_cand = perturb(rainhas_iniciais)
        f_inicial = f(rainhas_iniciais)
        f_cand = f(rainhas_cand)
        diffs.append(abs(f_cand - f_inicial))

    # Calcula a média das diferenças
    media_diff = np.mean(diffs)

    # Define a temperatura inicial de modo que a probabilidade de aceitar uma solução pior seja alta (~80%)
    T0 = -media_diff / np.log(0.8)
    
    return T0

rainhas_opt = np.random.randint(low=0, high=8, size=8)
f_opt = f(rainhas_opt)

T0 = definir_temperatura_inicial(rainhas_iniciais=rainhas_opt)
T = T0

it_max = 1000
sigma = 0.2

i = 0
f_otimos = []

while i < it_max:
    rainhas_cand = perturb(rainhas_opt)
    f_cand = f(rainhas_cand)

    if f_cand > f_opt or np.exp(-(f_opt - f_cand) / T) >= np.random.uniform(0, 1):
        rainhas_opt = rainhas_cand
        f_opt = f_cand

    i += 1
    f_otimos.append(f_opt)
    T *= 0.99

# Mostra a solução encontrada
print(f'Solução final: {rainhas_opt}')
print(f'Número de pares de rainhas não atacantes: {f_opt}')

# Gráfico da evolução do valor ótimo
plt.plot(f_otimos)
plt.xlabel('Iteração')
plt.ylabel('f_opt (pares não atacantes)')
plt.title('Evolução do valor ótimo')
plt.show()