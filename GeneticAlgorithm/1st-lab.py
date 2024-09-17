import numpy as np
import matplotlib.pyplot as plt

p = 20
n_bits_var = 8
N = 100
epochs = 10
tolerancia = 1e-6

populacao_inicial = np.random.randint(low=0, high=2, size=(N, p * n_bits_var))
print(f"População inicial: {populacao_inicial}")

def f(x):
    A = 10
    p = len(x)
    somatorio = 0
    
    for i in range(p):
        somatorio += (x[i]**2) - A * np.cos(2 * np.pi * x[i])
    
    return somatorio

def phi(cromossomo, intervalo=(-10, 10)):
    s = 0
    
    for i in range(len(cromossomo)):
        s += cromossomo[len(cromossomo) - i - 1] * 2**i
    
    return intervalo[0] + (intervalo[1] - intervalo[0]) / (2**len(cromossomo) - 1) * s

cromossomo_exemplo = populacao_inicial[0]
variaveis_reais = phi(cromossomo_exemplo)
print("Variáveis Reais do Cromossomo Exemplo:\n", variaveis_reais)

t = 0
# melhor_aptidao = float('inf')
# geracoes_sem_melhoria = 0

# while t <= epochs:
#     aptidoes = []
    
#     for individuo in populacao_inicial:
#         variaveis_reais = phi(cromossomo=individuo)
#         aptidao = f(variaveis_reais)
#         aptidoes.append(aptidao)
    
#     melhor_aptidao_atual = min(aptidoes)
    
#     if abs(melhor_aptidao_atual - melhor_aptidao) < tolerancia:
#         geracoes_sem_melhoria += 1
#     else:
#         geracoes_sem_melhoria = 0
    
#     if geracoes_sem_melhoria >= 3:
#         print(f"Critério de parada atingido na geração {t}. Melhor aptidão: {melhor_aptidao_atual}")
#         break
    
#     melhor_aptidao = melhor_aptidao_atual
    
#     t += 1