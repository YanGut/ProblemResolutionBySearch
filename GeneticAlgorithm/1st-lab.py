import numpy as np
import matplotlib.pyplot as plt

p = 20
n_bits_var = 10
N = 100
epochs = 10
tolerancia = 1e-6

def f(x):
    A = 10
    p = len(x)
    somatorio = 0
    
    for i in range(p):
        somatorio += (x[i]**2) - A * np.cos(2 * np.pi * x[i])
    
    return somatorio

populacao_inicial = np.random.randint(low=0, high=2, size=(N, p * n_bits_var))
print(f"População inicial: {populacao_inicial}")

def decodifica_cromossomo(cromossomo, n_bits_per_var, intervalo=(-10, 10)):
    p = len(cromossomo) // n_bits_per_var
    variaveis = []
    
    for i in range(p):
        bits = cromossomo[i * n_bits_per_var:(i + 1) * n_bits_per_var]
        valor_binario = int(''.join(bits.astype(str)), 2)
        valor_real = intervalo[0] + (intervalo[1] - intervalo[0]) * (valor_binario / (2**n_bits_per_var - 1))
        variaveis.append(valor_real)
    
    return variaveis

cromossomo_exemplo = populacao_inicial[0]
variaveis_reais = decodifica_cromossomo(cromossomo_exemplo, n_bits_var)
print("Variáveis Reais do Cromossomo Exemplo:\n", variaveis_reais)

t = 0
melhor_aptidao = float('inf')
geracoes_sem_melhoria = 0

while t <= epochs:
    aptidoes = []
    
    for individuo in populacao_inicial:
        variaveis_reais = decodifica_cromossomo(cromossomo=individuo, n_bits_per_var=n_bits_var)
        aptidao = f(variaveis_reais)
        aptidoes.append(aptidao)
    
    melhor_aptidao_atual = min(aptidoes)
    
    if abs(melhor_aptidao_atual - melhor_aptidao) < tolerancia:
        geracoes_sem_melhoria += 1
    else:
        geracoes_sem_melhoria = 0
    
    if geracoes_sem_melhoria >= 3:
        print(f"Critério de parada atingido na geração {t}. Melhor aptidão: {melhor_aptidao_atual}")
        break
    
    melhor_aptidao = melhor_aptidao_atual
    
    t += 1