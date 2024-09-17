import numpy as np
import matplotlib.pyplot as plt

def f(x):
    A = 10
    p = len(x)
    somatorio = A * p
    for i in range(p):
        somatorio += (x[i]**2) - A * np.cos(2 * np.pi * x[i])
    return somatorio

def aptidao(x):
    return f(x) + 1

def inicializa_populacao(N, p, intervalo=(-10, 10)):
    return np.random.uniform(low=intervalo[0], high=intervalo[1], size=(N, p))

def selecao_por_torneio(populacao, aptidoes, tamanho_torneio=3):
    selecionados = []
    N = len(populacao)
    for _ in range(N):
        torneio = np.random.choice(N, tamanho_torneio, replace=False)
        vencedor = torneio[np.argmin(np.array(aptidoes)[torneio])]
        selecionados.append(populacao[vencedor])
    return np.array(selecionados)

def crossover_sbx(pai1, pai2, eta=15):
    u = np.random.uniform(size=pai1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (2 * (1 - u)) ** (-1 / (eta + 1)))
    filho1 = 0.5 * ((1 + beta) * pai1 + (1 - beta) * pai2)
    filho2 = 0.5 * ((1 - beta) * pai1 + (1 + beta) * pai2)
    return filho1, filho2

def aplicar_crossover(populacao, taxa_recombinacao):
    nova_populacao = []
    np.random.shuffle(populacao)
    
    for i in range(0, len(populacao) - 1, 2):
        pai1 = populacao[i]
        pai2 = populacao[i + 1]
        
        if np.random.rand() < taxa_recombinacao:
            filho1, filho2 = crossover_sbx(pai1, pai2)
        else:
            filho1, filho2 = pai1, pai2
        
        nova_populacao.append(filho1)
        nova_populacao.append(filho2)
    
    if len(populacao) % 2 != 0:
        nova_populacao.append(populacao[-1])
    
    return np.array(nova_populacao)

def mutacao_gaussiana(individuo, taxa_mutacao, intervalo=(-10, 10)):
    n = len(individuo)
    for i in range(n):
        if np.random.rand() < taxa_mutacao:
            individuo[i] += np.random.normal(0, 0.1)  # Média 0, desvio padrão 0.1
            individuo[i] = np.clip(individuo[i], intervalo[0], intervalo[1])  # Garantir que está dentro do intervalo
    return individuo

def aplicar_mutacao(populacao, taxa_mutacao):
    nova_populacao = []
    for individuo in populacao:
        individuo_mutado = mutacao_gaussiana(individuo.copy(), taxa_mutacao)
        nova_populacao.append(individuo_mutado)
    return np.array(nova_populacao)

def AlgoritmoGeneticoPontoFlutuante(N=100, p=20, epocas=10, taxa_mutacao=0.1, taxa_recombinacao=0.9, tolerancia=1e-6):
    t = 0
    melhor_aptidao = float('inf')
    geracoes_sem_melhoria = 0
    intervalo = (-10, 10)
    populacao_inicial = inicializa_populacao(N=N, p=p, intervalo=intervalo)
    
    while t <= epocas:
        # Avaliar a população
        aptidoes = [aptidao(individuo) for individuo in populacao_inicial]
        
        melhor_aptidao_atual = min(aptidoes)
        
        # Critério de parada baseado na melhoria
        if abs(melhor_aptidao_atual - melhor_aptidao) < tolerancia:
            geracoes_sem_melhoria += 1
        else:
            geracoes_sem_melhoria = 0
        
        if geracoes_sem_melhoria >= 3:
            print(f"Critério de parada atingido na geração {t}. Melhor aptidão: {melhor_aptidao_atual}")
            break
        
        melhor_aptidao = melhor_aptidao_atual
        
        # Seleção por torneio
        nova_populacao = selecao_por_torneio(populacao_inicial, aptidoes)
        
        # Aplicar recombinação (crossover SBX)
        nova_populacao = aplicar_crossover(nova_populacao, taxa_recombinacao)
        
        # Aplicar mutação gaussiana
        nova_populacao = aplicar_mutacao(nova_populacao, taxa_mutacao)
        
        populacao_inicial = nova_populacao
        
        t += 1
        print(f"Melhor aptidão da geração {t}: {melhor_aptidao_atual}")
    
    return melhor_aptidao
