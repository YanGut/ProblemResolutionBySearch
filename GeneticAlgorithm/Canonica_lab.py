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

def phi(cromossomo, intervalo=(-10, 10)):    
    valor_binario = int(''.join(cromossomo.astype(str)), 2)
    return intervalo[0] + (intervalo[1] - intervalo[0]) * (valor_binario / (2**len(cromossomo) - 1))

# Função para decodificar um cromossomo inteiro em várias variáveis reais
def decodifica_cromossomo(cromossomo, n_bits_per_var, intervalo=(-10, 10)):
    p = len(cromossomo) // n_bits_per_var  # Número de variáveis
    variaveis = []
    for i in range(p):
        # Para cada variável, pegamos um subconjunto de bits
        bits_var = cromossomo[i * n_bits_per_var:(i + 1) * n_bits_per_var]
        # Aplicamos a função phi para converter os bits em valor real
        valor_real = phi(bits_var, intervalo)
        variaveis.append(valor_real)
    return variaveis

def inicializa_populaca(N, p, nd):
    return np.random.randint(0, 2, size=(N, p * nd))

def roleta(C, probs):
    i = 0
    s = probs[i]
    r = np.random.uniform()
    
    while s < r and i < len(probs) - 1:  # Garantir que i não exceda o tamanho
        i += 1
        s += probs[i]
    
    return C[i, :]

def avaliar_populacao(C, nd):
    avaliacao = []
    for i in range(C.shape[0]):
        # Decodificar cromossomos e calcular aptidão
        variaveis_reais = decodifica_cromossomo(cromossomo=C[i, :], n_bits_per_var=nd)
        avaliacao.append(aptidao(variaveis_reais))
    return avaliacao

def selecao_por_roleta(C, aptidoes):
    total = np.sum(aptidoes)
    probs = [a / total for a in aptidoes]
    
    selecionados = np.empty((0, C.shape[1]), dtype=int)
    
    for _ in range(C.shape[0]):
        selecionado = roleta(C, probs).reshape(1, C.shape[1])
        selecionados = np.vstack((selecionados, selecionado))
    
    return selecionados

def crossover(pai1, pai2):
    ponto_de_crossover = np.random.randint(1, len(pai1) - 1)
    filho1 = np.concatenate([pai1[:ponto_de_crossover], pai2[ponto_de_crossover:]])
    filho2 = np.concatenate([pai2[:ponto_de_crossover], pai1[ponto_de_crossover:]])
    return filho1, filho2

def aplicar_crossover(populacao, taxa_recombinacao):
    nova_populacao = []
    np.random.shuffle(populacao)
    
    for i in range(0, len(populacao) - 1, 2):
        pai1 = populacao[i]
        pai2 = populacao[i+1]
        
        if np.random.rand() < taxa_recombinacao:
            filho1, filho2 = crossover(pai1=pai1, pai2=pai2)
        else:
            filho1, filho2 = pai1, pai2
    
        nova_populacao.append(filho1)
        nova_populacao.append(filho2)
    
    if len(populacao) % 2 != 0:
        nova_populacao.append(populacao[-1])
    
    return np.array(nova_populacao)

def mutacao(individuo, indice_mutacao):
    for i in range(len(individuo)):
        if np.random.rand() < indice_mutacao:
            individuo[i] = 1 - individuo[i]
    return individuo

def aplicar_mutacao(populacao, indice_mutacao):
    nova_populacao = []
    for individuo in populacao:
        individuo_mutado = mutacao(individuo=individuo.copy(), indice_mutacao=indice_mutacao)
        nova_populacao.append(individuo_mutado)
    return np.array(nova_populacao)

def AlgoritmoGeneticoCanonico(N=100, p=20, nd=8, epocas=10, taxa_mutacao=0.1, taxa_recombinacao=0.9, tolerancia=1e-6):
    t = 0
    melhor_aptidao = float('inf')
    geracoes_sem_melhoria = 0
    populacao_inicial = inicializa_populaca(N=N, p=p, nd=nd)
    
    while t <= epocas:
        # Avaliar a população
        aptidoes = avaliar_populacao(populacao_inicial, nd)
        
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
        
        nova_populacao = selecao_por_roleta(populacao_inicial, aptidoes)
        
        nova_populacao = aplicar_crossover(nova_populacao, taxa_recombinacao)
        
        nova_populacao = aplicar_mutacao(nova_populacao, taxa_mutacao)
        
        populacao_inicial = nova_populacao
        
        t += 1
        print(f"Melhor aptidão da geração {t}: {melhor_aptidao}")

    return melhor_aptidao