import numpy as np
import matplotlib.pyplot as plt
import time

def calcular_ataques(rainhas):
    """Calcula o número de pares de rainhas que se atacam."""
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
    """Função de aptidão: quanto maior, melhor. Máximo possível é 28 (nenhuma rainha atacando)."""
    return 28 - calcular_ataques(rainhas)

def perturb(rainhas):
    """Perturbação da solução atual alterando a posição de uma rainha em uma coluna aleatória."""
    nova_solucao = np.copy(rainhas)
    coluna = np.random.randint(low=0, high=len(rainhas))
    nova_solucao[coluna] = np.random.randint(0, 8)
    return nova_solucao

def definir_temperatura_inicial(rainhas_iniciais, num_simulacoes=100):
    """Define a temperatura inicial baseada na média de diferenças de aptidão entre soluções candidatas."""
    diffs = []
    
    for _ in range(num_simulacoes):
        rainhas_cand = perturb(rainhas_iniciais)
        f_inicial = f(rainhas_iniciais)
        f_cand = f(rainhas_cand)
        diffs.append(abs(f_cand - f_inicial))

    media_diff = np.mean(diffs)

    T0 = -media_diff / np.log(0.8)
    
    return T0

def tempera_simulada(tipo_resfriamento):
    """Executa a têmpera simulada com o tipo de resfriamento especificado."""
    tempo_inicial = time.time()
    
    it_max = 1000
    f_otimos = []
    solucoes_encontradas = []

    while len(solucoes_encontradas) < 92:
        rainhas_opt = np.random.randint(low=0, high=8, size=8)
        f_opt = f(rainhas_opt)
        
        T0 = definir_temperatura_inicial(rainhas_iniciais=rainhas_opt)
        T = T0
        i = 0
        
        while i < it_max:
            rainhas_cand = perturb(rainhas_opt)
            f_cand = f(rainhas_cand)

            if f_cand > f_opt or np.exp(-(f_opt - f_cand) / T) >= np.random.uniform(0, 1):
                rainhas_opt = rainhas_cand
                f_opt = f_cand

            i += 1
            f_otimos.append(f_opt)

            if tipo_resfriamento == 1:
                T *= 0.99  # Resfriamento simples multiplicativo
            elif tipo_resfriamento == 2:
                T = T / 1 + (0.99 * np.sqrt(T))  # Resfriamento adaptativo
            elif tipo_resfriamento == 3:
                delta_T = (T0 - T) / it_max  # Resfriamento linear
                T -= delta_T

            if f_opt == 28:
                # Verifica se a solução já foi encontrada anteriormente
                if not any(np.array_equal(rainhas_opt, sol) for sol in solucoes_encontradas):
                    solucoes_encontradas.append(np.copy(rainhas_opt))
                    print(f'Solução {len(solucoes_encontradas)} encontrada: {rainhas_opt}')
                break
        
        tempo_final = time.time()
        tempo_total = tempo_final - tempo_inicial
        
        print(f'\nTempo total de execução: {tempo_total:.2f} segundos')
        print(f'Número total de soluções distintas encontradas: {len(solucoes_encontradas)}')

    # Resultados finais
    print(f'Solução final: {rainhas_opt}')
    print(f'Número de pares de rainhas não atacantes: {f_opt}')

    # Gráfico de evolução do valor ótimo
    plt.plot(f_otimos)
    plt.xlabel('Iteração')
    plt.ylabel('f_opt (pares não atacantes)')
    plt.title('Evolução do valor ótimo')
    plt.show()

# Escolha qual tipo de resfriamento usar: 1, 2 ou 3
tipo_resfriamento = int(input("Escolha o tipo de resfriamento (1, 2 ou 3): "))
tempera_simulada(tipo_resfriamento)