import numpy as np
import pandas as pd
from Canonica_lab import AlgoritmoGeneticoCanonico
from Flutuante_lab import AlgoritmoGeneticoPontoFlutuante

N = 100  # número de indivíduos
p = 20  # dimensões
nd = 8
epocas = 100
indice_de_mutacao = 0.1
indice_de_recombinacao = 0.85
tamanho_do_torneio = 10

# Função para rodar o algoritmo e coletar o melhor valor de aptidão
def run_algorithm(algoritmo, N, p, nd=None):
    resultados = []
    for _ in range(100):
        if algoritmo == "Canonico":
            resultado = AlgoritmoGeneticoCanonico(
                N=N, 
                p=p, 
                nd=nd, 
                epocas=epocas, 
                taxa_mutacao=indice_de_mutacao, 
                taxa_recombinacao=indice_de_recombinacao)
        elif algoritmo == "Flutuante":
            resultado = AlgoritmoGeneticoPontoFlutuante(
                N=N, 
                p=p, 
                epocas=epocas, 
                taxa_mutacao=indice_de_mutacao, 
                taxa_recombinacao=indice_de_recombinacao)
        resultados.append(resultado)
    return resultados

# Executar os algoritmos e coletar os resultados
resultados_canonico = run_algorithm("Canonico", N, p, nd)
resultados_flutuante = run_algorithm("Flutuante", N, p)

# Função para calcular estatísticas
def calcular_estatisticas(resultados):
    menor = np.min(resultados)
    maior = np.max(resultados)
    media = np.mean(resultados)
    desvio_padrao = np.std(resultados)
    return menor, maior, media, desvio_padrao

# Calcular estatísticas para os dois algoritmos
estatisticas_canonico = calcular_estatisticas(resultados_canonico)
estatisticas_flutuante = calcular_estatisticas(resultados_flutuante)

# Criar tabela com os resultados
tabela_comparativa = pd.DataFrame({
    "Método": ["Algoritmo Genético Canônico", "Algoritmo Genético em Ponto Flutuante"],
    "Menor Aptidão": [estatisticas_canonico[0], estatisticas_flutuante[0]],
    "Maior Aptidão": [estatisticas_canonico[1], estatisticas_flutuante[1]],
    "Média Aptidão": [estatisticas_canonico[2], estatisticas_flutuante[2]],
    "Desvio Padrão Aptidão": [estatisticas_canonico[3], estatisticas_flutuante[3]]
})

print(tabela_comparativa)
