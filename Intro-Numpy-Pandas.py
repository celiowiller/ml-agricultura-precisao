# importar o recurso numpy para as operações com dados
import numpy as np # aqui, a lib numpy recebe um alias "apelido"

# criar uma lista com numeros
grupo = [[31, 25, 56], [4, 89, 102], [78, 875, 123]]

# neste passo, vamos cria uma matriz usando um recurso do numpy
# definir uma nova variavel para receber como valor o método/função matrix() - oriunda do numpy
umaMatriz = np.matrix(grupo) # a tarefa que o método/função matrix() cumpre é: transformar um conjunto de dados numa matriz
print('---------------------------------------------')
# verificar se o data type da variavel umaMatriz é, realmente, uma matriz
print(type(umaMatriz))
print('---------------------------------------------')
# verificar o formato/dimensoes da matriz
print(np.shape(umaMatriz))
print('---------------------------------------------')
print(np.mean(umaMatriz)) # usando o método/função mean() -> este é o recurso que calcula a média simples considerando todos os valores do conjunto/matriz
print('---------------------------------------------')

print()
print('================= ALGUMAS OPERAÇÕES COM MATRIZES =======================')

# agora, serão criadas duas matrizes - ambas receberão valores distintos
matriz1 = np.array([['2', '4'], ['5', '-6']]) # é uma função/método - com origem, especifica, no numpy usada para criar vetores/arrays
matriz2 = np.array([['9', '-4'], ['3', '5']])

# definir a operação de soma de matrizes 
matrizResultante = matriz1 + matriz2

# exibir o resultado
print(matrizResultante)

print()
print('================= OPERAÇÕES COM DADOS - NUMPY/PANDAS =======================')

# IMPORTAR O RECURSO PANDAS
import pandas as pd

# uma Series nada mais é do que umamatriz unidimensional, ou seja, de uma unica linha
umaSerie = pd.Series([1, 2, 3, np.nan, 6, 8, 'Ola'])

# acima, np.nan é o recurso que oferece a possibilidade de trabalhar com um lemento não numérico(not-a-number): origem no numpy
print(umaSerie) 

print()
print('---------------------- criar alguns dataframes ------------------------------')

# neste passo, será definida uma nova variavel para receber um conjunto de valores
algumasDatas = pd.date_range('20250506', periods = 6)
print(algumasDatas)
# date_range: os dois parametros estabelecem uma data inicial -> 2025-05-06 e o intervalo de valores que queremos estabelecer para o periodo -> periods = 6
# por padrão, com frequencia diaria -> freq='D'

# o resultado é este: 2025-05-06, 2025-05-07, 2025-05-08, ......

# neste passo, será definido o primeiro dataframe usando com recurso de indice a variavel algumasDatas

df1 = pd.DataFrame(np.random.randn(6, 4), index=algumasDatas, columns=list('ABCD'))

# o 1º argumentodo método acima é a definição do numero de linhas e colunas do df: df de 6 x 4 -> 6 linhas x 4 colunas

# o 2º argumento - index=algumasDatas - define qual será o recurso de indice posicional principal do df

# o 3º argumento - colums=list('ABCD') - define a quantidade de colunas do df
print()
print('este é o Dataframe 1')
print(df1)

print()
print('----------------------- DATAFRAME 2 --------------------------------')

# gerar um novo dataframe; será gerado a partir de um dicionario 
df2 = pd.DataFrame({
    'A': 1., # valor constante repetido para todas as linhas
    'B': pd.Timestamp('20250603'), # mesmo valor de data para todas as linhas
    'C': pd.Series(1, index=list(range(4)), dtype='float64'), # 1.0 Série com 4 elementos do tipo float
    'D': np.array([3.0] * 4, dtype='int32'), # 3 array com 4 valores inteiros [3,3,3,3]
    'E': pd.Categorical(['teste', 659, 'novo teste', 95]), # categorico com valores variados
    'F': 'esta é uma string' # mesma string para todas as linhas
})

print()
print('este é o Dataframe 2')
print(df2)

#-----------------------------------------------------------------------------------------

print()
print('---------------------- observando os contexto dos dataframes --------------------')
print()

# palavra reservada dtypes(data types) demonstra os tipos de dados que compõem os dfs
print('composição dos data types do df1')
print(df1.dtypes)
print('composição dos data types do df2')
print(df2.dtypes)

print()

# neste passo, vamos fazer uma "leitura" resumida dos dfs; para este proposito será utilizado o método/função head(): por padrão, o metodo head() lê as primeiras 5 linhas de qualquer df 
print('primeiras linhas do df1')
print(df1.head(2))
print('primeiras linhas do df2')
print(df2.head(2))
print()

# agora, vamos fazer uma nova 'leitura' só que será das ultimas linhas do df
# para este proposito será usada a função/metodo tail(): por padrão tail() lê as ultimas 3 linhas do df
print('ultimas linhas do df1')
print(df1.tail(2))
print('ultimas linhas do df2')
print(df2.tail(2))
print()

# vamos, neste passo, observe os elementos de indice de cada df
print('indices do df1')
print(df1.index)
print('indices do df2')
print(df2.index)
print()

# observar as colunas de cada df
print('colunas do df1')
print(df1.columns)
print('colunas do df2')
print(df2.columns)
print()

# uso do método/função describe()
print('resumo estatistico do df1')
print(df1.describe())
print()
print('resumo estatistico df2')
print(df2.describe())
print()

'''
count: contagem da qtde numero de valores NÃO NULOS do df (df1 = 6, df2 = 4)
mean: média aritmética dos valores que compõem o df
std: desvio padrão(medida da dispersão dos dados)
min: é o menor valor - valor minimo - de cada coluna do df
25%: 1º quartil(25% dos dados estão abaixo destes valores indicados em cada coluna)
50%  2º quartil(50% dos valores gera, a partir do df, a mediana)
75%  3º quartil(75% dos dados estão abaixo destes valores indicados em cada coluna)
max: é o maior valor - valor maximo - de cada coluna do df

'''

# vamos "converter" os dfs em arrays numpy
print('convertendo o df1 num array numpy')
print(df1.to_numpy()) 
print()
print('convertendo o df2 num array numpy')
print(df2.to_numpy()) 
print()

# observar a organização dos dfs a partir de valores de uma coluna especifica
print('Ordenando o df1 por uma coluna especifica')
print(df1.sort_values(by = 'C', ascending = False))
print()
print('Ordenando o df2 por uma coluna especifica')
print(df2.sort_values(by = 'E', ascending = False))
print()

# observar os eixos dos dfs
print('Ordenando o df1 de forma decrescente  - pelo eixo das linhas')
print(df1.sort_index(axis = 0, ascending = False))
print()
# sort_index: ordena o df pelo indice
# axis = 0: indicando que esta ordenação será feita pelas LINHAS do df
# ascending = False: a ordenação ocorrerá em ordem decrescente
print('Ordenando o df2 de forma decrescente  - pelo eixo das linhas')
print(df2.sort_index(axis = 0, ascending = False))
print()

# ------------------------------------------------------------------------------------

print('=============== OPERAÇÕES COM SELEÇÕES/FATIAMENTO DE DFs ===================')
print()

print('Fatiando df1')
print(df1['A'])
print()
print('Fatiando df2')
print(df2['D'])


# aplicar a seleção de um intervalo de valores a partir dos dfs
print('Fatiando df1 - via intervalo')
print(df1[1:3]) # intervalo semi-aberto [...[ 
print()
print('Fatiando df2')
print(df2[2:4]) # intervalo semi-aberto [...[ 


print('Fatiando df1')
print(df1['2025-05-07': '2025-05-10'])
print()
print('Fatiando df2')
print(df2[1:3])
