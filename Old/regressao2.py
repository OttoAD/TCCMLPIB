import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt


# ----- LEITURA DO ARQUIVO -----
def readData(fileName, index = None, separator = ",", dates = False, period = None):
    """Função que toma como argumento o nome do arquivo, a coluna de índice, o separador do csv, parsing de datas e o periodo das datas
    para realizar a leitura dos dados e retornar um dataframe"""
    if period == None:
        return pd.read_csv(fileName, index_col = index, sep = separator, parse_dates = dates)
    else:
        data = pd.read_csv(fileName, index_col = index, sep = separator, parse_dates = dates)
        data.index = data.index.to_period(period)
        return data

# ----- NORMALIZAÇÃO DOS DADOS -----
def normalize(table, skip = None):
    """Divide todos os valores de um dataframe por seus respectivos desvios padrões.
    O parâmetro skip informa quantas colunas da esquerda para a direita devem ser puladas"""
    if skip == None:
        return table.div(table.std())
    else:
        table.iloc[:,skip:] = table.iloc[:,skip:].div(table.iloc[:,skip:].std())
        return table

# ----- REGRESSÃO LINEAR MULTIVARIADA -----
def linearRegression(table,target):
    """Aplica uma regressão linear multivariada e retorna um dataframe de pesos"""
    mult = table.T.dot(table) #MULT = X^t * X
    inverse = pd.DataFrame(np.linalg.inv(mult),mult.columns,mult.index) #INVERSE = MULTIPLICATION^(-1)
    return (inverse.dot(table.T)).dot(target) #INV * TRANSPOSE * Y

# ----- CALCULO DE INDICE DA TABELA -----
def calculateIndex(table, percentage):
    index = percentage*(table.shape[0])
    return np.round(index).astype(int)

# ----- TREINAMENTO DO MODELO -----
def trainModel(model, rowIndex, **kwargs):
    """Executa um modelo ao receber como argumentos:
    1. O modelo em questão (a referência da função)
    2. Os argumentos chave/valor da função referente ao modelo
    O modelo é executado do início do dataframe até rowIndex."""
    if model.__name__ == "linearRegression":
        trainedModel = linearRegression(kwargs['data'].iloc[:rowIndex,:], kwargs['target'].iloc[:rowIndex,:])
 
    return trainedModel

# ----- ERRO ABSOLUTO MÉDIO -----
def averageAbsoluteError(data1, data2):
    """Calcula o erro absoluto médio entre dois conjuntos de valores"""
    return data1.subtract(data2).abs().mean()

# ----- PLOTAGEM DOS GRÁFICOS -----
def plot(originalData, predictedData):
    data = pd.concat([originalData, predictedData], axis=1)
    data.columns = ['Real','Estimado']
    chart = data.plot(title = 'PIB Real x Estimado',xticks=data.index,rot=45)
    chart.set_xlabel('Periodo (Trimestral)')
    chart.set_ylabel('PIB (%)')
    chart.grid(True, which='minor', axis='x' )
    chart.grid(True, which='major', axis='y' )
    plt.show()

def barPlot(originalData, predictedData):
    data = pd.concat([originalData, predictedData], axis=1)
    data.columns = ['Real','Estimado']
    chart = data.plot(title = 'PIB Real x Estimado', kind='bar')
    chart.set_ylabel('PIB (%)')
    plt.grid(alpha=0.3)
    chart.set_axisbelow(True)
    for values in chart.patches:
        chart.annotate(str(np.round(values.get_height(),2)), (values.get_x() * 1.01, values.get_height() * 1.015), ha='center', va='center', fontsize = 6)

    plt.show()


# ----- EXECUÇÃO -----
#Aqui pode mudar os parametros
indexTable = readData(fileName = "Dados2.csv", index = "Periodo", dates = True, period = "Q")
originalGDP = readData(fileName = "PIB.csv", index = "Periodo", dates = True, period = "Q")

indexTable = normalize(indexTable,skip = 1)

rowIndex = calculateIndex(indexTable, percentage = 0.75)

aleatorio = np.random.permutation(indexTable.index)
shuflledIndex = indexTable.reindex(aleatorio)
shuffledGDP = originalGDP.reindex(aleatorio)
#Embaralhamento dos dados (pra voltar é só colocar OriginalGDP e indexTable)

weights = trainModel(model = linearRegression, rowIndex = rowIndex,
                         data = shuflledIndex,
                         target = shuffledGDP)

predictedGDP = shuflledIndex.iloc[:rowIndex,:].dot(weights)

GDPError = averageAbsoluteError(predictedGDP.iloc[:rowIndex,:],shuffledGDP.iloc[:rowIndex,:])

print("\n----- MATRIZ DE PESOS -----\n" + str(weights))
print("\n----- PIB PREVISTO -----\n" + str(predictedGDP))
print("\nERRO:\n" + str(GDPError))

#barPlot(shuffledGDP.iloc[:rowIndex,:],predictedGDP)
#plot(originalGDP.iloc[:rowIndex,:],predictedGDP)

#shuffledGDP = shuffledGDP.sort_index()
#predictedGDP = predictedGDP.sort_index()
#print(predictedGDP)
#plot(shuffledGDP.iloc[:rowIndex,:],predictedGDP)