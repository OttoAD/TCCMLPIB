import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

# ----- LEITURA DO ARQUIVO -----
def importCSVData(fileName,index=None,separator=",",dates=False):
    """Função que toma como argumento o nome do arquivo, a coluna de índice, o separador do csv e parsing de datas
    para realizar a leitura dos dados e retornar um dataframe com a função read_csv do pacote pandas"""
    return pd.read_csv(fileName,index_col=index,sep=separator,parse_dates=dates)

def importExcelData(fileName): #Função incompleta, ainda faltam os parses das planilhas
    return pd.ExcelFile(fileName)

# ----- NORMALIZAÇÃO DOS DADOS -----
def normalizeData(dataTable):
    """Função que toma como argumento um dataframe a ser normalizado(dividir todas as observações
    pelo desvio padrão associado a cada variável) e retorna um dataframe"""
    standardDeviation = dataTable.std()
    return dataTable.div(standardDeviation)

# ----- REGRESSÃO LINEAR MULTIVARIADA -----
def linearRegression(dataTable,target):
    """Função que toma como argumentos dois dataframes como insumo de uma regressão linear multivariada
    e retorna um dataframe com os pesos calculados"""
    mult = dataTable.T.dot(dataTable) #MULTIPLICATION = X^t * X
    inverse = pd.DataFrame(np.linalg.inv(mult),mult.columns,mult.index) #INVERSE = MULTIPLICATION^(-1)
    return (inverse.dot(dataTable.T)).dot(target) #INV * TRANSPOSE * Y

# ----- ERRO ABSOLUTO MÉDIO -----
def averageAbsoluteError(table1,table2):
    """Função que toma como argumentos duas tabelas(ou valores) e calcula o erro absoluto médio"""
    return table1.subtract(table2).abs().mean()

# ----- EXECUÇÃO -----
indexes = importCSVData(fileName="Dados2.csv",index="Periodo",dates=True)
indexes.iloc[:,1:] = normalizeData(indexes.iloc[:,1:])
originalGDP = importCSVData("PIB.csv",index="Periodo",dates=True)
weightTable = linearRegression(indexes.iloc[:66,:],originalGDP.iloc[:66,:])
predictedGDP = indexes.iloc[67:,:].dot(weightTable)
predictedGDP.index = predictedGDP.index.to_period("Q")
originalGDP.index = originalGDP.index.to_period("Q")
GDPError = averageAbsoluteError(predictedGDP.iloc[67:,:],originalGDP.iloc[67:,:])

# ----- PLOTAGEM DOS GRÁFICOS -----
data = pd.concat([originalGDP.iloc[67:,:],predictedGDP],axis=1)
data.columns = ['Real','Estimado']
chart = data.plot(title = 'PIB Real x Estimado',xticks=data.index,rot=45)
chart.set_xlabel('Periodo (Trimestral)')
chart.set_ylabel('PIB (%)')
chart.grid(True, which='minor', axis='x' )
chart.grid(True, which='major', axis='y' )

print("\n----- MATRIZ DE PESOS -----\n"+str(weightTable))
print("\n----- PIBS -----\n"+str(data))
print("\nERRO:\n"+str(GDPError))
plt.show()