import numpy as np
import scipy as sp
import pandas as pd

class Data:
    """Classe que modela a importação e manipulação dos dados."""

    def __init__(self):
        self._permutation = [] #The permutation is set after the first shuffle

    def get_permutation(self):
        return self._permutation
    
    def set_permutation(self, permutation):
        self._permutation = permutation

    def readData(self, fileName, index = None, separator = ",", dates = False, period = None):
        """Método que toma como argumento o nome do arquivo, a coluna de índice, o separador do csv, parsing de datas e o periodo das datas para realizar a leitura dos dados e retornar um dataframe"""
        if period == None:
            return pd.read_csv(fileName, index_col = index, sep = separator, parse_dates = dates)
        else:
            data = pd.read_csv(fileName, index_col = index, sep = separator, parse_dates = dates)
            data.index = data.index.to_period(period)
            return data

    def normalize(self, table, begin = 0, end = 0):
        """Divide todos os valores de um dataframe por seus respectivos desvios padrões.
        O parâmetro skip informa quantas colunas da esquerda para a direita devem ser puladas"""
        
        if begin == 0 and end == 0:
            table = table.div(table.std())
        elif begin != 0 and end == 0:
            table.iloc[:,begin:] = table.iloc[:,begin:].div(table.iloc[:,begin:].std())
        elif end > begin:
            table.iloc[:,begin:end] = table.iloc[:,begin:end].div(table.iloc[:,begin:end].std())
        
        return table

    def shuffle(self, table):
        """ Mistura as linhas do dataframe de acordo com uma permutação realizada sobre os indices """
        if self._permutation == []:
            self._permutation = np.random.permutation(table.index)
        
        return table.reindex(self._permutation)
    
    def avgerror(self, table1, table2):
        """Calcula o erro absoluto médio entre dois conjuntos de valores"""
        return table1.subtract(table2).abs().mean()