import numpy as np
import scipy as sp
import pandas as pd

class Data:
    """Classe que modela a importação e manipulação dos dados."""

    def readData(self, fileName, index = None, separator = ",", dates = False, period = None):
        """Método que toma como argumento o nome do arquivo, a coluna de índice, o separador do csv, parsing de datas e o periodo das datas
        para realizar a leitura dos dados e retornar um dataframe"""
        if period == None:
            return pd.read_csv(fileName, index_col = index, sep = separator, parse_dates = dates)
        else:
            data = pd.read_csv(fileName, index_col = index, sep = separator, parse_dates = dates)
            data.index = data.index.to_period(period)
            return data
    
    def normalize(self, table, skip = None):
        """Divide todos os valores de um dataframe por seus respectivos desvios padrões.
        O parâmetro skip informa quantas colunas da esquerda para a direita devem ser puladas"""
        if skip == None:
            return table.div(table.std())
        else:
            table.iloc[:,skip:] = table.iloc[:,skip:].div(table.iloc[:,skip:].std())
            return table
