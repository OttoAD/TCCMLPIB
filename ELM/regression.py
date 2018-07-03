import numpy as np
import scipy as sp
import pandas as pd


class LinearRegression:
    """Classe que modela uma regressão linear multivariada."""

    def pseudoinverse(self,table):
        mult = table.T.dot(table)  # MULT = X^t * X
        inverse = pd.DataFrame(np.linalg.inv(mult), mult.columns, mult.index)# INVERSE = MULTIPLICATION^(-1)
        return inverse.dot(table.T) #INV * TRANSPOSE

    def linearmodel(self, table, target):
        """Aplica uma regressão linear multivariada e retorna um dataframe de pesos"""
        return self.pseudoinverse(table).dot(target)  # PSEUDOINVERSE * Y
    
    #REPENSAR AS FUNÇÕES ABAIXO
    def calculateIndex(self, table, percentage):
        """Calcula o indice da tabela relativo à porcentagem fornecida"""
        index = percentage*(table.shape[0])
        return np.round(index).astype(int)

    def trainModel(self, rowIndex, data, target):
        """Executa um modelo linear de regressão ao receber como argumentos os argumentos chave/valor da função referente ao modelo.
        O modelo é executado do início do dataframe até rowIndex."""
        return self.linearmodel(data.iloc[:rowIndex, :], target.iloc[:rowIndex, :])