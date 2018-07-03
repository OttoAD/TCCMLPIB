import numpy as np
import scipy as sp
import pandas as pd
import regression as lr


class NeuralNetwork:
    """Classe que modela uma rede neural artificial simples.
    Implementa uma Extreme Learning Machine (ELM) utilizando dos conceitos de uma Single Layer Feedforward Network (SLFN)"""

    def __init__(self, neurons, weights = None, data = None):
        """Inicializador da classe que instancia os atributos de pesos e quantidade de neurônios"""
        if str(neurons).isalpha():
            raise ValueError("Neurons deve ser um valor numérico")

        if neurons <= 0:
            raise ValueError("Quantidade insuficiente de neurônios: '{}'".format(neurons))
        
        self._neurons = neurons
        self._Beta = None

        if (weights is None):
            self._weights = self.generate_weights(data, self._neurons)
            self._weights.columns = data.columns.values #Iguala o nome das colunas
        else:
            self._weights = weights

    
    def weights(self):
        """Método que retorna a matriz de pesos"""
        return self._weights
    
    def neurons(self):
        """Método que retorna o número de neurônios"""
        return self._neurons
    
    def Beta(self):
        """Método que retorna o valor do vetor Beta"""
        return self._Beta
    
    def set_Beta(self,B):
        self._Beta = B

    def generate_weights(self, data, n_neurons):
        """Método que gera aleatoriamente uma matriz de pesos de acordo com uma distribuição normal com média 0 e desvio padrão 1.
        O tamanho da matriz é [numero de features, numero de neuronios]"""
        
        if data is None:
            raise ValueError("Os dados de entrada não foram informados")
        
        n_features = data.shape[1]
        size = (n_neurons,n_features)
        return pd.DataFrame(np.random.normal(0, 1, size = size)) 

    def sigmoid(self, z):
        """Método que recebe um parâmetro Z e retorna o cálculo da Método sigmoid (Método logística)"""
        return 1.0/(1.0 + np.exp(-z))

    def feedforward(self, data):
        """Método que calcula a matriz H: Sigmoide(W*X)"""
        H = self._weights.dot(data.T)
        return H.apply(self.sigmoid)
    
    # def calculateBeta(self, H, data):
    #     """Método que recebe duas matrizes H e T e retorna o cálculo do peso beta.
    #    # Sendo que BETA é a pseudo-inversa da matriz de pesos multiplicada pela matriz alvo"""
    #     regressionModel = lr.LinearRegression()
    #     Hinverse = regressionModel.pseudoInverse(H)
    #     Beta = Hinverse.T.dot(data)
    #     return Beta