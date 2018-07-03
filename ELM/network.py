import numpy as np
import scipy as sp
import pandas as pd


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
        self._beta = None

        if weights is None:
            self._weights = self.generate_weights(data, self._neurons)
            self._weights.columns = data.columns.values #Equalize the column names
        else:
            self._weights = weights

    
    def weights(self):
        """Método que retorna a matriz de pesos"""
        return self._weights
    
    def neurons(self):
        """Método que retorna o número de neurônios"""
        return self._neurons
    
    def beta(self):
        """Método que retorna o valor do vetor Beta"""
        return self._beta
    
    def generate_weights(self, table, num_neurons, average = 0, std_dev = 1):
        """Método que gera aleatoriamente uma matriz de pesos de acordo com uma distribuição normal com média 0 e desvio padrão 1.
        O tamanho da matriz é [numero de features, numero de neuronios]"""
        
        if table is None:
            raise ValueError("Os dados de entrada não foram informados")
        
        num_features = table.shape[1]
        matrix_size = (num_neurons, num_features)
        return pd.DataFrame(np.random.normal(average, std_dev, size = matrix_size)) 

    def sigmoid(self, z):
        """Método que recebe um parâmetro Z e retorna o cálculo da função sigmoid (logística)"""
        return 1.0/(1.0 + np.exp(-z))

    def feedforward(self, table):
        """Método que calcula a matriz H: Sigmoide(W*X)"""
        H = self._weights.dot(table.T)
        return H.apply(self.sigmoid)
    
    def calculate_beta(self, H, target): #Beta = H^-1 * Target
        """Método que recebe duas matrizes H e T e retorna o cálculo da matriz de pesos beta"""
        #if self._neurons <= target.shape[0]: #if the number of neurons is lower or equal to the number of samples. The linear regression uses this one
        mult = H.T.dot(H)
        mult_inverse = pd.DataFrame(np.linalg.inv(mult), mult.columns, mult.index)
        pseudo_inverse = mult_inverse.dot(H.T)
    
        # else:    #else, if the number or neurons is higher than the number of samples
        #     mult = H.dot(H.T)
        #     mult_inverse = pd.DataFrame(np.linalg.inv(mult), mult.columns, mult.index)
        #     pseudo_inverse = H.T.dot(mult_inverse)

        self._beta = pseudo_inverse.T.dot(target)

    def train(self, table, target, percentage = 0.7):
        """Método responsável por realizar o treinamento da rede neural """
        if percentage == 1:
            index = table.shape[0]
        else:
            index = np.round(percentage*(table.shape[0])).astype(int)
        H = self.feedforward(table.iloc[:index,:])
        self.calculate_beta(H,target.iloc[:index,:])

    def validate(self, table, percentage = 0.3):
        """Método responsável por realizar a validação da rede neural"""
        if percentage == 1:
            index = 0
        else:
            index = table.shape[0] - np.round(percentage*(table.shape[0])).astype(int)

        H = self.feedforward(table.iloc[index:,:])
        return self._beta.T.dot(H)