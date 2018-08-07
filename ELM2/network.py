import numpy as np
import pandas as pd
import cross_validation as cv

class NeuralNetwork:
    """
    Classe que modela uma rede neural artificial simples.
    Implementa uma Extreme Learning Machine (ELM) utilizando dos conceitos de uma Single Layer Feedforward Network (SLFN)
    """

    def __init__(self, neurons, weights = None, data = None):
        """
        Inicializador da classe que instancia os atributos de pesos e quantidade de neurônios
        """

        self._neurons = neurons
        self._beta = None

        if weights is None:
            self._weights = self.generate_weights(data, self._neurons)
            self._weights.columns = data.columns.values #Equalize the column names
        else:
            self._weights = weights

    
    def weights(self):
        """
        Método que retorna a matriz de pesos
        """
        return self._weights
    
    def neurons(self):
        """
        Método que retorna o número de neurônios
        """
        return self._neurons
    
    def beta(self):
        """
        Método que retorna o valor do vetor Beta
        """
        return self._beta

    def sigmoid(self, z):
        """
        Método que recebe um parâmetro Z e retorna o cálculo da função sigmoid (logística)
        """
        return 1.0/(1.0 + np.exp(-z))
    
    def generate_weights(self, table, num_neurons, average = 0, std_dev = 1):
        """
        Método que gera aleatoriamente uma matriz de pesos de acordo com uma distribuição normal com média 0 e desvio padrão 1.
        O tamanho da matriz é [numero de features, numero de neuronios]
        """
        
        num_features = table.shape[1]
        matrix_size = (num_neurons, num_features)
        return pd.DataFrame(np.random.normal(average, std_dev, size = matrix_size))
    
    
    def feedforward(self, table):
        """
        Método que calcula a matriz H: Sigmoide(W*X)
        """
        H = self._weights.dot(table.T)
        return H.apply(self.sigmoid)
    
    def calculate_beta(self, H, target): #Beta = H^-1 * Target
        """
        Método que recebe duas matrizes H e T e retorna o cálculo da matriz de pesos beta
        """
        pass
    
    def train(self, table, target, percentage = 0.7):
        """
        Método responsável por realizar o treinamento da rede neural
                 
        Parameters
        ----------
        table :
        percentage :

        Returns
        ----------
                    
        """
        pass
    
    def test(self, table, percentage = 0.3):
        """
        Método responsável por realizar o teste da rede neural
        
        Parameters
        ----------
        table :
        percentage :

        Returns
        ----------
                    
        """
        pass
    
    def train_test_split(self, table, train_size = 0.7, test_size = 0.3):
        """
        """
        
        pass
    
    def validate(self, table, target, k_samples = 5):
        """"
        Parameters
        ----------
        table: a dataframe

        Returns
        -------
        
        """
        pass