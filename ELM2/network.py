import numpy as np
import pandas as pd
import cross_validation as cv
import data as dt

class NeuralNetwork:
    """
    Classe que modela uma rede neural artificial simples.
    Implementa uma Extreme Learning Machine (ELM) utilizando dos conceitos de uma Single Layer Feedforward Network (SLFN)
    """

    def __init__(self, neurons, C, weights = None, data = None):
        """
        Inicializador da classe que instancia os atributos de pesos e quantidade de neurônios
        """

        self._neurons = neurons
        self._beta = None
        self._C = C

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

    def C(self):
        return self._C
    
    def beta(self):
        """
        Método que retorna o valor do vetor Beta
        """
        return self._beta

    
    def generate_weights(self, table, num_neurons, average = 0, std_dev = 1):
        """
        Método que gera aleatoriamente uma matriz de pesos de acordo com uma distribuição normal com média 0 e desvio padrão 1.
        O tamanho da matriz é [numero de features, numero de neuronios]
        """
        
        num_features = table.shape[1]
        matrix_size = (num_neurons, num_features)
        return pd.DataFrame(np.random.normal(average, std_dev, size = matrix_size))
    def sigmoid(self, z):
        """
        Método que recebe um parâmetro Z e retorna o cálculo da função sigmoid (logística)
        """
        return 1.0/(1.0 + np.exp(-z))
    
    
    def feedforward(self, table):
        """
        Método que calcula a matriz H: Sigmoide(W*X)
        """
        H = self._weights.dot(table.T)
        return H.apply(self.sigmoid)
    
    def calculate_beta(self, H, target): #Beta = H^-1 * Target
        """shuffledGDP
        Método que recebe duas matrizes H e T e retorna o cálculo da matriz de pesos beta
        """
        #regularization = np.identity(target.shape[0])/(self._C)

        if self._neurons <= target.shape[0]: #if the number of neurons is lower or equal to the number of samples. The linear regression uses this one
        
            mult = H.T.dot(H)
            regularization = np.identity(mult.shape[0])/(self._C)
            mult = pd.DataFrame(regularization, mult.columns, mult.index).add(mult)
            mult_inverse = pd.DataFrame(np.linalg.inv(mult), mult.columns, mult.index)
            pseudo_inverse = mult_inverse.dot(H.T)

        # The commented line of code below does the exact shame thing as the chunk above:
        #pseudo_inverse = pd.DataFrame(np.linalg.pinv(H), H.columns, H.index)

        else: #else, if the number or neurons is higher than the number of samples
            mult = H.dot(H.T)
            regularization = np.identity(mult.shape[0])/(self._C)
            mult = pd.DataFrame(regularization, mult.columns, mult.index).add(mult)
            mult_inverse = pd.DataFrame(np.linalg.inv(mult), mult.columns, mult.index) #pvin works, inv doesn't
            pseudo_inverse = H.T.dot(mult_inverse)

        self._beta = pseudo_inverse.T.dot(target)
    
    def train(self, table, target, train_percentage = 70):
        """
        Método responsável por realizar o treinamento da rede neural
                 
        Parameters
        ----------
        table :
        percentage :

        Returns
        ----------
                    
        """
        if train_percentage == 100:
            index = table.shape[0]
        elif train_percentage > 0 and train_percentage < 100:
            index = (train_percentage*table.shape[0])//100
        #else erro
        H = self.feedforward(table = table.iloc[:index])
        self.calculate_beta(H = H, target = target.iloc[:index])
    
    def test(self, table, test_percentage = 30):
        """
        Método responsável por realizar o teste da rede neural
        
        Parameters
        ----------
        table :
        percentage :

        Returns
        ----------
                    
        """
        if test_percentage == 100:
            index = 0
        elif test_percentage > 0 and test_percentage < 100:
            index = table.shape[0] - ((test_percentage*table.shape[0])//100)

        H = self.feedforward(table.iloc[index:,:])
        return self._beta.T.dot(H)
    
    def validate(self, table, target, k_samples = 5):
        """"
        Parameters
        ----------
        table: a dataframe

        Returns
        -------
        
        """
        kval = cv.CrossValidation(k_samples)
        data = dt.Data()
        error = np.array([])
        for training, test, target_train, target_result in kval.KFold(table,target): #target2 é o target andando junto com o training
            self.train(training, target_train, train_percentage = 100)
            result = self.test(test, test_percentage = 100)
            #print(result.T)
            #print(target_result)
            #print(target2.shape) #questionar como fazer a comparação do valor previsto com o valor real
            error = np.append(error, data.mean_average_error(result.T, target_result).values)
            print(error)
            print(np.mean(error))