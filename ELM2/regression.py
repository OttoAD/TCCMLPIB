import numpy as np
import pandas as pd
import cross_validation as cv
import data as dt

class LinearModel:
    """
    This class is a model for the multivariate linear regression.

    Methods
    ----------
    weights :
        A get method which returns the weights attribute
    pseudoinverse :
        A method which pseudoinverts a dataframe
    linearmodel :
        A method which calculates the weights dataframe for a linear regression
    """

    def __init__(self):
       self._weights = None

    def weights(self):
        """
        This method is a getter which returns the _weights attribute.

        Parameters
        ----------
        None

        Returns
        ----------
        A dataframe of weights

        """
        
        return self._weights

    def pseudoinverse(self, table):
        """
        This method computes the Moore-Penrose pseudo-inverse of a given dataframe.

        Parameters
        ----------
        table : 
            A dataframe to be inverted
        
        Returns
        ----------
        A new dataframe for the inverted input table

        """

        mult = table.T.dot(table)  # MULT = X^t * X
        inverse = pd.DataFrame(np.linalg.inv(mult), mult.columns, mult.index)# INVERSE = MULTIPLICATION^(-1)
        return inverse.dot(table.T) #INV * TRANSPOSE

    def train(self, table, target, train_percentage = 70):
        """
        This method computes a dataframe of weights for a linear regression given a training set and a testing set.

        Parameters
        ----------
        table:
            The training dataframe
        target:
            The testing dataframe, in other words, the table which contains the desired values to be forecast
    
        Returns
        ----------
        A dataframe of weights
        
        """
        if train_percentage == 100:
            index = table.shape[0]
        elif train_percentage > 0 and train_percentage < 100:
            index = (train_percentage*table.shape[0])//100
        self._weights = self.pseudoinverse(table.iloc[:index, :]).dot(target.iloc[:index, :])  # PSEUDOINVERSE * Y
    
    def test(self, table, test_percentage = 30):
        """
        
        """
        #index = (training_percentage*table.shape[0])//100
        #self.train(table.iloc[:index, :], target.iloc[:index, :])
        if test_percentage == 100:
            index = 0
        elif test_percentage > 0 and test_percentage < 100:
            index = table.shape[0] - ((test_percentage*table.shape[0])//100)
        
        return table.iloc[index:].dot(self._weights)
    
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
            error = np.append(error, data.mean_average_error(result, target_result).values)
        print("MULTIVARIATE LINEAR REGRESSION ERRORS:")
        print("Mean Average Error: " + str(np.mean(error).round(decimals = 3)) + " | Standard Deviation: " + str(np.std(error).round(decimals = 3)))