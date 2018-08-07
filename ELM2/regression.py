import numpy as np
import pandas as pd

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

    def linear_regression(self, table, target):
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

        return self.pseudoinverse(table).dot(target)  # PSEUDOINVERSE * Y